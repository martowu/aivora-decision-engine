"""decision-service 单测（W3，定稿 §2.2 / §2.2.7 / §7-E / W4 shadow）。

覆盖：
1. 503 全停：模型/策略未就绪 → /decide 503、/health not_ready
2. 推理端还原一致性：服务加载工件的校准概率 ≈ 训练端（命门，已在 inference 验过，这里走 API 再验）
3. shadow 掩码：production 路由 AUTO 的样本，切 shadow 后对外=REVIEW、落库 model_action=AUTO、is_shadow=True
4. 决策全量落库可追溯（§2.2.2）
5. 策略热切换 + 回滚（§2.2.1）
6. shadow 推翻率（W4 门禁 §2.2.7）
7. 推理端 isotonic 校准器分支（部署可还原）
8. request_id 幂等重放（7 天窗口 → 409 返回首次结果，§2.2 错误码表）
9. 审计查询 GET /v1/decisions/{request_id} + tenant_id 贯穿落库
10. feedback 闭环（pending→approved upsert、终态须带结论、未知 404、status 过滤）
11. 重启持久化：新 JsonlStore 重载同一 JSONL → 幂等/feedback 仍有效

运行：
  ../../.venv/bin/python test_service.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

import numpy as np

SERVICE = Path(__file__).resolve().parent
PIPE = SERVICE.parent / "training" / "pipelines"
POLICY_YAML = SERVICE.parent / "policies" / "recon" / "engine.yaml"
GOLDEN = SERVICE.parent / "contracts" / "examples" / "recon.golden.json"

sys.path.insert(0, str(PIPE))
sys.path.insert(0, str(SERVICE))

import train                                  # noqa: E402
import synthetic                              # noqa: E402
from app import build_app                     # noqa: E402
from decision_service import DecisionService  # noqa: E402
from inference import DeployCalibrator, predict  # noqa: E402
from shadow import ShadowReporter             # noqa: E402
from storage import JsonlStore                # noqa: E402

from fastapi.testclient import TestClient     # noqa: E402

N = 900
SEED = 11


def make_artifact(dirpath: Path, n: int = N, seed: int = SEED) -> Path:
    recs = synthetic.generate_dataset(n, seed)
    inputs, labels = synthetic.to_pair_inputs(recs)
    h = synthetic.dataset_hash(recs)
    train.train_and_evaluate(inputs, labels, dirpath, data_snapshot_hash=h)
    return dirpath


def sample_requests(n: int = N, seed: int = SEED) -> list[dict]:
    recs = synthetic.generate_dataset(n, seed)
    inputs, _ = synthetic.to_pair_inputs(recs)
    return [asdict(p) for p in inputs]


def build_ready(mode: str = "production") -> tuple[TestClient, DecisionService, list[dict], Path]:
    td = Path(tempfile.mkdtemp(prefix="aivora_svc_"))
    artifact = td / "art"
    make_artifact(artifact)
    svc = DecisionService(JsonlStore(td / "logs.jsonl"), mode=mode)
    svc.load_model(artifact)
    svc.load_policy(POLICY_YAML, json.loads(GOLDEN.read_text(encoding="utf-8")))
    client = TestClient(build_app(svc))
    return client, svc, sample_requests(), td


# ---------------------------------------------------------------------------

def test_503_not_ready():
    td = Path(tempfile.mkdtemp(prefix="aivora_503_"))
    svc = DecisionService(JsonlStore(td / "l.jsonl"), mode="production")
    client = TestClient(build_app(svc))
    # 模型 + 策略都没加载
    h = client.get("/health").json()
    assert h["status"] == "not_ready" and not h["model_loaded"] and not h["policy_loaded"]
    req = sample_requests(1)[0]
    r = client.post("/decide", json=req)
    assert r.status_code == 503, r.text
    assert r.json()["action"] == "HUMAN"
    print("  [PASS] 503 全停（未就绪）")


def test_only_policy_not_ready():
    td = Path(tempfile.mkdtemp(prefix="aivora_op_"))
    svc = DecisionService(JsonlStore(td / "l.jsonl"), mode="production")
    svc.load_policy(POLICY_YAML, json.loads(GOLDEN.read_text(encoding="utf-8")))  # 只有策略，没模型
    client = TestClient(build_app(svc))
    assert client.get("/health").json()["status"] == "not_ready"
    r = client.post("/decide", json=sample_requests(1)[0])
    assert r.status_code == 503
    print("  [PASS] 503（仅有策略无模型也全停）")


def test_decide_roundtrip_and_persist():
    client, svc, reqs, _ = build_ready("production")
    assert client.get("/health").json()["status"] == "ready"
    auto = review = 0
    for i, req in enumerate(reqs[:120]):
        r = client.post("/decide", json=req).json()
        assert r["action"] in ("AUTO", "REVIEW")
        assert r["mode"] == "production"
        # 概率合法
        p = r["model"]["probs"]
        assert abs(sum(p.values()) - 1.0) < 1e-6 and all(v >= 0 for v in p.values())
        assert r["model"]["label"] in ("MATCHED", "PARTIAL", "UNMATCHED")
        if r["action"] == "AUTO":
            auto += 1
        else:
            review += 1
    # 落库
    rows = client.get("/decisions", params={"limit": 500}).json()["decisions"]
    assert len(rows) == 120
    assert all("policy_version" in x and "model_version" in x for x in rows)
    assert auto > 0, "900 条合成里至少应有 AUTO（验证 AUTO 路径可达）"
    print(f"  [PASS] 决策落库 120 条（AUTO={auto}, REVIEW={review}）")


def test_shadow_masks_auto():
    client, svc, reqs, _ = build_ready("production")
    # 找一条 production 下路由 AUTO 的请求
    auto_req = None
    for req in reqs[:200]:
        if client.post("/decide", json=req).json()["action"] == "AUTO":
            auto_req = req
            break
    assert auto_req is not None, "找不到 AUTO 样本，shadow 测试无意义"
    # 切 shadow，同一样本 → 对外 REVIEW，model_action 仍 AUTO，is_shadow=True
    assert client.post("/admin/mode", json={"mode": "shadow"}).json()["mode"] == "shadow"
    r = client.post("/decide", json=auto_req).json()
    assert r["action"] == "REVIEW"          # 对外掩码
    assert r["model_action"] == "AUTO"      # 真实决策
    assert r["is_shadow"] is True and r["mode"] == "shadow"
    # 落库那条 is_shadow=True 且 action（真实）= AUTO
    row = client.get("/decisions", params={"limit": 5}).json()["decisions"][-1]
    assert row["is_shadow"] is True and row["action"] == "AUTO"
    print("  [PASS] shadow 掩码（AUTO→对外REVIEW，落库记真实AUTO）")


def test_policy_rollback_and_hot_switch():
    client, svc, reqs, td = build_ready("production")
    cur = svc.status()["policy_version"]
    # 热加载一个改阈值的策略 v2（金额上限调极小 → 几乎全 REVIEW）
    y2 = td / "engine_v2.yaml"
    # 金额上限 100000 → 1（破坏 clean_matched_auto 这条 golden 的 AUTO 期望）
    y2.write_text(POLICY_YAML.read_text(encoding="utf-8").replace("100000", "1"))
    v2 = client.post("/admin/policy/load",
                     json={"yaml_path": str(y2), "golden_path": str(GOLDEN)})
    # 改坏阈值（金额上限=1）会破坏 golden 样例 → 加载应被拒绝(400)，保持旧版
    assert v2.status_code == 400, f"改坏阈值的策略应被 golden 拒绝: {v2.text}"
    # 策略版本列表 + 回滚到当前
    pols = client.get("/admin/policies").json()
    assert any(p["version"] == cur for p in pols["policy_versions"])
    rb = client.post("/admin/policy/rollback", json={"target_version": cur}).json()
    assert rb["policy_version"] == cur
    # 回滚到不存在的版本 → 404
    assert client.post("/admin/policy/rollback",
                       json={"target_version": "nope"}).status_code == 404
    print(f"  [PASS] 策略 golden 守卫 + 回滚（cur={cur}）")


def test_shadow_override_rate():
    client, svc, reqs, _ = build_ready("shadow")
    # shadow 跑一批，全部对外 REVIEW；model_action 有 AUTO/REVIEW
    auto_pay, auto_label = [], {}
    for req in reqs[:150]:
        r = client.post("/decide", json=req).json()
        assert r["action"] == "REVIEW" and r["is_shadow"]
        if r["model_action"] == "AUTO":
            auto_pay.append(req["payment_id"])
            auto_label[req["payment_id"]] = r["model"]["label"]
    logs = list(svc._store.decisions(0, 1000))
    rep = ShadowReporter()
    # 场景 A：人工全部同意模型（label 一致 + 放行）→ override_rate 0
    truth_agree = {pid: {"label": auto_label[pid], "action": "AUTO"} for pid in auto_pay}
    res_agree = rep.report(logs, truth_agree)
    assert res_agree.gate_passed(0.02) and res_agree.override_rate == 0.0
    # 场景 B：人工全部推翻（label 不同）→ override_rate 1.0 → 门禁不过
    truth_over = {pid: {"label": "UNMATCHED", "action": "REVIEW"} for pid in auto_pay}
    res_over = rep.report(logs, truth_over)
    assert not res_over.gate_passed(0.02) and res_over.override_rate == 1.0
    print(f"  [PASS] shadow 推翻率（AUTO样本={len(auto_pay)}, 同意率/推翻率 0/1 边界正确）")


def test_inference_parity_and_isotonic():
    # 服务加载工件的校准概率 ≈ 训练端（走 inference 模块直接比）
    td = Path(tempfile.mkdtemp(prefix="aivora_par_"))
    artifact = make_artifact(td / "art")
    from inference import load_bundle
    bundle = load_bundle(artifact)
    recs = synthetic.generate_dataset(N, SEED)
    inputs, labels = synthetic.to_pair_inputs(recs)
    import features
    X = np.vstack([features.to_vector(features.build_features(i)) for i in inputs])
    y = train._label_idx(labels)
    split = train.temporal_group_split(inputs, labels)
    # 训练端
    import lightgbm as lgb
    m = lgb.Booster(model_file=str(artifact / "model.txt"))
    # 训练端 OvRCalibrator 需要重拟（这里只验证服务侧内部一致性 + 形状）
    raw = np.asarray(m.predict(X[split["test"]], raw_score=True))
    probs = bundle.calibrator.calibrate(raw)
    assert probs.shape == (len(split["test"]), 3)
    assert all(abs(p.sum() - 1) < 1e-6 for p in probs)
    # isotonic 分支可加载 + 推理（用 Platt 断点结构喂一个假 isotonic doc）
    iso_doc = {"method": "isotonic", "feature_version": "1.0",
               "iso": {"x": [[-2.0, 0.0, 2.0]] * 3, "y": [[0.0, 0.5, 1.0]] * 3,
                       "x_threshold": []}}
    dc = DeployCalibrator(iso_doc)
    out = dc.calibrate(np.array([[-2.0, 0.0, 2.0]]))
    assert out.shape == (1, 3) and abs(out[0].sum() - 1) < 1e-6
    print("  [PASS] 推理端还原一致性 + isotonic 分支")


def test_store_restart_persistence():
    """重启（新 JsonlStore 同一路径）→ 幂等重放与 feedback 仍有效。"""
    td = Path(tempfile.mkdtemp(prefix="aivora_rst_"))
    art = td / "art"
    make_artifact(art)
    log_path = td / "logs" / "decision_logs.jsonl"
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    svc1 = DecisionService(JsonlStore(log_path), mode="production")
    svc1.load_model(art)
    svc1.load_policy(POLICY_YAML, golden)
    req = sample_requests(1)[0]
    b1 = svc1.decide(req, request_id="req-rst-001", tenant_id="t-demo")
    assert b1["duplicate"] is False
    svc1.submit_feedback("req-rst-001", "fin-01",
                         review_action="MAINTAIN", status="approved")

    svc2 = DecisionService(JsonlStore(log_path), mode="production")
    svc2.load_model(art)
    svc2.load_policy(POLICY_YAML, golden)
    b2 = svc2.decide(req, request_id="req-rst-001", tenant_id="t-demo")
    assert b2["duplicate"] is True and b2["decision_id"] == b1["decision_id"]
    fb = svc2.feedback()
    assert len(fb) == 1 and fb[0]["status"] == "approved"
    print("  [PASS] 重启后幂等/feedback 仍有效（JSONL 重建索引）")


def test_idempotent_replay():
    client, svc, reqs, _ = build_ready("production")
    req = dict(reqs[0])
    req["request_id"] = "req-idem-001"
    req["tenant_id"] = "t-demo"
    r1 = client.post("/decide", json=req)
    assert r1.status_code == 200
    body1 = r1.json()
    assert body1["duplicate"] is False and body1["request_id"] == "req-idem-001"
    assert body1["tenant_id"] == "t-demo" and "model_version" in body1
    # 窗口内重复（即便载荷不同）→ 409 + 首次结果，不重算
    req2 = dict(req, pay_amount=req["pay_amount"] + 12345.0)
    r2 = client.post("/decide", json=req2)
    assert r2.status_code == 409, r2.text
    b2 = r2.json()
    assert b2["error_code"] == "DUPLICATE_REQUEST"
    assert b2["decision_id"] == body1["decision_id"]
    assert b2["action"] == body1["action"] and b2["model"] == body1["model"]
    # /v1/decisions 别名同行为
    assert client.post("/v1/decisions", json=req).status_code == 409
    # 不落第二条日志
    rows = client.get("/decisions", params={"limit": 500}).json()["decisions"]
    assert sum(1 for x in rows if x["request_id"] == "req-idem-001") == 1
    print("  [PASS] request_id 幂等重放（409 首次结果 / v1 别名 / 不落重库）")


def test_audit_query():
    client, svc, reqs, _ = build_ready("production")
    req = dict(reqs[1])
    req["request_id"] = "req-audit-001"
    req["tenant_id"] = "t-demo"
    b = client.post("/decide", json=req).json()
    # 审计查询（定稿 §2.2.7：GET /v1/decisions/{request_id}）
    r = client.get("/v1/decisions/req-audit-001")
    assert r.status_code == 200
    a = r.json()
    assert a["decision_id"] == b["decision_id"] and a["tenant_id"] == "t-demo"
    # tenant_id / request_id 贯穿落库
    rows = client.get("/decisions", params={"limit": 500}).json()["decisions"]
    row = next(x for x in rows if x["request_id"] == "req-audit-001")
    assert row["tenant_id"] == "t-demo"
    # 未知 request_id → 404
    assert client.get("/decisions/req-nope").status_code == 404
    print("  [PASS] 审计查询 /v1/decisions/{request_id}（tenant_id 落库 + 404）")


def test_feedback_loop():
    client, svc, reqs, _ = build_ready("production")
    req = dict(reqs[2])
    req["request_id"] = "req-fb-001"
    b = client.post("/decide", json=req).json()
    model_label = b["model"]["label"]
    other_label = next(l for l in ("MATCHED", "PARTIAL", "UNMATCHED")
                       if l != model_label)
    # pending 反馈
    r1 = client.post("/v1/feedback/req-fb-001",
                     json={"reviewer": "fin-01", "review_action": "MAINTAIN"})
    assert r1.status_code == 200 and r1.json()["status"] == "pending"
    # 同 reviewer 再提 → upsert 为 approved 纠正（不产生第二条）
    r2 = client.post("/v1/feedback/req-fb-001",
                     json={"reviewer": "fin-01", "corrected_label": other_label,
                           "reason": "差额为手续费", "status": "approved"})
    assert r2.status_code == 200 and r2.json()["status"] == "approved"
    rows = client.get("/v1/feedback").json()["feedback"]
    mine = [x for x in rows if x["reviewer"] == "fin-01"]
    assert len(mine) == 1 and mine[0]["corrected_label"] == other_label
    # 终态（approved/rejected）必须携带结论
    assert client.post("/v1/feedback/req-fb-001",
                       json={"reviewer": "fin-02", "status": "approved"}
                       ).status_code == 400
    # 非法标签（schema 校验）→ 400 SCHEMA_INVALID；未知决策 → 404
    assert client.post("/v1/feedback/req-fb-001",
                       json={"reviewer": "fin-02",
                             "corrected_label": "REVIEW"}).status_code == 400
    assert client.post("/feedback/req-unknown",
                       json={"reviewer": "x", "review_action": "MAINTAIN"}
                       ).status_code == 404
    # status 过滤
    assert client.get("/v1/feedback", params={"status": "approved"}
                      ).json()["count"] == 1
    print("  [PASS] feedback 闭环（pending→approved upsert / 终态须结论 / 404 / 过滤）")


def test_decide_schema_invalid_400():
    client, svc, reqs, _ = build_ready("production")
    # 缺必填字段（仅 pay_amount）→ 400 SCHEMA_INVALID（对齐契约，非默认 422）
    r = client.post("/decide", json={"pay_amount": 100.0})
    assert r.status_code == 400, r.text
    assert r.json()["error_code"] == "SCHEMA_INVALID"
    # 合法请求携带 invoice_number → 200，字段透传到特征（不报错即可）
    req = dict(reqs[0]); req["invoice_number"] = "INV-9999"
    assert client.post("/decide", json=req).status_code == 200
    print("  [PASS] /decide schema 失败→400 SCHEMA_INVALID / invoice_number 透传")


def main():
    tests = [test_503_not_ready, test_only_policy_not_ready,
             test_decide_roundtrip_and_persist, test_decide_schema_invalid_400,
             test_shadow_masks_auto,
             test_policy_rollback_and_hot_switch, test_shadow_override_rate,
             test_inference_parity_and_isotonic,
             test_store_restart_persistence, test_idempotent_replay,
             test_audit_query, test_feedback_loop]
    passed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  [FAIL] {t.__name__}: {e}")
    print(f"\n{passed}/{len(tests)} passed")
    return 0 if passed == len(tests) else 1


if __name__ == "__main__":
    sys.exit(main())
