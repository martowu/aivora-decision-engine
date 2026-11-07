"""PostgresStore 集成测试（W3 交付物，infra/migrations 0001 DDL）。

与 JsonlStore 同接口的生产后端，对**真实 Postgres 15** 验证：
1. 五表 DDL 已建 + recon 任务注册
2. 决策落库往返（字段保真：probs dict / is_shadow / input_digest / latency_ms）
3. request_id 幂等（重复 append → 首次结果保持，ON CONFLICT 不报错）
4. decisions(since_ts, limit) 过滤与升序
5. feedback upsert（(request_id, reviewer) 唯一）+ status 过滤
6. 策略注册表：register / 指针切换 / 回滚 / 版本列表
7. 模型注册表：production 指针 / 版本列表
8. 重启持久化：新 PostgresStore 同 URL → 数据仍在

运行（需先起 Postgres 并执行 migrations/0001_init.sql）：
  AIVORA_TEST_DB_URL=postgresql://aivora:***@localhost:5432/aivora_decision \
      ../../.venv/bin/python test_postgres_store.py

未设置 AIVORA_TEST_DB_URL 或连不上 → 输出 SKIP（本地 JsonlStore 路径由
test_service.py 覆盖，不阻塞 CI）。
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

SERVICE = Path(__file__).resolve().parent
PIPE = SERVICE.parent / "training" / "pipelines"
POLICY_YAML = SERVICE.parent / "policies" / "recon" / "engine.yaml"
GOLDEN = SERVICE.parent / "contracts" / "examples" / "recon.golden.json"

sys.path.insert(0, str(PIPE))
sys.path.insert(0, str(SERVICE))
sys.path.insert(0, str(SERVICE.parent / "training" / "policies"))

DB_URL = os.environ.get("AIVORA_TEST_DB_URL", "")


def _try_open():
    if not DB_URL:
        print("SKIP: 未设置 AIVORA_TEST_DB_URL（Postgres 集成测试跳过，"
              "JsonlStore 路径见 test_service.py）")
        return None
    from postgres_store import PostgresStore
    st = PostgresStore(DB_URL)
    try:
        st.open(timeout_s=10)
    except Exception as e:  # noqa: BLE001
        print(f"SKIP: 无法连接 {DB_URL}（{type(e).__name__}）")
        return None
    return st


def _mk_log(rid: str, action: str = "REVIEW", shadow: bool = False,
            ts: float | None = None) -> "object":
    from storage import DecisionLog
    return DecisionLog(
        decision_id=f"DEC-{abs(hash(rid)) & 0xFFFFFFFF:08x}",
        payment_id=f"PAY-{rid}", invoice_id=f"INV-{rid}",
        action=action, policy_rule="r04-auto-match", policy_reason="HIGH_CONFIDENCE",
        policy_version="1.0", model_version="recon-v1:abc123",
        feature_version="1.0", label="MATCHED", conf=0.97,
        probs={"MATCHED": 0.97, "PARTIAL": 0.02, "UNMATCHED": 0.01},
        is_shadow=shadow, ts=ts if ts is not None else time.time(),
        request_id=rid, tenant_id="t-test",
        input_digest="d" * 64, latency_ms=8,
    )


def main() -> int:
    st = _try_open()
    if st is None:
        return 0
    import psycopg
    tests = []

    def test(name):
        def deco(fn):
            tests.append((name, fn))
            return fn
        return deco

    # ------------------------------------------------------------------
    @test("1. 五表 DDL + 任务注册")
    def t01():
        with st._conn() as conn:
            rows = conn.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema='public' ORDER BY table_name").fetchall()
        names = {r["table_name"] for r in rows}
        need = {"decision_tasks", "decision_models", "decision_policies",
                "decision_logs", "decision_feedback"}
        assert need <= names, f"缺表: {need - names}"
        with st._conn() as conn:
            d = conn.execute(
                "SELECT spec_jsonb FROM decision_tasks "
                "WHERE task_id='recon.transaction_match' AND version='1.0'").fetchone()
        assert d is not None and d["spec_jsonb"]["labels"] == \
            ["MATCHED", "PARTIAL", "UNMATCHED"]
        print("  [PASS] 五表齐全 + recon 任务已注册")

    # ------------------------------------------------------------------
    @test("2. 决策落库往返（字段保真）")
    def t02():
        st.append_decision(_mk_log("pg-rid-001", action="AUTO", shadow=True, ts=1730000000.0))
        got = st.decision_by_request_id("pg-rid-001")
        assert got is not None
        assert got.action == "AUTO" and got.is_shadow is True
        assert got.label == "MATCHED" and abs(got.conf - 0.97) < 1e-9
        assert got.probs == {"MATCHED": 0.97, "PARTIAL": 0.02, "UNMATCHED": 0.01}
        assert got.tenant_id == "t-test" and got.input_digest == "d" * 64
        assert got.latency_ms == 8 and got.policy_rule == "r04-auto-match"
        assert abs(got.ts - 1730000000.0) < 1.0
        print("  [PASS] 决策往返：AUTO/shadow/probs/digest/latency 全保真")

    # ------------------------------------------------------------------
    @test("3. request_id 幂等（重复 append 不覆盖首次）")
    def t03():
        st.append_decision(_mk_log("pg-rid-001", action="REVIEW", shadow=False,
                                   ts=1730000999.0))   # 同 rid 不同载荷
        got = st.decision_by_request_id("pg-rid-001")
        assert got.action == "AUTO" and got.is_shadow is True, "首次结果被覆盖！"
        assert got.latency_ms == 8
        print("  [PASS] 幂等：ON CONFLICT DO NOTHING，首次结果保持权威")

    # ------------------------------------------------------------------
    @test("4. decisions(since_ts, limit) 过滤 + 升序")
    def t04():
        st.append_decision(_mk_log("pg-rid-002", ts=1730001000.0))
        st.append_decision(_mk_log("pg-rid-003", ts=1730002000.0))
        all_rows = list(st.decisions(0, 100))
        ids = [d.request_id for d in all_rows]
        assert ids == sorted(ids, key=lambda r: next(
            d.ts for d in all_rows if d.request_id == r))
        late = list(st.decisions(1730001500.0, 100))
        assert [d.request_id for d in late] == ["pg-rid-003"]
        limited = list(st.decisions(0, 2))
        assert len(limited) == 2
        print(f"  [PASS] 过滤/升序/limit（全量 {len(all_rows)} 条）")

    # ------------------------------------------------------------------
    @test("5. feedback upsert + status 过滤")
    def t05():
        rec = {"request_id": "pg-rid-001", "reviewer": "fin-01",
               "reviewed_at": time.time(), "model_label": "MATCHED",
               "corrected_label": None, "review_action": "MAINTAIN",
               "correction_reason": None, "status": "pending",
               "tenant_id": "t-test"}
        st.append_feedback(rec)
        rec2 = dict(rec, status="approved", corrected_label="PARTIAL",
                    correction_reason="部分付款")
        st.append_feedback(rec2)   # 同 (rid, reviewer) → upsert
        rows = st.feedback_records()
        mine = [r for r in rows if r["reviewer"] == "fin-01"]
        assert len(mine) == 1 and mine[0]["status"] == "approved"
        assert mine[0]["corrected_label"] == "PARTIAL"
        assert len(st.feedback_records(status="approved")) == 1
        print("  [PASS] feedback upsert（pending→approved 单行）+ 过滤")

    # ------------------------------------------------------------------
    @test("6. 策略注册表（register/切换/回滚）")
    def t06():
        import json
        import yaml
        doc = yaml.safe_load(POLICY_YAML.read_text(encoding="utf-8"))
        from strategy import PolicyEngine
        eng1 = PolicyEngine.from_dict(doc)
        st.register_policy("1.0", eng1.snapshot())
        assert st.current_policy_version() == "1.0"
        # v1.1：改金额上限（快照不同）
        doc2 = yaml.safe_load(POLICY_YAML.read_text(encoding="utf-8")
                              .replace("100000", "80000"))
        eng2 = PolicyEngine.from_dict(doc2)
        st.register_policy("1.1", eng2.snapshot())
        assert st.current_policy_version() == "1.1"
        vers = st.policy_versions()
        act = [p["version"] for p in vers if p["active"]]
        assert act == ["1.1"] and len(vers) == 2
        st.rollback_policy("1.0")
        vers = st.policy_versions()
        act = [p["version"] for p in vers if p["active"]]
        assert act == ["1.0"], "回滚后 active 指针错误"
        # 回滚目标可从快照无损重建引擎（strategy.from_dict 往返）
        snap = next(p["snapshot"] for p in vers if p["version"] == "1.0")
        eng_rb = PolicyEngine.from_dict(snap)
        assert eng_rb.version == "1.0" and eng_rb.policy_id == "recon.transaction_match"
        assert len(eng_rb.rules) == len(eng1.rules)
        print("  [PASS] 策略指针切换 + 回滚 + 快照无损重建")

    # ------------------------------------------------------------------
    @test("7. 模型注册表（production 指针）")
    def t07():
        st.set_current_model_version("recon-v1:abc123")
        assert st.current_model_version() == "recon-v1:abc123"
        st.set_current_model_version("recon-v1:def456")
        assert st.current_model_version() == "recon-v1:def456"
        assert "recon-v1:abc123" in st.model_versions()
        with st._conn() as conn:
            d = conn.execute(
                "SELECT data_snapshot_hash FROM decision_models "
                "WHERE model_id='recon-v1' AND version='recon-v1:def456'").fetchone()
        assert d["data_snapshot_hash"] == "def456", "data_snapshot_hash 未锚定（定稿 §2.3 硬约束）"
        print("  [PASS] 模型 production 指针 + data_snapshot_hash 锚定")

    # ------------------------------------------------------------------
    @test("8. 重启持久化（新 store 同 URL）")
    def t08():
        st2 = _try_open()
        assert st2.decision_by_request_id("pg-rid-001").action == "AUTO"
        assert st2.current_policy_version() == "1.0"
        assert st2.current_model_version() == "recon-v1:def456"
        assert len(st2.feedback_records(status="approved")) == 1
        st2.close()
        print("  [PASS] 重启后决策/策略/模型/feedback 全部仍在")

    # ------------------------------------------------------------------
    failed = 0
    for name, fn in tests:
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            print(f"  [FAIL] {name}: {e}")
            failed += 1
            break   # 前置失败则后续无意义
    st.close()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
