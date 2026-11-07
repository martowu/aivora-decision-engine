"""决策服务核心（W3 交付物，定稿 §2.2 / §7-E / §2.2.7 API 契约）。

把 模型工件 + 策略引擎 + 存储 装成一个可运行、可单测的决策单元。
app.py 是它的 FastAPI 薄壳。

安全契约（定稿 §7-E 事故回滚）：
- 模型工件 / 策略 任一未加载或加载失败 → ServiceUnavailable（HTTP 503 全停），
  绝不静默降级到"无策略裸 AUTO"；
- 推理 / 求值抛异常 → ServiceUnavailable（503），该笔交上游人工，不产生决策；
- shadow 模式 → 模型照常推理落库（is_shadow=True），但对外动作掩码成 REVIEW，
  AUTO 永不穿透（W4 门禁：推翻率 ≤2% 才切生产）。

API 契约（定稿 §2.2.7 / v1.0 §2.2.7）：
- request_id 幂等：调用方提供 request_id 时，重复请求返回首次结果
  （响应 duplicate=True，app 层映射 409 DUPLICATE_REQUEST）。
  幂等键对逻辑请求唯一、重放语义永久有效（定稿"7 天窗口"= 最短保留期；
  决策日志审计保留期更长，见 §2.2.2 全量落库）；
- tenant_id 贯穿落库（一期单租户，行级隔离预留）；
- input_digest：任务输入摘要 sha256（定稿 §2.2.8）；latency_ms：推理+策略耗时
  （G-Shadow P95≤100ms 监控源）；
- feedback：人工复核回写 (request_id, reviewer) 唯一，status pending→approved/rejected；
  仅 approved 的修正可进入下一版训练集（定稿 §3-8，训练侧规则）。
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import threading
import time
from pathlib import Path
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "training" / "policies"))
from strategy import PolicyEngine, PolicyCompileError, build_context  # noqa: E402

from inference import (ArtifactLoadError, InferenceBundle, predict,  # noqa: E402
                       load_bundle)
from shadow import shadow_mask  # noqa: E402
from storage import (DecisionLog, JsonlStore, Store, make_decision_id,  # noqa: E402
                     make_request_id)

import yaml  # noqa: E402

LABELS = ("MATCHED", "PARTIAL", "UNMATCHED")
FEEDBACK_STATUSES = ("pending", "approved", "rejected")

class ServiceUnavailableError(RuntimeError):
    """对应 HTTP 503：模型/策略不可用或推理失败 → 全停。"""


MODES = ("production", "shadow")


def response_from_log(log: DecisionLog) -> dict:
    """从落库记录重建响应（409 幂等重放 / 审计查询共用，保证口径一致）。

    落库记的是**真实决策**（shadow 期也是 AUTO），对外动作按当时 mode 掩码还原。
    """
    external = shadow_mask(log.action) if log.is_shadow else log.action
    return {
        "decision_id": log.decision_id,
        "request_id": log.request_id,
        "tenant_id": log.tenant_id,
        "action": external,
        "model_action": log.action,
        "mode": "shadow" if log.is_shadow else "production",
        "model": {"label": log.label, "conf": log.conf, "probs": log.probs},
        "policy": {"rule": log.policy_rule, "reason": log.policy_reason,
                   "version": log.policy_version},
        "model_version": log.model_version,
        "feature_version": log.feature_version,
        "is_shadow": log.is_shadow,
    }


class DecisionService:
    def __init__(self, store: Store, mode: str = "production",
                 idempotency_window_s: float = 7 * 86400):
        if mode not in MODES:
            raise ValueError(f"mode 必须是 {MODES}")
        self._store = store
        self._mode = mode
        self._lock = threading.RLock()
        self._idempotency_window_s = idempotency_window_s
        self._bundle: Optional[InferenceBundle] = None
        self._policy: Optional[PolicyEngine] = None
        self._policy_registry: dict[str, PolicyEngine] = {}  # version -> engine

    # ---- 加载 / 热切换（原子替换，失败保持旧版 → 不中断在途） ----
    def load_model(self, artifact_dir: str | Path) -> str:
        bundle = load_bundle(artifact_dir)   # 可能抛 ArtifactLoadError
        with self._lock:
            self._bundle = bundle
            self._store.set_current_model_version(bundle.model_version)
        return bundle.model_version

    def load_policy(self, yaml_path: str | Path,
                    golden: Optional[list[dict]] = None,
                    version: Optional[str] = None) -> str:
        doc = yaml.safe_load(Path(yaml_path).read_text(encoding="utf-8"))
        eng = PolicyEngine.from_dict(doc)
        # 黄金样例校验（启动期 + 每次切换都跑，不符则拒绝加载）
        if golden:
            eng.validate(golden)
        v = version or eng.version
        with self._lock:
            self._policy_registry[v] = eng
            self._policy = eng
            self._store.register_policy(v, eng.snapshot())
        return v

    def rollback_policy(self, target_version: str) -> str:
        with self._lock:
            if target_version not in self._policy_registry:
                # 内存里没编译过的版本，从 store 快照重建
                snap = next((p["snapshot"] for p in self._store.policy_versions()
                             if p["version"] == target_version), None)
                if snap is None:
                    raise ValueError(f"策略版本 {target_version} 不存在")
                eng = PolicyEngine.from_dict(snap)
                self._policy_registry[target_version] = eng
            self._policy = self._policy_registry[target_version]
            self._store.rollback_policy(target_version)
        return target_version

    def set_mode(self, mode: str) -> str:
        if mode not in MODES:
            raise ValueError(f"mode 必须是 {MODES}")
        with self._lock:
            self._mode = mode
        return mode

    def status(self) -> dict:
        with self._lock:
            return {
                "mode": self._mode,
                "model_loaded": self._bundle is not None,
                "model_version": self._bundle.model_version if self._bundle else None,
                "policy_loaded": self._policy is not None,
                "policy_version": self._policy.version if self._policy else None,
                "feature_version": self._bundle.feature_version if self._bundle else None,
            }

    # ---- 核心决策 ----
    def _require_ready(self) -> tuple[InferenceBundle, PolicyEngine, str]:
        with self._lock:
            if self._bundle is None or self._policy is None:
                raise ServiceUnavailableError(
                    "模型或策略未就绪（503 全停）：model="
                    f"{self._bundle is not None} policy={self._policy is not None}")
            return self._bundle, self._policy, self._mode

    def decide(self, pair_kwargs: dict,
               request_id: Optional[str] = None,
               tenant_id: str = "default") -> dict:
        """统一决策入口。

        幂等：request_id 提供且已有首次决策 → 不重算、不落新日志，
        返回首次结果（duplicate=True，app 层映射 409）。重放语义永久有效
        （幂等键对逻辑请求唯一；定稿 7 天窗口 = 最短保留期）。
        """
        bundle, policy, mode = self._require_ready()
        from inference import PairInput   # 延迟导入避免循环

        # ---- 幂等短路（在推理之前，保证重复请求零副作用）----
        if request_id:
            prev = self._store.decision_by_request_id(request_id)
            if prev is not None:
                resp = response_from_log(prev)
                resp["duplicate"] = True
                return resp

        req_id = request_id or make_request_id()
        t0 = time.perf_counter()

        try:
            pair = PairInput(**pair_kwargs)
            out = predict(bundle, pair)
        except (ArtifactLoadError, ServiceUnavailableError):
            raise
        except Exception as e:  # 推理失败 → 全停，交人工（绝不静默降级）
            raise ServiceUnavailableError(f"推理失败 503: {type(e).__name__}: {e}") from e

        ctx = build_context(
            features=out["features"], calib_probs=out["probs"],
            label=out["label"],
            payment={"amount": pair.pay_amount, "currency": pair.pay_currency,
                     "channel": pair.channel, "payer_id": pair.payer_id})
        try:
            dec = policy.decide(ctx)
        except PolicyCompileError as e:
            raise ServiceUnavailableError(f"策略编译错误 503: {e}") from e
        latency_ms = int((time.perf_counter() - t0) * 1000)

        # 输入摘要（定稿 §2.2.8）：任务输入全部字段的规范化 sha256
        digest_src = json.dumps(dataclasses.asdict(pair), sort_keys=True,
                                ensure_ascii=False, default=str)
        input_digest = hashlib.sha256(digest_src.encode("utf-8")).hexdigest()

        model_action = dec.action                       # 模型/策略真实决策
        external_action = shadow_mask(model_action) if mode == "shadow" else model_action

        log = DecisionLog(
            decision_id=make_decision_id(),
            payment_id=pair.payment_id, invoice_id=pair.invoice_id,
            action=model_action,                    # 落库记真实决策（shadow 也记 AUTO）
            policy_rule=dec.rule, policy_reason=dec.reason,
            policy_version=policy.version,
            model_version=bundle.model_version,
            feature_version=bundle.feature_version,
            label=out["label"], conf=out["conf"],
            probs={l: p for l, p in zip(LABELS, out["probs"])},
            is_shadow=(mode == "shadow"), ts=time.time(),
            request_id=req_id, tenant_id=tenant_id,
            input_digest=input_digest, latency_ms=latency_ms,
        )
        self._store.append_decision(log)

        resp = response_from_log(log)
        resp["duplicate"] = False
        resp["latency_ms"] = latency_ms
        return resp

    # ---- 审计查询（GET /v1/decisions/{request_id}）----
    def audit(self, request_id: str) -> Optional[dict]:
        log = self._store.decision_by_request_id(request_id)
        if log is None:
            return None
        return response_from_log(log)

    # ---- 人工复核回写（POST /v1/feedback/{request_id}，定稿 §2.2.7 / §3-8）----
    def submit_feedback(self, request_id: str, reviewer: str,
                        corrected_label: Optional[str] = None,
                        review_action: Optional[str] = None,
                        reason: Optional[str] = None,
                        status: str = "pending") -> dict:
        if not reviewer:
            raise ValueError("reviewer 必填")
        if status not in FEEDBACK_STATUSES:
            raise ValueError(f"status 必须是 {FEEDBACK_STATUSES}")
        if corrected_label is not None and corrected_label not in LABELS:
            raise ValueError(f"corrected_label 必须是 {LABELS} 之一")
        if status in ("approved", "rejected") and corrected_label is None \
                and not review_action:
            # 终态必须携带结论（维持=review_action，纠正=corrected_label）
            raise ValueError("approved/rejected 反馈须携带 corrected_label 或 review_action")

        log = self._store.decision_by_request_id(request_id)
        if log is None:
            raise KeyError(request_id)   # app 层映射 404

        rec = {
            "request_id": request_id,
            "decision_id": log.decision_id,
            "tenant_id": log.tenant_id,
            "reviewer": reviewer,
            "reviewed_at": time.time(),
            "model_label": log.label,          # 复核对照基线
            "corrected_label": corrected_label,
            "review_action": review_action,
            "correction_reason": reason,
            "status": status,
        }
        self._store.append_feedback(rec)
        rec = dict(rec)
        rec["note"] = ("仅 status=approved 的修正进入下一版训练集（定稿 §3-8）"
                       if status == "approved" else "pending：待审核后才可作训练标签")
        return rec

    def feedback(self, status: Optional[str] = None,
                 limit: int = 1000) -> list[dict]:
        return self._store.feedback_records(status=status, limit=limit)
