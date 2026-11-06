"""决策服务核心（W3 交付物，定稿 §2.2 / §7-E）。

把 模型工件 + 策略引擎 + 存储 装成一个可运行、可单测的决策单元。
app.py 是它的 FastAPI 薄壳。

安全契约（定稿 §7-E 事故回滚）：
- 模型工件 / 策略 任一未加载或加载失败 → ServiceUnavailable（HTTP 503 全停），
  绝不静默降级到"无策略裸 AUTO"；
- 推理 / 求值抛异常 → ServiceUnavailable（503），该笔交上游人工，不产生决策；
- shadow 模式 → 模型照常推理落库（is_shadow=True），但对外动作掩码成 REVIEW，
  AUTO 永不穿透（W4 门禁：推翻率 ≤2% 才切生产）。
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "training" / "policies"))
from strategy import PolicyEngine, PolicyCompileError, build_context  # noqa: E402

from inference import (ArtifactLoadError, InferenceBundle, predict,  # noqa: E402
                       load_bundle)
from storage import DecisionLog, JsonlStore, Store, make_decision_id  # noqa: E402
from shadow import shadow_mask  # noqa: E402

import yaml  # noqa: E402


class ServiceUnavailableError(RuntimeError):
    """对应 HTTP 503：模型/策略不可用或推理失败 → 全停。"""


MODES = ("production", "shadow")


class DecisionService:
    def __init__(self, store: Store, mode: str = "production"):
        if mode not in MODES:
            raise ValueError(f"mode 必须是 {MODES}")
        self._store = store
        self._mode = mode
        self._lock = threading.RLock()
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

    def decide(self, pair_kwargs: dict) -> dict:
        bundle, policy, mode = self._require_ready()
        from inference import PairInput   # 延迟导入避免循环
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
            probs={l: p for l, p in zip(("MATCHED", "PARTIAL", "UNMATCHED"), out["probs"])},
            is_shadow=(mode == "shadow"), ts=__import__("time").time(),
        )
        self._store.append_decision(log)

        return {
            "decision_id": log.decision_id,
            "action": external_action,             # 对外（shadow 掩码成 REVIEW）
            "model_action": model_action,          # 真实决策（审计用）
            "mode": mode,
            "model": {"label": out["label"], "conf": out["conf"], "probs": out["probs"]},
            "policy": {"rule": dec.rule, "reason": dec.reason,
                       "version": policy.version},
            "feature_version": bundle.feature_version,
            "is_shadow": (mode == "shadow"),
        }
