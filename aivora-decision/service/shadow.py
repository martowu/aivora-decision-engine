"""Shadow 模式（W4 门禁核心件，定稿 §2.2.7 / §5-W4）。

Shadow 语义（最安全形态）：
- 模型照常推理并落 decision_logs（is_shadow=True）；
- 但**外部响应一律掩码成 REVIEW**，绝不让 AUTO 穿透触发自动核销；
- 推翻率（override rate）由 ShadowReporter 离线对照"人工真值"计算：
    在模型路由 AUTO 的样本里，人工实际决策与模型不一致的比例。
  W4 门禁：override rate ≤ 2% 才能切生产。

为什么离线对照：shadow 期请求里只有 (付款, 候选发票)，人工决策是事后才有的。
所以 shadow 跑一段时间 → 人工决策回填 → 用本模块对照算推翻率。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

# shadow 掩码：模型动作 → 对外动作
SHADOW_MASK = {"AUTO": "REVIEW", "REVIEW": "REVIEW"}


def shadow_mask(action: str) -> str:
    """shadow 期对外动作（AUTO 一律降级 REVIEW）。"""
    return SHADOW_MASK.get(action, "REVIEW")


@dataclass
class ShadowResult:
    n_decided: int                # 模型有决策的样本数
    n_auto: int                   # 模型路由 AUTO 的样本数
    n_auto_overridden: int        # AUTO 但人工不同意（推翻）
    override_rate: Optional[float]  # n_auto_overridden / n_auto（n_auto=0 → None）
    agree_rate: Optional[float]      # (n_auto - n_auto_overridden) / n_auto
    details: list[dict]

    def gate_passed(self, max_override: float = 0.02) -> bool:
        return self.override_rate is not None and self.override_rate <= max_override

    def to_dict(self) -> dict:
        return {"n_decided": self.n_decided, "n_auto": self.n_auto,
                "n_auto_overridden": self.n_auto_overridden,
                "override_rate": None if self.override_rate is None
                else round(self.override_rate, 6),
                "agree_rate": None if self.agree_rate is None
                else round(self.agree_rate, 6),
                "gate_passed_max_0.02": self.gate_passed(0.02)}


class ShadowReporter:
    """对照模型决策与人工真值，算推翻率。

    model_logs  : Iterable[DecisionLog]（is_shadow 决策）
    human_truth : {payment_id: {"label": str, "action": "AUTO"/"REVIEW"}}
                  人工回填的真值。label=人工认定的三态；action=人工是否放行自动核销。

    "推翻"判定：模型 action==AUTO，但 人工 action!=AUTO 或 人工 label != 模型 label。
    （模型想自动核销成 MATCHED，但人工认为不该自动 / 人工认定是别的态 → 推翻）
    """

    def report(self, model_logs: Iterable, human_truth: dict,
               max_auto: int = 10_000) -> ShadowResult:
        details, n_decided, n_auto, n_over = [], 0, 0, 0
        for log in model_logs:
            if not getattr(log, "is_shadow", False):
                continue
            n_decided += 1
            if log.action != "AUTO":
                continue
            n_auto += 1
            truth = human_truth.get(log.payment_id)
            if truth is None:
                continue  # 人工还没回填，不计入推翻率（避免分母污染）
            overridden = (truth.get("action", "REVIEW") != "AUTO"
                          or truth.get("label") != log.label)
            if overridden:
                n_over += 1
            if len(details) < max_auto:
                details.append({"payment_id": log.payment_id, "model_label": log.label,
                                "human_label": truth.get("label"),
                                "human_action": truth.get("action"),
                                "overridden": overridden})
        orate = (n_over / n_auto) if n_auto else None
        a_rate = ((n_auto - n_over) / n_auto) if n_auto else None
        return ShadowResult(n_decided, n_auto, n_over, orate, a_rate, details)
