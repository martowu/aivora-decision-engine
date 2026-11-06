"""特征工程 v1.0 —— 版本化纯函数（定稿 §3-6、§2.2.3）。

约束（硬性）：
- 纯函数：输入 = 任务 Schema 对象（payment + candidate_invoice + 预计算历史统计），
  输出 = 固定顺序的 numpy 向量 + 命名特征 dict。同输入必同输出（可单测、可复现）。
- point-in-time 安全：所有统计类输入（counterparty_match_rate_hist 等）必须是
  付款日之前的快照，由上游数据管线保证；本模块只消费、不查询，因此天然 PIT 安全。
- 结构 flag 透传：复杂结构 flag 不进模型训练样本（训练集排除），但保留在特征向量中
  供策略引擎消费（定稿 §7-D：测试集保留以验证拒绝能力）。
- 禁止事后信息：本模块不接受"历史核销结果""是否曾被人工处理"等字段（见 FORBIDDEN_INPUTS）。

改特征逻辑必须 FEATURE_VERSION +0.1，并同步更新单测与文档。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Mapping

import numpy as np

FEATURE_VERSION = "1.0"

LABELS = ("MATCHED", "PARTIAL", "UNMATCHED")

# 事后信息黑名单（定稿 §1.3.6）：出现即抛错，防泄漏
FORBIDDEN_INPUTS = (
    "historical_recon_result",
    "was_manually_reviewed",
    "final_business_outcome",
)

# 复杂结构 flag（策略直路由 REVIEW；训练集排除，测试集保留 —— 定稿 §7-D）
COMPLEX_FLAGS = ("is_multi_mapping", "has_split_merge", "currency_mismatch")

CHANNELS = ("BANK_TRANSFER", "CHECK", "ONLINE")
AMOUNT_BUCKETS = (("low", 0, 50_000), ("mid", 50_000, 500_000), ("high", 500_000, float("inf")))

FEATURE_NAMES: list[str] = [
    # 金额组
    "amount_diff", "amount_diff_ratio", "amount_abs_diff", "partial_ratio",
    "amount_log10",
    # 时间组
    "date_diff_days", "is_cross_period", "is_cross_year",
    # 主体组
    "supplier_exact_match", "supplier_fuzzy_score",
    "counterparty_match_rate_hist", "is_first_counterparty",
    # 单据组
    "invoice_number_in_reference", "invoice_number_fuzzy_hit",
    "reference_len_norm", "note_len_norm",
    # 结构 flag 组（策略消费）
    "is_multi_mapping", "has_split_merge", "currency_mismatch", "has_fee",
    # 渠道组
    "channel_BANK_TRANSFER", "channel_CHECK", "channel_ONLINE",
    "amount_bucket_low", "amount_bucket_mid", "amount_bucket_high",
]
assert len(FEATURE_NAMES) == 26, f"特征数应为 26，实际 {len(FEATURE_NAMES)}"


@dataclass(frozen=True)
class PairInput:
    """一个 (付款, 候选发票) 对的全部输入。字段即任务 Schema + 预计算历史统计。"""

    payment_id: str
    invoice_id: str
    pay_amount: float
    inv_amount: float
    pay_currency: str
    inv_currency: str
    pay_date: str            # YYYY-MM-DD
    inv_date: str
    payer_id: str
    supplier_id: str
    supplier_name: str
    channel: str
    reference: str           # 付款附言（脱敏后）
    note: str                # 发票备注（脱敏后）
    fee: float               # 可验证手续费（无则 0）
    # —— 预计算历史统计（上游保证 point-in-time，本模块只消费）——
    counterparty_match_rate_hist: float   # 付款日之前该主体历史匹配率 [0,1]
    is_first_counterparty: bool
    # —— 结构 flag（上游判定）——
    is_multi_mapping: bool
    has_split_merge: bool


def _norm_name(s: str) -> str:
    return "".join(ch for ch in (s or "") if ch.isalnum()).lower()


def _fuzzy(a: str, b: str) -> float:
    na, nb = _norm_name(a), _norm_name(b)
    if not na or not nb:
        return 0.0
    return SequenceMatcher(None, na, nb).ratio()


def _days_between(pay: str, inv: str) -> int:
    from datetime import date
    y1, m1, d1 = map(int, pay.split("-"))
    y2, m2, d2 = map(int, inv.split("-"))
    return (date(y1, m1, d1) - date(y2, m2, d2)).days


def build_features(inp: PairInput) -> dict[str, float]:
    """纯函数特征构建。返回 dict（顺序 = FEATURE_NAMES）。"""
    for f in FORBIDDEN_INPUTS:
        if f in vars(inp) or f in getattr(inp, "__dict__", {}):
            raise ValueError(f"forbidden post-hoc field: {f}")

    amount_diff = inp.pay_amount - inp.inv_amount
    amount_diff_ratio = amount_diff / inp.inv_amount if inp.inv_amount else 0.0
    partial_ratio = min(inp.pay_amount / inp.inv_amount, 1.5) if inp.inv_amount else 0.0
    date_diff = _days_between(inp.pay_date, inp.inv_date)
    cross_period = 1.0 if (inp.pay_date[:7] != inp.inv_date[:7]) else 0.0
    cross_year = 1.0 if (inp.pay_date[:4] != inp.inv_date[:4]) else 0.0
    exact = 1.0 if inp.payer_id == inp.supplier_id or _norm_name(
        inp.payer_id) == _norm_name(inp.supplier_id) else 0.0
    fuzzy = _fuzzy(str(inp.payer_id), str(inp.supplier_id))
    ref = inp.reference or ""
    inv_no_exact = 1.0 if inp.invoice_id and inp.invoice_id in ref else 0.0
    inv_no_fuzzy = 1.0 if _fuzzy(inp.invoice_id, ref) >= 0.8 else 0.0
    currency_mm = 1.0 if inp.pay_currency != inp.inv_currency else 0.0
    fee_present = 1.0 if inp.fee and inp.fee > 0 else 0.0

    feats: dict[str, float] = {
        "amount_diff": round(amount_diff, 4),
        "amount_diff_ratio": round(amount_diff_ratio, 6),
        "amount_abs_diff": round(abs(amount_diff), 4),
        "partial_ratio": round(partial_ratio, 6),
        "amount_log10": round(math.log10(max(inp.inv_amount, 1.0)), 6),
        "date_diff_days": float(date_diff),
        "is_cross_period": cross_period,
        "is_cross_year": cross_year,
        "supplier_exact_match": exact,
        "supplier_fuzzy_score": round(fuzzy, 6),
        "counterparty_match_rate_hist": round(float(inp.counterparty_match_rate_hist), 6),
        "is_first_counterparty": 1.0 if inp.is_first_counterparty else 0.0,
        "invoice_number_in_reference": inv_no_exact,
        "invoice_number_fuzzy_hit": inv_no_fuzzy,
        "reference_len_norm": round(math.log10(len(ref) + 1), 6),
        "note_len_norm": round(math.log10(len(inp.note or "") + 1), 6),
        "is_multi_mapping": 1.0 if inp.is_multi_mapping else 0.0,
        "has_split_merge": 1.0 if inp.has_split_merge else 0.0,
        "currency_mismatch": currency_mm,
        "has_fee": fee_present,
    }
    for ch in CHANNELS:
        feats[f"channel_{ch}"] = 1.0 if inp.channel == ch else 0.0
    amt = inp.inv_amount
    for name, lo, hi in AMOUNT_BUCKETS:
        feats[f"amount_bucket_{name}"] = 1.0 if lo <= amt < hi else 0.0

    # 完整性自检：一个不多一个不少
    missing = [k for k in FEATURE_NAMES if k not in feats]
    extra = [k for k in feats if k not in FEATURE_NAMES]
    if missing or extra:
        raise ValueError(f"feature set mismatch: missing={missing} extra={extra}")
    return {k: feats[k] for k in FEATURE_NAMES}  # 固定顺序


def to_vector(feats: dict[str, float]) -> np.ndarray:
    return np.array([feats[k] for k in FEATURE_NAMES], dtype=np.float64)


def is_complex(f: Mapping[str, Any]) -> bool:
    """策略引擎用：命中任一复杂 flag → 直路由 REVIEW（不进 AUTO 通道）。"""
    return any(f.get(k, 0) >= 1 for k in COMPLEX_FLAGS)
