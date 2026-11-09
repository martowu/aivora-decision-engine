"""合成数据集生成器（贴近生产结构，信号可学习但有噪声，可复现）。

与 extract.py 里那个 SyntheticSource 的区别：
- 那个是"管线验证用"（付款/发票互相独立随机，只验证脱敏 + hash 稳定）；
- 这个是"训练/演示用"——先建**主体（counterparty）人群**，每个主体有内在
  "干净度"，再按干净度生成 (付款, 候选发票, 真值标签) 三元组。真值由构造决定，
  所以有可信的 ground truth。

关键设计（让信号**真实但不平凡**）：
- MATCHED / PARTIAL / UNMATCHED 之间刻意制造**特征重叠**：
  * UNMATCHED 里有一部分"主体一致但金额不对 + 附言无发票号"（最难，靠金额/发票区分）；
  * PARTIAL 里有一部分带手续费（fee 正好解释差额，靠 partial_ratio + has_fee 区分）；
  * MATCHED 里有一部分金额有小额舍入差（贴近容差边界）。
- 10% 复杂结构（多对多/拆合/跨币种）横切各标签 → 策略层应一律路由 REVIEW。
- 主体按"进入月份 + 活跃窗口"分布 → 早期主体落 train、后期落 test，
  group split 天然防同主体跨集（模拟真实"新主体随时间出现"）。

**红线（定稿 §1-D2）**：合成数据仅用于验证流程 + 演示决策链路，
**不得进入三硬门禁计数**。真实验收必须用脱敏 Recon 数据。

可复现：固定 seed → 同一 dataset_hash → 同字节 JSONL。
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict
from pathlib import Path

FEATURE_VERSION = "1.0"
N_COUNTERPARTIES = 80          # 主体人群规模
MONTHS = 10                    # 2026-01 .. 2026-10
YEAR = 2026


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _round2(x: float) -> float:
    return round(x + 1e-9, 2)


class Counterparty:
    __slots__ = ("id", "clean", "entry_month", "active")

    def __init__(self, idx: int, clean: float, entry_month: int, active: tuple[int, int]):
        self.id = f"CP-{idx:03d}"
        self.clean = clean              # 内在干净度 [0.4, 0.99]
        self.entry_month = entry_month  # 首次出现的月份
        self.active = active            # (start_month, end_month) 活跃窗口


def _make_counterparties(rng: random.Random) -> list[Counterparty]:
    cps = []
    for idx in range(1, N_COUNTERPARTIES + 1):
        clean = _clip(rng.gauss(0.72, 0.15), 0.40, 0.99)
        # 进入月份：越靠后的主体越多落在 test 时间段（模拟主体随时间新增）
        entry = rng.randint(1, MONTHS)
        end = min(MONTHS, entry + rng.randint(2, 7))
        cps.append(Counterparty(idx, clean, entry, (entry, end)))
    return cps


def _sample_base_amount(rng: random.Random) -> float:
    if rng.random() < 0.80:
        return _round2(rng.uniform(2_000, 100_000))
    return _round2(rng.uniform(100_000, 800_000))  # 20% 大额 → 触发金额上限规则


def _gen_record(i: int, rng: random.Random, cps: list[Counterparty]) -> tuple[dict, str]:
    cp = rng.choice(cps)
    m0, m1 = cp.active
    month = rng.randint(m0, m1)
    day = rng.randint(1, 28)
    pay_date = f"{YEAR}-{month:02d}-{day:02d}"

    # 标签：按全局权重，干净度调高 MATCHED 概率（真实：老客户更常精准匹配）
    p_matched = _clip(cp.clean, 0.35, 0.9)
    r = rng.random()
    if r < p_matched * 0.72:
        label = "MATCHED"
    elif r < p_matched * 0.72 + 0.28:
        label = "PARTIAL"
    else:
        label = "UNMATCHED"

    base = _sample_base_amount(rng)
    inv_amount = base
    payer_id = cp.id
    channel = rng.choice(("BANK_TRANSFER", "CHECK", "ONLINE"))
    invoice_id = f"INV-{cp.id}-{month:02d}-{i:05d}"

    # 复杂结构（横切各标签）→ 策略层路由 REVIEW
    complex_roll = rng.random()
    is_multi = complex_roll < 0.04
    has_split = (not is_multi) and complex_roll < 0.08
    cur_mm = (not is_multi and not has_split) and complex_roll < 0.12

    if label == "MATCHED":
        if rng.random() < 0.90:
            pay_amount = base                      # 精准
        else:
            pay_amount = _round2(base + rng.uniform(-40, 40))  # 小额舍入差
        supplier_id = payer_id if rng.random() < 0.95 else f"SP-{rng.randint(1,999):03d}"
        ref_has_inv = rng.random() < 0.95
        reference = (f"付款 {invoice_id} 全额" if ref_has_inv
                     else "往来款 无票号")
        inv_day = max(1, day - rng.randint(0, 8))
        rate = _clip(cp.clean + rng.gauss(0, 0.04), 0, 1)
        fee = 0.0
    elif label == "PARTIAL":
        pay_amount = _round2(base * rng.uniform(0.5, 0.95))
        supplier_id = payer_id if rng.random() < 0.90 else f"SP-{rng.randint(1,999):03d}"
        ref_has_inv = rng.random() < 0.85
        reference = (f"付款 {invoice_id} 部分" if ref_has_inv else "部分款项")
        inv_day = max(1, day - rng.randint(0, 12))
        rate = _clip(cp.clean - 0.08 + rng.gauss(0, 0.05), 0, 1)
        fee = _round2(base - pay_amount) if rng.random() < 0.4 else 0.0  # 40% 带手续费
    else:  # UNMATCHED
        pay_amount = _round2(base * rng.uniform(0.3, 1.6))
        # 难点：40% 主体一致但金额不对 + 附言无票号（靠金额/发票号区分）
        if rng.random() < 0.40:
            supplier_id = payer_id
            reference = rng.choice(["其他付款", "往来款", "预付款"])  # 无票号
        else:
            supplier_id = f"SP-{rng.randint(1,999):03d}"
            reference = rng.choice([
                "其他付款", "往来款",
                f"付款 INV-{cp.id}-{rng.randint(1,MONTHS):02d}-XXXXX"  # 含"别的"票号
            ])
        inv_day = max(1, day - rng.randint(0, 35))
        rate = _clip(cp.clean - 0.25 + rng.gauss(0, 0.06), 0, 1)
        fee = 0.0

    pay_currency = "CNY"
    inv_currency = "USD" if cur_mm else "CNY"
    inv_date = f"{YEAR}-{month:02d}-{inv_day:02d}"
    first = (month == m0) and rng.random() < 0.5

    rec = {
        "payment_id": f"PAY-{i:06d}",
        "invoice_id": invoice_id,
        "pay_amount": pay_amount,
        "inv_amount": inv_amount,
        "pay_currency": pay_currency,
        "inv_currency": inv_currency,
        "pay_date": pay_date,
        "inv_date": inv_date,
        "payer_id": payer_id,
        "supplier_id": supplier_id,
        "supplier_name": f"供应商{cp.id}有限公司",
        "channel": channel,
        "reference": reference,
        "note": "发票备注" * rng.randint(1, 3),
        "fee": fee,
        "counterparty_match_rate_hist": round(rate, 4),
        "is_first_counterparty": first,
        "is_multi_mapping": is_multi,
        "has_split_merge": has_split,
        "true_label": label,
        "counterparty_clean": round(cp.clean, 4),   # 仅用于事后分析，非模型输入
        "counterparty_entry_month": m0,
    }
    return rec, label


def generate_dataset(n: int = 1500, seed: int = 42) -> list[dict]:
    """生成 n 条 (付款, 候选发票, 真值) 记录。seed 固定 → 完全可复现。"""
    rng = random.Random(seed)
    cps = _make_counterparties(rng)
    out = []
    for i in range(n):
        rec, _ = _gen_record(i, rng, cps)
        out.append(rec)
    return out


def dataset_hash(records: list[dict]) -> str:
    """对记录做稳定序列化后 sha256（排除事后分析字段，只锚定决策相关字段）。"""
    h = hashlib.sha256()
    h.update(f"synthetic:{FEATURE_VERSION}\n".encode())
    for r in sorted(records, key=lambda x: x["payment_id"]):
        canon = {k: v for k, v in sorted(r.items())
                 if k not in ("counterparty_clean", "counterparty_entry_month")}
        h.update((json.dumps(canon, ensure_ascii=False, default=str) + "\n").encode())
    return h.hexdigest()


def write_dataset(records: list[dict], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return path


def load_dataset(path: str | Path) -> list[dict]:
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


# ---------------------------------------------------------------------------
# 转 PairInput（复用 features.py 的 Schema）
# ---------------------------------------------------------------------------

def to_pair_inputs(records: list[dict]):
    """records → (list[PairInput], list[label])，与 features.PairInput 对齐。"""
    from features import PairInput
    inputs, labels = [], []
    for r in records:
        inputs.append(PairInput(
            payment_id=r["payment_id"], invoice_id=r["invoice_id"],
            pay_amount=r["pay_amount"], inv_amount=r["inv_amount"],
            pay_currency=r["pay_currency"], inv_currency=r["inv_currency"],
            pay_date=r["pay_date"], inv_date=r["inv_date"],
            payer_id=r["payer_id"], supplier_id=r["supplier_id"],
            supplier_name=r["supplier_name"], channel=r["channel"],
            reference=r["reference"], note=r["note"], fee=r["fee"],
            counterparty_match_rate_hist=r["counterparty_match_rate_hist"],
            is_first_counterparty=r["is_first_counterparty"],
            is_multi_mapping=r["is_multi_mapping"],
            has_split_merge=r["has_split_merge"],
            invoice_number=r.get("invoice_number", r["invoice_id"]),
        ))
        labels.append(r["true_label"])
    return inputs, labels


if __name__ == "__main__":
    recs = generate_dataset(1500, 42)
    from collections import Counter
    c = Counter(r["true_label"] for r in recs)
    complex_n = sum(1 for r in recs if r["is_multi_mapping"] or r["has_split_merge"]
                    or r["pay_currency"] != r["inv_currency"])
    print(json.dumps({
        "n": len(recs),
        "label_dist": dict(c),
        "complex_structure": complex_n,
        "dataset_hash": dataset_hash(recs)[:16],
        "sample": {k: recs[0][k] for k in
                   ("payment_id", "pay_amount", "inv_amount", "reference",
                    "payer_id", "supplier_id", "true_label")},
    }, ensure_ascii=False, indent=2))
