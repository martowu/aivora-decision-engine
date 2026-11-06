#!/usr/bin/env python3
"""W1 仿真原始表生成器（演练专用，定稿 §1-D2 红线）。

生成一套贴近 Recon 库结构的**原始**（未脱敏）表，供 W1 三份模板的
**流程演练**使用：
- extract.py --source csv        → 演练 §7-C 管线（清洗/脱敏/质检/快照 hash）
- sim_inventory_duckdb.py        → 演练数据盘点附录 A 的 SQL（DuckDB 执行）
- sim_w1_reports.py              → 演练基线测量（200 笔分层）+ 工作坊试点（100 例 Kappa）

产出（<out>/ 下，全部 git-ignore）：
  payments.csv        payment_id, tenant_id, doc_type, amount, currency, date,
                      payer_id, channel, reference, created_at
  invoices.csv        invoice_id, tenant_id, amount, currency, date,
                      supplier_id, supplier_name, note
  recon_results.csv   payment_id, invoice_id, final_label, confirmed_by,
                      reconciled_at, is_correct, role（演练真值 + 行角色，真实库无此列）
  counterparty_stats.csv  payer_id, month, match_rate（PIT 历史匹配率快照）
  sim_meta.json       seed / 规模 / 缺陷注入明细（可复核）

刻意注入的"真实感"：
- 标签分布按主体"干净度"生成（MATCHED/PARTIAL/UNMATCHED 特征重叠，同 synthetic.py 思路）；
- 候选集 1–5 张发票（3% 的付款 >50 张，演练截断策略）；
- 结构 flag：多对多 4% / 拆分 3% / 跨币种 2% / 手续费（PARTIAL 40%）/ 跨期 ~15%；
- 质量缺陷：主键重复 17、关键字段空值 31、未来日期 6、跨年跨度>1y 发票 5
  （演练盘点 §3 + CsvReconSource 清洗规则，extract 质检硬门槛须清洗后才过）；
- 标签来源四分类：human 55% / historical 25% / rule 15% / none 5%
  （gold 池 = human + historical，对照 G-数据 门槛计数）。

红线：**本脚本产出的任何数字都是演练值**，不得进入 G-数据 门禁、
不得用于 D5 go/no-go、不得用于模型验收（合成/仿真数据只验证 pipeline）。

用法：
  python simulate_w1.py --out ../out/sim-w1 [--seed 42]
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path

TENANT = "t-sim"
DOC_TYPE = "AP"
YEAR = 2026
MONTHS = (7, 8, 9)                 # 2026-07-01 .. 2026-09-30（3 个完整自然月）
N_COUNTERPARTIES = 40
PAYMENTS_PER_CP = (80, 120)        # 40 × ~100 ≈ 4,000 笔
LABEL_SOURCE_WEIGHTS = (("HUMAN", 0.55), ("HISTORICAL_CONFIRMED", 0.25),
                        ("RULE_AUTO", 0.15), ("NONE", 0.05))
CHANNELS = ("BANK_TRANSFER", "BANK_TRANSFER", "BANK_TRANSFER", "CHECK", "ONLINE")
SUPPLIER_SUFFIX = "供应链有限公司"


def _round2(x: float) -> float:
    return round(x + 1e-9, 2)


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


@dataclass
class Counterparty:
    idx: int
    cid: str
    clean: float
    entry_month: int
    active_end: int


@dataclass
class SimState:
    seed: int
    payments: list[dict] = field(default_factory=list)
    invoices: list[dict] = field(default_factory=list)
    recon: list[dict] = field(default_factory=list)
    defects: dict = field(default_factory=dict)
    _inv_seq: int = 0

    def next_invoice(self, cp: Counterparty, month: int) -> str:
        self._inv_seq += 1
        return f"INV-{cp.idx:03d}-{month:02d}-{self._inv_seq:05d}"


def _base_amount(rng: random.Random) -> float:
    if rng.random() < 0.80:
        return _round2(rng.uniform(2_000, 100_000))
    return _round2(rng.uniform(100_000, 800_000))


def _label_for(rng: random.Random, clean: float) -> str:
    p_matched = _clip(clean, 0.35, 0.90)
    r = rng.random()
    if r < p_matched * 0.72:
        return "MATCHED"
    if r < p_matched * 0.72 + 0.28:
        return "PARTIAL"
    return "UNMATCHED"


def _source_for(rng: random.Random) -> str:
    r, acc = rng.random(), 0.0
    for name, w in LABEL_SOURCE_WEIGHTS:
        acc += w
        if r < acc:
            return name
    return "NONE"


def _make_counterparties(rng: random.Random) -> list[Counterparty]:
    out = []
    for i in range(1, N_COUNTERPARTIES + 1):
        entry = rng.choices(MONTHS, weights=(0.5, 0.35, 0.15))[0]
        out.append(Counterparty(
            idx=i, cid=f"CP-{i:03d}",
            clean=_clip(rng.gauss(0.72, 0.15), 0.40, 0.99),
            entry_month=entry,
            active_end=MONTHS[-1],
        ))
    return out


def gen_core(rng: random.Random, cps: list[Counterparty], st: SimState) -> None:
    """核心：每笔付款 → 真值标签 → 发票（正确或干扰）→ recon 结果 → 结构 flag。"""
    pay_seq = 0
    for cp in cps:
        for _ in range(rng.randint(*PAYMENTS_PER_CP)):
            pay_seq += 1
            month = rng.choices(MONTHS, weights=(0.4, 0.35, 0.25))[0]
            if month < cp.entry_month:
                month = cp.entry_month
            day = rng.randint(1, 28)
            pay_date = date(YEAR, month, day)
            label = _label_for(rng, cp.clean)
            base = _base_amount(rng)
            amount = _round2(base)
            channel = rng.choice(CHANNELS)
            first_cp = (month == cp.entry_month)

            # —— 结构 flag（横切）——
            roll = rng.random()
            is_multi = roll < 0.04
            is_split = (not is_multi) and roll < 0.07
            is_crosscur = (not is_multi and not is_split) and roll < 0.09
            inv_currency = "USD" if is_crosscur else "CNY"

            # —— 日期：发票一般先于付款；~15% 跨月 ——
            inv_day = day - rng.randint(0, 8)
            inv_month = month
            if rng.random() < 0.15 and month > 7:
                inv_month = month - 1
                if inv_day > 28:
                    inv_day = 28
            inv_date = date(YEAR, inv_month, max(1, inv_day))

            # —— 金额与附言（按标签）——
            fee = 0.0
            if label == "MATCHED":
                if rng.random() < 0.08 and base < 50_000:      # 小额尾差
                    amount = _round2(base - rng.uniform(0.1, 40))
                ref = f"付款 {st.next_invoice(cp, inv_month)} 全额" if rng.random() < 0.95 else "往来款"
            elif label == "PARTIAL":
                amount = _round2(base * rng.uniform(0.5, 0.95))  # 部分付款
                if rng.random() < 0.4:                          # 手续费可验证
                    fee = _round2(base - amount)
                    amount = _round2(base - fee)
                ref = f"付款 INV-{cp.idx:03d} 部分" if rng.random() < 0.85 else "部分款项"
            else:  # UNMATCHED
                amount = _round2(base * rng.uniform(0.3, 1.6))
                if rng.random() < 0.4:
                    ref = rng.choice(["其他付款", "往来款", "预付款"])
                else:
                    ref = f"付款 INV-{rng.randint(1, N_COUNTERPARTIES):03d}-XX-XXXXX"

            pid = f"PAY-{pay_seq:06d}"
            st.payments.append({
                "payment_id": pid, "tenant_id": TENANT, "doc_type": DOC_TYPE,
                "amount": amount, "currency": "CNY", "date": pay_date.isoformat(),
                "payer_id": cp.cid, "channel": channel, "reference": ref,
                "created_at": (datetime.combine(pay_date, datetime.min.time())
                               + timedelta(hours=rng.randint(0, 23))).isoformat(sep=" "),
            })

            # —— 发票 ——
            inv_amount = _round2(base if label != "UNMATCHED" else base * rng.uniform(0.9, 1.1))
            inv_id = st.next_invoice(cp, inv_month)
            supplier = cp.cid if label != "UNMATCHED" or rng.random() < 0.4 \
                else f"CP-{rng.randint(1, N_COUNTERPARTIES):03d}"
            st.invoices.append({
                "invoice_id": inv_id, "tenant_id": TENANT, "amount": inv_amount,
                "currency": inv_currency, "date": inv_date.isoformat(),
                "supplier_id": supplier,
                "supplier_name": f"供应商{supplier}{SUPPLIER_SUFFIX}",
                "note": f"发票备注（手续费{fee}）" if fee else "发票备注",
            })
            # 第一条 recon 行 = 该付款的 ground truth（is_correct=1，final_label=付款真值；
            # UNMATCHED 付款表示"人工/系统判定无匹配"这条结论本身被确认）。
            st.recon.append({
                "payment_id": pid, "invoice_id": inv_id, "final_label": label,
                "confirmed_by": _source_for(rng), "reconciled_at": pay_date.isoformat(),
                "is_correct": 1, "role": "RESULT",
            })

            # —— 结构 flag 落表 ——
            if is_multi:      # 多对多：同一付款再核销第二张发票
                inv2_id = st.next_invoice(cp, inv_month)
                st.invoices.append({
                    "invoice_id": inv2_id, "tenant_id": TENANT,
                    "amount": _round2(amount * 0.6), "currency": "CNY",
                    "date": inv_date.isoformat(), "supplier_id": cp.cid,
                    "supplier_name": f"供应商{cp.cid}{SUPPLIER_SUFFIX}",
                    "note": "拆分发票二",
                })
                st.recon.append({
                    "payment_id": pid, "invoice_id": inv2_id, "final_label": "PARTIAL",
                    "confirmed_by": "HUMAN", "reconciled_at": pay_date.isoformat(),
                    "is_correct": 0, "role": "RESULT",   # 多对多的第二张：真实核销链接（非候选）
                })
            if is_split:      # 拆分：同一张发票被第二笔付款核销
                pid2 = f"PAY-{pay_seq + 10_000:06d}"
                amt2 = _round2(amount * 0.4)
                st.payments.append({
                    "payment_id": pid2, "tenant_id": TENANT, "doc_type": DOC_TYPE,
                    "amount": amt2, "currency": "CNY", "date": pay_date.isoformat(),
                    "payer_id": cp.cid, "channel": channel,
                    "reference": f"付款 {inv_id} 二期",
                    "created_at": (datetime.combine(pay_date, datetime.min.time())
                                   + timedelta(hours=rng.randint(0, 23))).isoformat(sep=" "),
                })
                st.recon.append({
                    "payment_id": pid2, "invoice_id": inv_id, "final_label": "PARTIAL",
                    "confirmed_by": "HUMAN", "reconciled_at": pay_date.isoformat(),
                    "is_correct": 1, "role": "RESULT",   # 第二笔付款自己的 ground truth
                })
            if not is_multi:  # 干扰候选（让候选集 >1）
                for _ in range(rng.randint(0, 3)):
                    oth = rng.choice(cps)
                    d_id = st.next_invoice(oth, month)
                    st.invoices.append({
                        "invoice_id": d_id, "tenant_id": TENANT,
                        "amount": _round2(rng.uniform(2_000, 200_000)),
                        "currency": "CNY", "date": inv_date.isoformat(),
                        "supplier_id": oth.cid,
                        "supplier_name": f"供应商{oth.cid}{SUPPLIER_SUFFIX}",
                        "note": "干扰候选",
                    })
                    st.recon.append({
                        "payment_id": pid, "invoice_id": d_id, "final_label": "UNMATCHED",
                        "confirmed_by": "NONE", "reconciled_at": pay_date.isoformat(),
                        "is_correct": 0, "role": "CANDIDATE",
                    })

    # —— >50 候选的付款（3%）：演练截断策略 ——
    for p in rng.sample(st.payments, int(len(st.payments) * 0.03)):
        if p["payer_id"].startswith("CP-"):
            month = int(p["date"][5:7])
            for j in range(55):
                oth = rng.choice(cps)
                d_id = st.next_invoice(oth, month)
                st.invoices.append({
                    "invoice_id": d_id, "tenant_id": TENANT,
                    "amount": _round2(rng.uniform(2_000, 200_000)),
                    "currency": "CNY", "date": p["date"],
                    "supplier_id": oth.cid,
                    "supplier_name": f"供应商{oth.cid}{SUPPLIER_SUFFIX}",
                    "note": "批量干扰候选",
                })
                st.recon.append({
                    "payment_id": p["payment_id"], "invoice_id": d_id,
                    "final_label": "UNMATCHED", "confirmed_by": "NONE",
                    "reconciled_at": p["date"], "is_correct": 0, "role": "CANDIDATE",
                })


def inject_defects(rng: random.Random, st: SimState) -> None:
    """质量缺陷（演练盘点 §3 与清洗规则）：重复主键/空值/未来日期/跨度>1y。"""
    defects = {}
    # 17 笔完全重复付款（created_at 不同，清洗保留最新）
    dups = rng.sample(st.payments, 17)
    for p in dups:
        dup = dict(p)
        dup["created_at"] = (datetime.fromisoformat(p["created_at"]) + timedelta(hours=1)).isoformat(sep=" ")
        st.payments.append(dup)
    defects["duplicate_payment_rows"] = 17
    # 31 条关键字段空值
    nulls = rng.sample(st.payments, 31)
    for i, p in enumerate(nulls):
        p = rng.choice(nulls)
        kind = i % 3
        if kind == 0:
            p["amount"] = None
        elif kind == 1:
            p["date"] = None
        else:
            p["payer_id"] = ""
    defects["null_critical_payments"] = 31
    # 6 笔未来日期
    for p in rng.sample(st.payments, 6):
        p["date"] = "2026-12-15"
    defects["future_date_payments"] = 6
    # 5 张跨度>1y 发票（2024 年）
    for inv in rng.sample(st.invoices, 5):
        inv["date"] = "2024-05-10"
    defects["span_gt_1y_invoices"] = 5
    # 5 笔跨度>1y 付款（2024 年，演练 A.5 span_gt_1y 与清洗 C4b）
    for i, p in enumerate(rng.sample(st.payments, 5)):
        p["date"] = "2024-06-15"
        p["created_at"] = "2024-06-15 10:00:00"
    defects["span_gt_1y_payments"] = 5
    st.defects = defects


def gen_counterparty_stats(rng: random.Random, cps: list[Counterparty]) -> list[dict]:
    """PIT 历史匹配率快照：某月 = 该月之前该主体核销为 MATCHED 的比例。

    （仿真近似：按干净度估算每月新增核销；真实库由上游 T-1 跑批落表。）"""
    rows = []
    weights = {7: 0.40, 8: 0.35, 9: 0.25}
    for cp in cps:
        hist_total, hist_matched = 0, 0
        for month in MONTHS:
            if hist_total:
                rate = _clip(hist_matched / hist_total, 0, 1)
            else:  # 无历史 → 上游缺省（特征侧 None→0.5 的统计侧近似）
                rate = round(rng.uniform(0.5, 0.9), 4)
            rows.append({"payer_id": cp.cid, "month": f"{YEAR}-{month:02d}",
                         "match_rate": round(rate, 4)})
            n_month = int(100 * weights[month])
            hist_total += n_month
            hist_matched += int(n_month * _clip(cp.clean, 0.35, 0.90))
    return rows


def _write_csv(path: Path, rows: list[dict], cols: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


PAY_COLS = ["payment_id", "tenant_id", "doc_type", "amount", "currency", "date",
            "payer_id", "channel", "reference", "created_at"]
INV_COLS = ["invoice_id", "tenant_id", "amount", "currency", "date",
            "supplier_id", "supplier_name", "note"]
REC_COLS = ["payment_id", "invoice_id", "final_label", "confirmed_by",
            "reconciled_at", "is_correct", "role"]
ST_COLS = ["payer_id", "month", "match_rate"]


def main() -> None:
    ap = argparse.ArgumentParser(description="W1 仿真原始表生成器（演练专用）")
    ap.add_argument("--out", required=True, help="输出目录（git-ignore）")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(a.seed)
    st = SimState(seed=a.seed)
    cps = _make_counterparties(rng)
    gen_core(rng, cps, st)
    inject_defects(rng, st)
    stats = gen_counterparty_stats(rng, cps)

    _write_csv(out / "payments.csv", st.payments, PAY_COLS)
    _write_csv(out / "invoices.csv", st.invoices, INV_COLS)
    _write_csv(out / "recon_results.csv", st.recon, REC_COLS)
    _write_csv(out / "counterparty_stats.csv", stats, ST_COLS)

    labels = {}
    for r in st.recon:
        if r["is_correct"]:
            labels[r["final_label"]] = labels.get(r["final_label"], 0) + 1
    sources = {}
    for r in st.recon:
        if r["is_correct"]:
            sources[r["confirmed_by"]] = sources.get(r["confirmed_by"], 0) + 1
    meta = {
        "seed": a.seed,
        "generated_at": datetime.now().isoformat(sep=" ", timespec="seconds"),
        "window": f"{YEAR}-07-01 .. {YEAR}-09-30",
        "tenant": TENANT,
        "n_counterparties": len(cps),
        "n_payments": len(st.payments),
        "n_invoices": len(st.invoices),
        "n_recon_pairs": len(st.recon),
        "correct_pair_labels": labels,
        "correct_pair_sources": sources,
        "defects_injected": st.defects,
        "redline": "演练专用：数字不得进入 G-数据 门禁 / D5 go-no-go / 模型验收（定稿 §1-D2）",
    }
    (out / "sim_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False),
                                       encoding="utf-8")
    print(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
