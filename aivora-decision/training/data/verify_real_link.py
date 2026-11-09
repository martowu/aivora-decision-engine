#!/usr/bin/env python3
"""真实 Recon 数据（seed）→ PairInput → build_features → 校准推理 链路验证。

红线：seed 数据仅验证管线/特征/推理链路可跑通，指标不入门禁（定稿 §1-D2）。
工件用 train.py 自检产物（合成数据训练），仅证明"真实输入能进模型并出校准概率"。
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PIPE = HERE.parent / "pipelines"
SERVICE = HERE.parent.parent / "service"
sys.path.insert(0, str(PIPE))
sys.path.insert(0, str(SERVICE))
sys.path.insert(0, str(HERE))

from features import PairInput, build_features  # noqa: E402
from inference import load_bundle, predict  # noqa: E402

RAW = HERE.parent / "out" / "recon-seed"
ART = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(r"D:\tmp\selftest-artifact")


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def main() -> None:
    pays = {r["payment_id"]: r for r in csv.DictReader(
        (RAW / "payments.csv").open(encoding="utf-8"))}
    invs = {r["invoice_id"]: r for r in csv.DictReader(
        (RAW / "invoices.csv").open(encoding="utf-8"))}
    ress = list(csv.DictReader((RAW / "recon_results.csv").open(encoding="utf-8")))
    stats = {}
    for r in csv.DictReader((RAW / "counterparty_stats.csv").open(encoding="utf-8")):
        stats[(r["payer_id"], r["month"])] = float(r["match_rate"])

    # 每付款的 RESULT 行数（多对多 flag）
    n_per_pay = {}
    for r in ress:
        n_per_pay[r["payment_id"]] = n_per_pay.get(r["payment_id"], 0) + 1

    bundle = load_bundle(ART)
    print(f"工件: {bundle.model_version}  特征维: {bundle.num_feature}")
    print(f"{'付款(前8)':<10}{'发票号':<10}{'pay':>8}{'inv':>8}{'币种':<9}"
          f"{'票号入附言':>8}{'币种不符':>8}  {'校准概率 M/P/U':<24}  top-1")
    print("-" * 100)
    for r in ress:
        p, i = pays[r["payment_id"]], invs[r["invoice_id"]]
        payer_id = p["payer_id"]
        month = (p["date"] or "")[:7]
        rate = stats.get((payer_id, month), 0.5)
        # is_first_counterparty：该 payer 在此月无更早历史
        earlier = [k for k in stats if k[0] == payer_id and k[1] < month]
        first = len(earlier) == 0
        pair = PairInput(
            payment_id=p["payment_id"][:8], invoice_id=r["invoice_id"][:8],
            pay_amount=_f(p["amount"]), inv_amount=_f(i["amount"]),
            pay_currency=p.get("currency", "USD"), inv_currency=i.get("currency", "USD"),
            pay_date=p["date"], inv_date=i["date"],
            payer_id=payer_id[:8], supplier_id=i["supplier_id"][:8],
            supplier_name=i.get("supplier_name", ""),
            channel=p.get("channel", "BANK_TRANSFER"),
            reference=p.get("reference", ""), note=i.get("note", ""),
            fee=0.0,
            counterparty_match_rate_hist=rate,
            is_first_counterparty=first,
            is_multi_mapping=n_per_pay.get(r["payment_id"], 0) >= 2,
            has_split_merge=False,
            invoice_number=i.get("invoice_number", ""),
        )
        out = predict(bundle, pair)
        f = out["features"]
        prob = " ".join(f"{v:.3f}" for v in out["probs"])
        print(f"{pair.payment_id:<10}{i.get('invoice_number',''):<10}"
              f"{pair.pay_amount:>8.0f}{pair.inv_amount:>8.0f}"
              f"{pair.pay_currency+'/'+pair.inv_currency:<9}"
              f"{int(f['invoice_number_in_reference']):>10}"
              f"{int(f['currency_mismatch']):>8}  {prob:<24}  {out['label']}")
    print("\n（红线：seed 数据仅验证链路，概率来自合成训练工件，不代表真实效果）")


if __name__ == "__main__":
    main()
