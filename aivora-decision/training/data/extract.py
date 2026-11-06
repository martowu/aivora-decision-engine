#!/usr/bin/env python3
"""抽取管线 CLI（定稿 §7-C）。

用法：
  python extract.py --start 2026-08-01 --end 2026-10-31 --tenant t-demo \
      --source synthetic --output ../out --salt-file /secure/salt.txt

产出（output 目录下）：
  payments.csv / invoices.csv / labels.csv
  quality_report.json
  snapshot.json          ← 含 data_snapshot_hash（锚定 decision_models 表）

重跑保证：同参数（含同盐值）重跑 → 文件字节一致、data_snapshot_hash 一致。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipe import (  # noqa: E402
    PIPELINE_VERSION,
    DataSource,
    RunConfig,
    SyntheticSource,
    mask_record,
    quality_check,
    write_output,
)


def parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def build_source(name: str, n: int) -> DataSource:
    if name == "synthetic":
        return SyntheticSource(n=n)
    raise SystemExit(f"unknown source: {name}（生产 ReconSource 待接入真实表结构）")


def main() -> None:
    ap = argparse.ArgumentParser(description="Recon 数据抽取/脱敏/质检管线")
    ap.add_argument("--start", required=True, help="时间窗起 YYYY-MM-DD")
    ap.add_argument("--end", required=True, help="时间窗止 YYYY-MM-DD")
    ap.add_argument("--tenant", required=True, help="租户 id，逗号分隔（一期单租户）")
    ap.add_argument("--doc-type", default="AP", help="AP,AR 逗号分隔")
    ap.add_argument("--source", default="synthetic", help="synthetic | recon（recon 未接入）")
    ap.add_argument("--n", type=int, default=50, help="synthetic 样本数")
    ap.add_argument("--salt-file", required=True, help="假名化盐值文件路径（不进 git）")
    ap.add_argument("--output", required=True, help="输出目录")
    ap.add_argument("--format", default="csv", choices=["csv", "parquet"])
    args = ap.parse_args()

    salt = Path(args.salt_file).read_text().strip()
    if not salt:
        raise SystemExit("salt file is empty")

    cfg = RunConfig(
        time_start=parse_date(args.start),
        time_end=parse_date(args.end),
        tenant_ids=tuple(t.strip() for t in args.tenant.split(",")),
        doc_types=tuple(d.strip().upper() for d in args.doc_type.split(",")),
        masking_salt=salt,
        output_dir=Path(args.output),
        format=args.format,
    )
    src = build_source(args.source, args.n)

    # 1. 抽取 + 脱敏
    payments = [mask_record(dict(r), cfg) for r in src.fetch_payments(cfg)]
    invoices = [mask_record(dict(r), cfg) for r in src.fetch_candidate_invoices(
        cfg, [p["payment_id"] for p in payments])]

    # 2. 质检
    q_pay = quality_check(payments, pk="payment_id", critical=("amount", "date", "payer_id"))
    q_inv = quality_check(invoices, pk="invoice_id", critical=("amount", "date", "supplier_id"))
    if not (q_pay.passed and q_inv.passed):
        print("QUALITY FAILED:", json.dumps({"payments": q_pay.to_dict(),
                                            "invoices": q_inv.to_dict()}, ensure_ascii=False),
              file=sys.stderr)
        raise SystemExit(2)

    # 3. 标签来源（不脱敏，仅来源分类）
    labels = [
        {"payment_id": p["payment_id"],
         "tenant_id": p["tenant_id"],
         "label_source": src.fetch_label_source(cfg, p["payment_id"]) or "unlabeled"}
        for p in payments
    ]

    # 4. 输出
    p_pay = write_output(payments, cfg.output_dir, "payments", cfg, pk="payment_id")
    p_inv = write_output(invoices, cfg.output_dir, "invoices", cfg, pk="invoice_id")
    p_lab = write_output(labels, cfg.output_dir, "labels", cfg, pk="payment_id")

    # 5. 快照 hash（对全部脱敏行计算，锚定 decision_models.data_snapshot_hash）
    snap_hash = _hash_all(payments, invoices, labels, cfg)

    snap = {
        "data_snapshot_hash": snap_hash,
        "pipeline_version": PIPELINE_VERSION,
        "run_fingerprint": cfg.fingerprint(),
        "source": src.name,
        "counts": {"payments": len(payments), "invoices": len(invoices), "labels": len(labels)},
        "quality": {"payments": q_pay.to_dict(), "invoices": q_inv.to_dict()},
        "files": {"payments": str(p_pay), "invoices": str(p_inv), "labels": str(p_lab)},
    }
    (cfg.output_dir / "snapshot.json").write_text(
        json.dumps(snap, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"data_snapshot_hash": snap_hash,
                      "counts": snap["counts"],
                      "quality_passed": True}, ensure_ascii=False))


def _hash_all(payments, invoices, labels, cfg: RunConfig) -> str:
    """对三类脱敏行分别稳定化后拼接计算 hash（行内带类型前缀避免跨类主键碰撞）。

    排序键 = (_rowtype, 主键)，保证全序唯一 → 重跑字节级一致。
    """
    tagged = (
        [{"_rowtype": "pay", **r} for r in payments]
        + [{"_rowtype": "inv", **r} for r in invoices]
        + [{"_rowtype": "lab", **r} for r in labels]
    )
    tagged.sort(key=lambda r: (r["_rowtype"],
                               str(r.get("payment_id") or ""),
                               str(r.get("invoice_id") or "")))
    h = hashlib.sha256()
    h.update(f"pipe:{PIPELINE_VERSION}\n".encode())
    h.update(f"run:{cfg.fingerprint()}\n".encode())
    for r in tagged:
        h.update((json.dumps(dict(sorted(r.items())), ensure_ascii=False, default=str) + "\n").encode())
    return h.hexdigest()


if __name__ == "__main__":
    main()
