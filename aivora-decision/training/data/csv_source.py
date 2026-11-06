"""CsvReconSource：从 CSV 原始表接入 §7-C 抽取管线（DataSource 实现）。

两个用途：
1. **W1 演练**：读 simulate_w1.py 产出的仿真原始表，走完整管线
   （清洗 → 脱敏 → 质检 → 快照 hash）；
2. **生产接入点**：真实 Recon 按同 schema 导出 CSV 后，零改动接入
   （定稿 §7-C / training/data/README"生产接入点"）。

清洗规则（对应数据盘点报告 §3"需清洗"项，规则固化在此，可审计）：
- C1 付款主键重复 → 保留 created_at 最新一条；
- C2 付款关键字段空值（amount 空/≤0、date 空、payer_id 空）→ 剔除；
- C3 付款日期在未来（> 运行当天）→ 剔除；
- C4 发票跨度异常（invoice.date < 运行当天 − 1 年）→ 剔除该发票及其 recon 行；
- C4b 付款跨度异常（payment.date < 运行当天 − 1 年）→ 剔除及其 recon 行；
- C5 清洗后孤立的 recon 行（付款或发票已被剔除）→ 剔除。

recon_results 的 `role` 列（仿真/演练口径）：RESULT = 真实核销结果链接
（ground truth 行 + 多对多/拆分的真实关联行）；CANDIDATE = 初筛候选（干扰项）。
标签来源只取 RESULT 行；候选集统计取全部行。真实库无 role 列时（每付款一行
结果表），两个口径自然合一。

清洗计数记录在 `source.last_cleaning`，供盘点报告 §3 回填。
"""

from __future__ import annotations

import csv
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from pipe import DataSource, RunConfig


class CsvReconSource(DataSource):
    name = "csv"

    def __init__(self, csv_dir: str | Path):
        d = Path(csv_dir)
        self.payments = self._read(d / "payments.csv")
        self.invoices = self._read(d / "invoices.csv")
        self.recon = self._read(d / "recon_results.csv")
        self.last_cleaning: dict[str, int] = {}
        self._clean()
    @staticmethod
    def _read(path: Path) -> list[dict]:
        with path.open(encoding="utf-8") as f:
            return [dict(r) for r in csv.DictReader(f)]

    # ------------------------------------------------------------------
    def _clean(self) -> None:
        c = {"duplicate_payment_rows": 0, "null_critical_payments": 0,
             "future_date_payments": 0, "span_gt_1y_invoices": 0,
             "span_gt_1y_payments": 0, "orphan_recon_rows": 0}

        # C1 去重（保留 created_at 最新）
        by_pid: dict[str, dict] = {}
        for p in self.payments:
            pid = p.get("payment_id") or ""
            old = by_pid.get(pid)
            if old is None:
                by_pid[pid] = p
            else:
                c["duplicate_payment_rows"] += 1
                if (p.get("created_at") or "") > (old.get("created_at") or ""):
                    by_pid[pid] = p
        self.payments = list(by_pid.values())

        # C2 关键字段空值
        keep = []
        for p in self.payments:
            amt = p.get("amount")
            try:
                amt_ok = amt not in (None, "") and float(amt) > 0
            except (TypeError, ValueError):
                amt_ok = False
            if not (amt_ok and p.get("date") and p.get("payer_id")):
                c["null_critical_payments"] += 1
                continue
            keep.append(p)
        self.payments = keep

        # C3 未来日期
        today = date.today()
        keep = []
        for p in self.payments:
            if datetime.strptime(p["date"], "%Y-%m-%d").date() > today:
                c["future_date_payments"] += 1
                continue
            keep.append(p)
        self.payments = keep

        # C4 跨度>1y 发票（相对运行当天，避免把盘点窗前的正常历史数据误删）
        horizon = today - timedelta(days=365)
        keep_inv = []
        for i in self.invoices:
            if i.get("date") and datetime.strptime(i["date"], "%Y-%m-%d").date() < horizon:
                c["span_gt_1y_invoices"] += 1
                continue
            keep_inv.append(i)
        self.invoices = keep_inv

        # C4b 跨度>1y 付款
        keep = []
        for p in self.payments:
            if p.get("date") and datetime.strptime(p["date"], "%Y-%m-%d").date() < horizon:
                c["span_gt_1y_payments"] += 1
                continue
            keep.append(p)
        self.payments = keep

        # C5 孤立 recon 行
        pids = {p["payment_id"] for p in self.payments}
        iids = {i["invoice_id"] for i in self.invoices}
        keep_rec = []
        for r in self.recon:
            if r.get("payment_id") in pids and r.get("invoice_id") in iids:
                keep_rec.append(r)
            else:
                c["orphan_recon_rows"] += 1
        self.recon = keep_rec

        self.last_cleaning = c

    # ------------------------------------------------------------------
    def _in_window(self, row: Mapping[str, Any], cfg: RunConfig,
                   field: str = "date") -> bool:
        if cfg.tenant_ids and row.get("tenant_id") not in cfg.tenant_ids:
            return False
        if cfg.doc_types and row.get("doc_type") and \
                row["doc_type"].upper() not in cfg.doc_types:
            return False
        d = row.get(field)
        if not d:
            return False
        return cfg.time_start.isoformat() <= d <= cfg.time_end.isoformat()

    def fetch_payments(self, cfg: RunConfig) -> Iterable[Mapping[str, Any]]:
        for p in self.payments:
            if self._in_window(p, cfg):
                yield dict(p)

    def fetch_candidate_invoices(
        self, cfg: RunConfig, payment_ids: Sequence[str]
    ) -> Iterable[Mapping[str, Any]]:
        pids = set(payment_ids)
        seen: set[str] = set()
        inv_by_id = {i["invoice_id"]: i for i in self.invoices}
        for r in self.recon:
            if r.get("payment_id") in pids and r.get("invoice_id") in inv_by_id:
                iid = r["invoice_id"]
                if iid not in seen:
                    seen.add(iid)
                    yield dict(inv_by_id[iid])

    def fetch_label_source(self, cfg: RunConfig, payment_id: str) -> str | None:
        """标签来源（定稿 §3-8）：只取该付款的 RESULT 行（真实核销结果），
        优先 is_correct=1 的 ground truth 行；CANDIDATE（初筛干扰）行不作标签来源。
        真实库（每付款一行结果表）无 role 列时退化为全行，语义不变。"""
        mapping = {"HUMAN": "human_review",
                   "HISTORICAL_CONFIRMED": "historical_confirmed",
                   "RULE_AUTO": "rule_inferred",
                   "NONE": None}
        rows = [r for r in self.recon if r.get("payment_id") == payment_id
                and r.get("role", "RESULT") == "RESULT"]
        for r in rows:
            if str(r.get("is_correct", "")) == "1":
                return mapping.get(r.get("confirmed_by"))
        for r in rows:
            if r.get("final_label") != "UNMATCHED":
                return mapping.get(r.get("confirmed_by"))
        return None
