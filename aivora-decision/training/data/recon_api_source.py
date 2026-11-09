#!/usr/bin/env python3
"""ReconApiSource —— 从 Aivora Recon 真实 API 拉取数据，转储为 W1 演练用的 4 份原始 CSV。

设计目标（定稿 §7-B Shadow 对接的前置 + §7-C 可重跑管线）：
把 Recon 的 REST API 数据转换成与 `simulate_w1.py` **完全相同 schema** 的原始表，
从而让既有管线零改动复用：
    extract.py --source csv --csv-dir <本脚本产出>      → 脱敏/质检/快照 hash
    sim_w1_reports.py --raw <本脚本产出>                 → 基线/工作坊演练
    sim_inventory_duckdb.py（对 <本脚本产出> 建表）        → 盘点 SQL

只读红线：本脚本只调用 GET + 登录 POST，绝不触发 /matching/run、
PATCH /matching/{id}/{action}、settings 等任何写操作。

凭证：从环境变量读取，不进 argv、不落盘
    RECON_API_BASE     默认 https://recon.ivlacloud.com
    RECON_API_EMAIL    必填
    RECON_API_PASSWORD 必填

字段/标签映射（⚠ 标签语义为**演练默认**，须标签工作坊确认）：
  Recon matching_results 每行 = 一个 (付款, 候选发票) 对：
  - 该付款的"胜出"对（status ∈ CONFIRMED/AUTO_MATCHED）→ role=RESULT, is_correct=1
  - 同一付款的其余对（引擎产出但未选中）→ role=RESULT, is_correct=0
  - final_label：
      EXACT + difference NONE 且胜出      → MATCHED
      match_type PARTIAL / difference ∈ {FX,SHORT_PAY,OVER_PAY,FEE,ROUNDING} → PARTIAL
      EXACT + NONE 但未胜出（干扰对）       → UNMATCHED（假设，待工作坊确认）
      无结果行的付款                        → 不产对（fetch_label_source=unlabeled）
  - confirmed_by：CONFIRMED→HUMAN，AUTO_MATCHED→RULE_AUTO，PENDING_REVIEW→NONE
  - payer_id：付款方名称解析到 customers（normalized_name/aliases），未解析→UNK-<hash8>
  - channel：source BANK_FEED→BANK_TRANSFER，其余→OTHER
  - reference：payment_reference（含票号，如 INV-1005——对应 invoice_number_in_reference 特征）
  - counterparty_stats：PIT 历史匹配率（付款日之前该 payer 的胜出对占比；无历史→0.5 缺省）

用法：
  RECON_API_EMAIL=... RECON_API_PASSWORD=... \\
    python recon_api_source.py --out ../out/recon-seed [--limit 0]
  （--limit 0=全量；演练可先小量）
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

DOC_TYPE = "AR"   # Recon 实际业务=应收核销（客户付款 ↔ AR 发票），非 AP
TENANT_FALLBACK = "unknown-tenant"

# difference_type → 是否"可解释差异"（PARTIAL 信号）
PARTIAL_DIFFS = {"FX", "SHORT_PAY", "OVER_PAY", "FEE", "ROUNDING", "TAX"}
# 胜出状态（引擎/人工认定的最终匹配）
WINNING_STATUS = {"CONFIRMED", "AUTO_MATCHED"}
# confirmed_by 映射
CONFIRMED_BY = {"CONFIRMED": "HUMAN", "AUTO_MATCHED": "RULE_AUTO", "PENDING_REVIEW": "NONE"}
# channel 映射
CHANNEL = {"BANK_FEED": "BANK_TRANSFER", "MANUAL": "OTHER", "API": "OTHER"}


def _norm_name(s: str) -> str:
    return "".join(ch for ch in (s or "") if ch.isalnum()).upper()


def _hash8(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:8]


class ReconApiSource:
    """只读访问 Recon REST API，缓存到内存，按需转储 CSV。"""

    def __init__(self, base: str, email: str, password: str, timeout: float = 90.0):
        self.base = base.rstrip("/")
        self._client = httpx.Client(base_url=self.base, timeout=timeout, follow_redirects=True)
        self._email = email
        self._password = password
        self._authed = False
        # 缓存
        self.payments: list[dict] = []
        self.invoices: list[dict] = []
        self.customers: list[dict] = []
        self.results: list[dict] = []
        self.tenant_id: str = TENANT_FALLBACK

    # ------------------------------------------------------------------ auth
    def auth(self) -> None:
        r = self._client.post("/api/auth/login",
                              json={"email": self._email, "password": self._password})
        if r.status_code != 200:
            raise SystemExit(f"登录失败 HTTP {r.status_code}: {r.text[:200]}")
        me = self._client.get("/api/auth/me").json()
        if not me.get("success"):
            raise SystemExit(f"/auth/me 异常: {str(me)[:200]}")
        self.tenant_id = me["data"].get("tenant_id", TENANT_FALLBACK)
        self.tenant_name = me["data"].get("tenant_name", "")
        self._authed = True

    # ------------------------------------------------------------------ fetch
    def _get(self, path: str) -> list | dict:
        if not self._authed:
            self.auth()
        r = self._client.get(path)
        if r.status_code != 200:
            raise SystemExit(f"GET {path} HTTP {r.status_code}: {r.text[:200]}")
        j = r.json()
        return j.get("data", j)

    def fetch_all(self) -> None:
        if self._authed and self.payments:
            return
        self.auth()
        self.payments = self._get("/api/payments")
        self.invoices = self._get("/api/invoices")
        self.customers = self._get("/api/customers")
        self.results = self._get("/api/matching/results")

    # ------------------------------------------------------------- resolution
    def _resolve_payer(self, payer_name: str) -> str:
        """付款方名称 → customer id（normalized_name 或 aliases 命中），否则 UNK-<hash8>。"""
        target = _norm_name(payer_name)
        for c in self.customers:
            if target == _norm_name(c.get("normalized_name")) or target == _norm_name(c.get("name")):
                return c["id"]
            for a in (c.get("aliases") or []):
                if target == _norm_name(a):
                    return c["id"]
        return f"UNK-{_hash8(payer_name or '?')}"

    # ------------------------------------------------------------ label logic
    def _label_for(self, res: dict, is_winning: bool) -> str:
        mtype = (res.get("match_type") or "").upper()
        diff = (res.get("difference_type") or "").upper()
        if mtype == "PARTIAL" or diff in PARTIAL_DIFFS:
            return "PARTIAL"
        # EXACT + NONE（或空 diff）
        if is_winning:
            return "MATCHED"
        # 未胜出的 EXACT/NONE 对 = 干扰候选（标签假设，工作坊确认）
        return "UNMATCHED"

    # ------------------------------------------------------------------ dump
    def dump(self, out_dir: Path, limit: int = 0) -> dict:
        self.fetch_all()
        out_dir.mkdir(parents=True, exist_ok=True)

        pays = self.payments[:limit] if limit > 0 else self.payments
        keep_pay = {p["id"] for p in pays}
        inv_by_id = {i["id"]: i for i in self.invoices}
        res_by_pay: dict[str, list[dict]] = defaultdict(list)
        for r in self.results:
            if r.get("payment_id") in keep_pay:
                res_by_pay[r["payment_id"]].append(r)

        # ---- payments.csv（对齐 simulate_w1 PAY_COLS）----
        pay_rows = []
        for p in pays:
            pay_rows.append({
                "payment_id": p["id"], "tenant_id": p.get("tenant_id", self.tenant_id),
                "doc_type": DOC_TYPE,
                "amount": p.get("amount"), "currency": p.get("currency", "USD"),
                "date": p.get("payment_date"),
                "payer_id": self._resolve_payer(p.get("payer_name", "")),
                "channel": CHANNEL.get((p.get("source") or "").upper(), "OTHER"),
                "reference": p.get("payment_reference") or "",
                "created_at": p.get("created_at", ""),
                "_payer_name": p.get("payer_name", ""),   # 中间列，写 CSV 前剔除
            })

        # ---- invoices.csv + recon_results.csv ----
        inv_rows, rec_rows = [], []
        seen_inv: set[str] = set()
        for pid, rows in res_by_pay.items():
            winning = [r for r in rows if (r.get("status") or "").upper() in WINNING_STATUS]
            win_ids = {r["invoice_id"] for r in winning}
            for r in rows:
                iid = r["invoice_id"]
                inv = inv_by_id.get(iid, {})
                if iid not in seen_inv:
                    seen_inv.add(iid)
                    inv_rows.append({
                        "invoice_id": iid,
                        "tenant_id": inv.get("tenant_id", self.tenant_id),
                        "amount": inv.get("amount", r.get("matched_amount")),
                        "currency": inv.get("currency", r.get("invoice_currency")),
                        "date": inv.get("invoice_date"),
                        "supplier_id": inv.get("customer_id") or
                                       self._resolve_payer(inv.get("customer_name", "")),
                        "supplier_name": inv.get("customer_name", ""),
                        "note": "",
                        "invoice_number": inv.get("invoice_number", ""),
                    })
                rec_rows.append({
                    "payment_id": pid, "invoice_id": iid,
                    "final_label": self._label_for(r, r["invoice_id"] in win_ids and bool(winning)),
                    "confirmed_by": CONFIRMED_BY.get((r.get("status") or "").upper(), "NONE"),
                    "reconciled_at": (r.get("created_at") or "")[:10],
                    "is_correct": 1 if r["invoice_id"] in win_ids and bool(winning) else 0,
                    "role": "RESULT",
                    # 保留引擎事实供审计/工作坊
                    "match_type": r.get("match_type"), "match_score": r.get("match_score"),
                    "difference_type": r.get("difference_type"),
                    "difference_amount": r.get("difference_amount"),
                    "status": r.get("status"), "reason": r.get("reason"),
                })

        # 无结果行的付款：其候选发票（若有同 payer 的 OPEN 发票）作为干扰对
        # ——演练从简：不臆造，仅记录（fetch_label_source=unlabeled）

        # ---- counterparty_stats.csv（PIT 历史匹配率）----
        # 对每个 payer 的每个活跃月：该月**开始之前**该 payer 付款中"有胜出对"的占比；
        # 无历史 → 0.5 缺省（与特征侧 None→0.5 对齐）。
        payer_of_pay: dict[str, str] = {}
        for p in pays:
            payer_of_pay[p["id"]] = self._resolve_payer(p.get("payer_name", ""))
        events: dict[str, list[tuple[str, bool]]] = defaultdict(list)
        for p in pays:
            rows = res_by_pay.get(p["id"], [])
            has_win = any((r.get("status") or "").upper() in WINNING_STATUS for r in rows)
            events[payer_of_pay[p["id"]]].append((p.get("payment_date") or "", has_win))
        stats_rows = []
        for payer, evs in events.items():
            evs.sort(key=lambda e: e[0])
            for m in sorted({d[:7] for d, _ in evs if d}):
                prior = [w for d, w in evs if d and d[:7] < m]
                rate = (sum(prior) / len(prior)) if prior else 0.5
                stats_rows.append({"payer_id": payer, "month": m, "match_rate": round(rate, 4)})

        self._write_csv(out_dir / "payments.csv", pay_rows,
                        ["payment_id", "tenant_id", "doc_type", "amount", "currency",
                         "date", "payer_id", "channel", "reference", "created_at"])
        self._write_csv(out_dir / "invoices.csv", inv_rows,
                        ["invoice_id", "invoice_number", "tenant_id", "amount", "currency",
                         "date", "supplier_id", "supplier_name", "note"])
        self._write_csv(out_dir / "recon_results.csv", rec_rows,
                        ["payment_id", "invoice_id", "final_label", "confirmed_by",
                         "reconciled_at", "is_correct", "role", "match_type",
                         "match_score", "difference_type", "difference_amount",
                         "status", "reason"])
        self._write_csv(out_dir / "counterparty_stats.csv", stats_rows,
                        ["payer_id", "month", "match_rate"])

        meta = {
            "source": "recon-api", "base": self.base, "tenant_id": self.tenant_id,
            "tenant_name": self.tenant_name, "doc_type": DOC_TYPE,
            "n_payments": len(pay_rows), "n_invoices": len(inv_rows),
            "n_results": len(rec_rows), "n_counterparties": len(events),
            "label_note": "final_label 为演练默认映射，须标签工作坊确认（见模块 docstring）",
            "readonly": True,
            "retrieved_at": date.today().isoformat(),
        }
        (out_dir / "sim_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False),
                                               encoding="utf-8")
        self._client.close()
        return meta

    @staticmethod
    def _write_csv(path: Path, rows: list[dict], cols: list[str]) -> None:
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow(r)


def main() -> None:
    ap = argparse.ArgumentParser(description="Recon API → 原始 CSV 转储（只读）")
    ap.add_argument("--out", required=True, help="输出目录（git-ignore）")
    ap.add_argument("--limit", type=int, default=0, help="付款条数上限（0=全量）")
    ap.add_argument("--base", default=os.environ.get("RECON_API_BASE", "https://recon.ivlacloud.com"))
    a = ap.parse_args()
    email = os.environ.get("RECON_API_EMAIL", "")
    password = os.environ.get("RECON_API_PASSWORD", "")
    if not email or not password:
        raise SystemExit("需设置 RECON_API_EMAIL / RECON_API_PASSWORD 环境变量")
    src = ReconApiSource(a.base, email, password)
    meta = src.dump(Path(a.out), a.limit)
    print(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
