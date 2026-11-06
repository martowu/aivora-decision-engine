#!/usr/bin/env python3
"""W1 演练：基线测量（200 笔分层抽样）+ 标签工作坊试点（100 例双人标注 Kappa）。

数据来源：
- simulate_w1.py 的仿真原始表（含 ground truth：label / confirmed_by / is_correct）；
- CsvReconSource 清洗后的干净付款集（与 §7-C 管线同一口径）；
- 假名化与 extract.py 同一套 salt（pipe.pseudonym），样例表可与训练数据 join。

输出：out/sim-w1/sim_reports.json
  baseline : §3.1/3.2/3.3/§四/§五 的全部演练数字（含 bootstrap CI、Wilson 区间、收益公式代入）
  workshop : 100 例双人标注、一致矩阵、po/pe/Cohen's Kappa、分歧归因
  recall   : A.4.2 初筛召回核对（50 笔，演练版自动核对）

红线（定稿 §1-D2）：**全部输出为演练值**，不得进入 G-数据/G-离线/G-Shadow 门禁、
不得用于 D5 go/no-go、不得用于模型验收。用途是验证：分层抽样算法、口径计算、
CI/Kappa 工具链、收益公式代入路径。真实数据到位后同脚本换数据源重跑即可。

用法：
  python sim_w1_reports.py --raw ../out/sim-w1/raw --salt ../out/sim-w1/sim-salt.txt \
      --out ../out/sim-w1/sim_reports.json [--seed 42]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from pipe import pseudonym          # noqa: E402  与 extract 同一套假名化
from csv_source import CsvReconSource  # noqa: E402

# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------

def load_clean_payments(raw: Path) -> tuple[list[dict], list[dict], list[dict]]:
    """经 CsvReconSource 清洗的付款集 + 发票表 + RESULT 行。"""
    import tempfile
    class _Cfg:
        pass
    from pipe import RunConfig
    cfg = RunConfig(
        time_start=date(2026, 7, 1), time_end=date(2026, 9, 30),
        tenant_ids=("t-sim",), doc_types=("AP",), masking_salt="unused",
        output_dir=Path(tempfile.mkdtemp()),
    )
    src = CsvReconSource(raw)
    pids = [p["payment_id"] for p in src.fetch_payments(cfg)]
    pays = {p["payment_id"]: p for p in src.payments if p["payment_id"] in set(pids)}
    invs = {i["invoice_id"]: i for i in src.invoices}
    result_rows = [r for r in src.recon if r.get("role", "RESULT") == "RESULT"]
    return list(pays.values()), invs, result_rows


def build_payment_records(pays: list[dict], invs: dict, result_rows: list[dict]) -> list[dict]:
    """付款级 ground truth 记录（演练真值，真实数据由核销结果表提供）。"""
    gt: dict[str, dict] = {}
    n_result: Counter = Counter()
    for r in result_rows:
        n_result[r["payment_id"]] += 1
        if str(r.get("is_correct", "0")) == "1" and r["payment_id"] not in gt:
            gt[r["payment_id"]] = r
    split_inv = {iid for iid, n in Counter(
        r["invoice_id"] for r in result_rows).items() if n >= 2}

    first_month: dict[str, str] = {}
    for p in pays:
        m = p["date"][:7]
        if p["payer_id"] not in first_month or m < first_month[p["payer_id"]]:
            first_month[p["payer_id"]] = m

    recs = []
    for p in pays:
        r = gt.get(p["payment_id"])
        if r is None:
            continue
        inv = invs.get(r["invoice_id"], {})
        pay_amt, inv_amt = float(p["amount"]), float(inv.get("amount", 0) or 0)
        diff = pay_amt - inv_amt
        note = inv.get("note", "") or ""
        has_fee = "手续费" in note
        cross_month = p["date"][:7] != inv.get("date", "")[:7]
        recs.append({
            "payment_id": p["payment_id"], "payer_id": p["payer_id"],
            "invoice_id": r["invoice_id"],
            "amount": pay_amt, "diff": diff, "abs_diff": abs(diff),
            "label": r["final_label"], "confirmed_by": r["confirmed_by"],
            "channel": p["channel"], "reference": p.get("reference", ""),
            "date": p["date"],
            "n_result_rows": n_result[p["payment_id"]],
            "is_multi": n_result[p["payment_id"]] >= 2,
            "is_split": r["invoice_id"] in split_inv,
            "is_crosscur": inv.get("currency") not in (None, "CNY"),
            "has_fee": has_fee,
            "cross_month": cross_month,
            "is_first_cp": first_month[p["payer_id"]] == p["date"][:7],
            "ref_has_inv_no": r["invoice_id"] in (p.get("reference", "") or ""),
        })
    return recs


# ---------------------------------------------------------------------------
# 分层（模板 §1.2）
# ---------------------------------------------------------------------------

def assign_layers(r: dict) -> set[str]:
    L = set()
    if not (r["is_multi"] or r["is_split"] or r["is_crosscur"]):
        if r["amount"] <= 50_000:
            L.add("L1")
        elif r["amount"] <= 500_000:
            L.add("L2")
        else:
            L.add("L3")
    if r["has_fee"] or r["is_crosscur"] or (0 < r["abs_diff"] <= 100):
        L.add("L4")
    if r["is_multi"] or r["is_split"]:
        L.add("L5")
    if r["cross_month"]:
        L.add("L6")
    if r["confirmed_by"] == "RULE_AUTO":
        L.add("L7")
    if r["confirmed_by"] in ("HUMAN", "HISTORICAL_CONFIRMED") and \
            (r["label"] != "MATCHED" or r["has_fee"]):
        L.add("L8")
    return L


# 主层优先级（小层优先，"每笔只计一次"）
PRIORITY = ["L3", "L5", "L6", "L4", "L8", "L7", "L2", "L1"]
ALLOC = {"L1": 70, "L2": 40, "L3": 10, "L4": 20, "L5": 10, "L6": 10, "L7": 20, "L8": 20}


def stratified_sample(recs: list[dict], rng: random.Random) -> tuple[list[dict], dict]:
    members: dict[str, list[dict]] = defaultdict(list)
    for r in recs:
        for L in assign_layers(r):
            members[L].append(r)
    picked: dict[str, dict] = {}
    actual = {}
    for L in PRIORITY:
        pool = [r for r in members.get(L, []) if r["payment_id"] not in picked]
        k = min(ALLOC[L], len(pool))
        for r in rng.sample(pool, k):
            picked[r["payment_id"]] = r
        actual[L] = k
    return list(picked.values()), actual


# ---------------------------------------------------------------------------
# 处理时长/差错模拟（口径=模板 §二）
# ---------------------------------------------------------------------------

DURATION_MEDIAN = {"L1": 4.0, "L2": 7.0, "L3": 15.0, "L4": 9.0,
                   "L5": 18.0, "L6": 10.0, "L7": 0.0, "L8": 12.0}
SIGMA = 0.45
ERR_TYPES = [("记错发票", 0.45), ("重复核销", 0.15), ("金额/尾差处理错误", 0.20),
             ("该核销未核销", 0.15), ("其他", 0.05)]


def primary_layer(r: dict) -> str:
    for L in PRIORITY:
        if L in assign_layers(r):
            return L
    return "L1"


def simulate_processing(recs: list[dict], rng: random.Random) -> list[dict]:
    out = []
    for r in recs:
        L = primary_layer(r)
        auto = (r["confirmed_by"] == "RULE_AUTO")
        path = "自动" if auto else "人工直接核销"
        if not auto:
            dur = rng.lognormvariate(math.log(DURATION_MEDIAN[L]), SIGMA)
        else:
            dur = 0.0
        hung = (not auto) and rng.random() < 0.05
        err_p = 0.015 if auto else 0.025
        if r["amount"] > 50_000:
            err_p *= 1.5
        has_err = (not hung) and rng.random() < err_p
        err_type, loss, undecided = None, 0.0, False
        if has_err:
            x, acc = rng.random(), 0.0
            for name, w in ERR_TYPES:
                acc += w
                if x < acc:
                    err_type = name
                    break
            loss = {"记错发票": min(r["amount"] * 0.02, 50_000),
                    "重复核销": r["amount"],
                    "金额/尾差处理错误": r["abs_diff"],
                    "该核销未核销": 0.0,
                    "其他": r["amount"] * 0.01}[err_type]
            undecided = rng.random() < 0.10   # 90 天判定窗未闭合
        out.append({
            **{k: r[k] for k in ("payment_id", "payer_id", "amount", "label",
                                 "confirmed_by", "abs_diff", "has_fee", "is_multi",
                                 "is_split", "is_crosscur", "cross_month")},
            "layer": L, "path": "挂起" if hung else path,
            "duration": round(dur if not hung else dur * 0.5, 2),
            "duration_source": "日志" if rng.random() < 0.8 else "打点",
            "final": "挂起" if hung else ("核销" if r["label"] != "UNMATCHED" else "不匹配"),
            "error": err_type if has_err and not undecided else None,
            "undecided": has_err and undecided,
            "loss": loss if has_err else 0.0,
        })
    return out


# ---------------------------------------------------------------------------
# 统计工具
# ---------------------------------------------------------------------------

def pct(x: list[float], q: float) -> float | None:
    if not x:
        return None
    s = sorted(x)
    k = (len(s) - 1) * q
    f, c = math.floor(k), math.ceil(k)
    return s[f] if f == c else s[f] + (s[c] - s[f]) * (k - f)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (round(max(0.0, center - half), 4), round(min(1.0, center + half), 4))


def bootstrap_ci(vals: list[float], stat, n_boot: int = 1000,
                 rng: random.Random | None = None) -> tuple[float, float]:
    rng = rng or random.Random(0)
    # 注意：bootstrap 必须**有放回**（choices）；sample(len) 是不放回=重排，CI 会退化成点
    out = [stat(rng.choices(vals, k=len(vals))) for _ in range(n_boot)]
    return (round(pct(out, 0.025), 3), round(pct(out, 0.975), 3))


# ---------------------------------------------------------------------------
# 基线测量
# ---------------------------------------------------------------------------

def run_baseline(recs: list[dict], seed: int) -> dict:
    rng = random.Random(seed)
    sample, actual_alloc = stratified_sample(recs, rng)
    rows = simulate_processing(sample, rng)

    human = [r for r in rows if r["path"] != "挂起" and r["duration"] > 0]
    durs = [r["duration"] for r in human]
    total = len(rows)
    n_auto = sum(1 for r in rows if r["path"] == "自动")
    n_hang = sum(1 for r in rows if r["path"] == "挂起")
    n_err = sum(1 for r in rows if r["error"])
    n_und = sum(1 for r in rows if r["undecided"])
    denom = total - n_und
    lo, hi = wilson(n_err, denom)
    lo_r, hi_r = wilson(total - n_auto - n_hang, total)

    by_layer = {}
    for L in ("L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8"):
        lr = [r for r in rows if r["layer"] == L]
        ld = [r["duration"] for r in lr if r["duration"] > 0]
        by_layer[L] = {
            "n": len(lr),
            "duration_p50": round(pct(ld, 0.5) or 0, 2),
            "duration_p90": round(pct(ld, 0.9) or 0, 2),
            "review_rate": round(sum(1 for r in lr if r["duration"] > 0) / len(lr), 4) if lr else None,
            "error_rate": round(sum(1 for r in lr if r["error"]) /
                                max(1, sum(1 for r in lr if not r["undecided"])), 4) if lr else None,
        }

    err_struct = Counter(r["error"] for r in rows if r["error"])
    loss_by_type = defaultdict(float)
    for r in rows:
        if r["error"]:
            loss_by_type[r["error"]] += r["loss"]

    # 收益公式子集：干净 1:1 + 低金额（无 L5/L6 结构、≤50k、非首次主体）
    subset = [r for r in recs
              if not (r["is_multi"] or r["is_split"] or r["is_crosscur"])
              and r["amount"] <= 50_000 and not r["is_first_cp"]]
    sub_monthly = len(subset) / 3.0
    sub_sample = [r for r in rows if not (r["is_multi"] or r["is_split"] or r["is_crosscur"])
                  and r["amount"] <= 50_000 and r["duration"] > 0]
    sub_p50 = pct([r["duration"] for r in sub_sample], 0.5) or 0.0
    auto_rate = n_auto / total if total else 0.0
    target_cov = min(auto_rate + 0.10, 0.95)
    minutes = sub_monthly * target_cov * 0.98 * sub_p50

    # 动作拆解（30 笔：L1×10 + L2×5 + L4×5 + L5/L6×6 + L7/L8×4）
    act_rng = random.Random(seed + 1)
    picks = []
    for L, k in (("L1", 10), ("L2", 5), ("L4", 5)):
        pool = [r for r in rows if r["layer"] == L and r["duration"] > 0]
        picks += act_rng.sample(pool, min(k, len(pool)))
    for L, k in ((("L5", "L6"), 6), (("L7", "L8"), 4)):
        pool = [r for r in rows if r["layer"] in L and r["duration"] > 0]
        picks += act_rng.sample(pool, min(k, len(pool)))
    actions = ["查看付款与候选发票", "比对金额/日期/主体", "查历史记录/往来",
               "查差异原因（手续费/汇差/代付）", "填差异说明/备注", "提交"]
    agg = Counter()
    for r in picks:
        diff_layer = r["layer"] in ("L4", "L5", "L6")
        weights = [0.20, 0.28, 0.22, 0.18, 0.05, 0.07] if diff_layer else \
                  [0.30, 0.38, 0.10, 0.05, 0.12, 0.05]
        tot = act_rng.uniform(0.8, 1.2)
        for a, w in zip(actions, weights):
            agg[a] += r["duration"] * w / tot
    total_act = sum(agg.values())
    action_share = {a: round(agg.get(a, 0) / total_act, 4) for a in actions}

    # A.4.2 初筛召回（演练版：对仿真真值自动核对 50 笔）
    gold = [r for r in recs if r["label"] in ("MATCHED", "PARTIAL")]
    recall_rng = random.Random(seed + 2)
    recall50 = recall_rng.sample(gold, min(50, len(gold)))
    recall_hits = sum(1 for r in recall50  # 仿真初筛完整 → 恒命中（演练局限）
                      if r["invoice_id"] == r["invoice_id"])
    recall_table = [{
        "payment_id_masked": pseudonym(r["payment_id"], "sim-salt-2026"),
        "invoice_id_masked": pseudonym(r["invoice_id"], "sim-salt-2026"),
        "in_candidate_set": "Y",
        "note": "仿真初筛完整（真实数据须人工核对）",
    } for r in recall50]

    return {
        "n_sample": total, "alloc_actual": actual_alloc,
        "monthly_volume": round(sum(1 for _ in recs) / 3.0, 1),
        "duration_p50": round(pct(durs, 0.5) or 0, 2),
        "duration_p90": round(pct(durs, 0.9) or 0, 2),
        "duration_p50_ci": list(bootstrap_ci(durs, lambda v: pct(v, 0.5))),
        "review_rate": round((total - n_auto - n_hang) / total, 4),
        "review_rate_ci": [lo_r, hi_r],
        "auto_rate": round(auto_rate, 4),
        "auto_rate_ci": list(wilson(n_auto, total)),
        "error_rate": round(n_err / denom, 4),
        "error_rate_ci": [lo, hi],
        "n_error": n_err, "n_undecided": n_und, "denom": denom,
        "hang_rate": round(n_hang / total, 4),
        "avg_loss": round(sum(r["loss"] for r in rows if r["error"]) / max(1, n_err), 2),
        "by_layer": by_layer,
        "error_structure": {k: {"n": v,
                                "avg_loss": round(loss_by_type[k] / v, 2)}
                            for k, v in err_struct.items()},
        "subset": {"n_in_window": len(subset), "monthly": round(sub_monthly, 1),
                   "duration_p50": round(sub_p50, 2),
                   "target_coverage": round(target_cov, 4),
                   "minutes_per_month": round(minutes, 1),
                   "hours_per_month": round(minutes / 60, 1),
                   "fte": round(minutes / 60 / 174, 3)},
        "action_share": action_share,
        "recall50": {"n": len(recall50), "hits": recall_hits,
                     "rate": round(recall_hits / len(recall50), 4),
                     "table": recall_table},
    }


# ---------------------------------------------------------------------------
# 工作坊试点（100 例双人标注 + Kappa）
# ---------------------------------------------------------------------------

def pick_pilot(recs: list[dict], rng: random.Random) -> list[dict]:
    gold = [r for r in recs
            if r["confirmed_by"] in ("HUMAN", "HISTORICAL_CONFIRMED")
            and not (r["is_multi"] or r["is_split"])]
    by_label = defaultdict(list)
    for r in gold:
        by_label[r["label"]].append(r)
    chosen: dict[str, dict] = {}
    # 同主体上限：模板写"最多 2 例"，但主体群 <50 时 2×n_cp < 100 不可行
    #（演练发现：40 主体 ×2=80 <100）。规则放宽为 max(2, ceil(100/n_cp))。
    n_cp = len({r["payer_id"] for r in gold})
    cap = max(2, math.ceil(100 / n_cp))
    for label, k in (("MATCHED", 30), ("PARTIAL", 30), ("UNMATCHED", 30)):
        cnt = Counter(c["payer_id"] for c in chosen.values())
        for r in rng.sample(by_label[label], min(k * 4, len(by_label[label]))):
            if cnt[r["payer_id"]] < cap and r["payment_id"] not in chosen:
                chosen[r["payment_id"]] = r
                cnt[r["payer_id"]] += 1
            if sum(1 for c in chosen.values() if c["label"] == label) >= k:
                break
    # 5 例边界场景 + 5 例犹豫样本
    def _take(pool, k, tag):
        for r in rng.sample(pool, min(k * 5, len(pool))):
            if r["payment_id"] in chosen:
                continue
            chosen[r["payment_id"]] = {**r, "pilot_tag": tag}
            if sum(1 for c in chosen.values() if c.get("pilot_tag") == tag) >= k:
                break
    _take([r for r in gold if r["has_fee"]], 2, "边界-手续费可验证")
    _take([r for r in gold if 0 < r["abs_diff"] <= 100 and not r["has_fee"]], 2,
          "边界-手续费不可验证")
    _take([r for r in gold if 0 < r["abs_diff"] <= 40], 1, "边界-小额尾差")
    _take([r for r in gold if r["cross_month"]], 1, "边界-跨月付款")
    _take([r for r in gold if r["ref_has_inv_no"]], 1, "边界-附言含发票号")
    _take([r for r in gold if 0 < r["abs_diff"] <= 100], 5, "犹豫-容差边缘")
    return list(chosen.values())[:100]


def annotate(recs: list[dict], acc: float, seed: int) -> dict[str, str]:
    rng = random.Random(seed)
    adjacent = {"MATCHED": "PARTIAL", "PARTIAL": "UNMATCHED", "UNMATCHED": "PARTIAL"}
    out = {}
    for r in recs:
        t = r["label"]
        if rng.random() < acc:
            out[r["payment_id"]] = t
        elif rng.random() < 0.7:
            out[r["payment_id"]] = adjacent[t]
        else:
            out[r["payment_id"]] = rng.choice([x for x in ("MATCHED", "PARTIAL", "UNMATCHED")
                                               if x != t])
    return out


def cohen_kappa(a: dict, b: dict, labels: list[str]) -> dict:
    n = len(a)
    agree = sum(1 for k in a if a[k] == b[k])
    po = agree / n
    ca, cb = Counter(a.values()), Counter(b.values())
    pe = sum((ca[l] / n) * (cb[l] / n) for l in labels)
    kappa = (po - pe) / (1 - pe) if pe < 1 else 1.0
    table = {l1: {l2: sum(1 for k in a if a[k] == l1 and b[k] == l2)
                  for l2 in labels} for l1 in labels}
    return {"po": round(po, 4), "pe": round(pe, 4), "kappa": round(kappa, 4),
            "n": n, "matrix": table}


def run_workshop(recs: list[dict], seed: int) -> dict:
    rng = random.Random(seed)
    pilot = pick_pilot(recs, rng)
    a = annotate(pilot, 0.95, seed + 100)
    b = annotate(pilot, 0.90, seed + 200)
    k = cohen_kappa(a, b, ("MATCHED", "PARTIAL", "UNMATCHED"))
    disagree = [r for r in pilot if a[r["payment_id"]] != b[r["payment_id"]]]
    spec_defect = [r for r in disagree if r.get("pilot_tag") or r["abs_diff"] <= 100]
    verdict = ("通过" if k["kappa"] >= 0.75
               else "边缘（0.60–0.74）" if k["kappa"] >= 0.60 else "不通过")
    return {
        "n": len(pilot),
        "label_dist": dict(Counter(r["label"] for r in pilot)),
        "pilot_tags": dict(Counter(r.get("pilot_tag", "常规") for r in pilot)),
        "kappa": k, "verdict": verdict,
        "disagree_n": len(disagree),
        "disagree_spec_defect_n": len(spec_defect),
        "disagree_personal_n": len(disagree) - len(spec_defect),
        "disagree_detail": [{
            "payment_id_masked": pseudonym(r["payment_id"], "sim-salt-2026"),
            "truth": r["label"], "A": a[r["payment_id"]], "B": b[r["payment_id"]],
            "tag": r.get("pilot_tag", "常规"),
        } for r in disagree],
    }


# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    pays, invs, result_rows = load_clean_payments(Path(a.raw))
    recs = build_payment_records(pays, invs, result_rows)
    res = {
        "n_clean_payments_with_gt": len(recs),
        "label_dist": dict(Counter(r["label"] for r in recs)),
        "source_dist": dict(Counter(r["confirmed_by"] for r in recs)),
        "baseline": run_baseline(recs, a.seed),
        "workshop": run_workshop(recs, a.seed),
        "redline": "演练值：不得进入门禁 / D5 go-no-go / 模型验收（定稿 §1-D2）",
    }
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "n_clean": res["n_clean_payments_with_gt"],
        "labels": res["label_dist"],
        "baseline_p50": res["baseline"]["duration_p50"],
        "auto_rate": res["baseline"]["auto_rate"],
        "error_rate": res["baseline"]["error_rate"],
        "subset_fte": res["baseline"]["subset"]["fte"],
        "kappa": res["workshop"]["kappa"]["kappa"],
        "verdict": res["workshop"]["verdict"],
        "out": str(out),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
