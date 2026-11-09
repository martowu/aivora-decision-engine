#!/usr/bin/env python3
"""W1 演练：在 DuckDB 上执行《W1 数据盘点报告》附录 A 的 SQL 原文。

从报告 markdown 中提取第一个 ```sql fence，替换 <start>/<end>/<tenant> 占位符，
对仿真原始表（read_csv_auto 建表）逐句执行，结果 JSON 输出。

用途（演练红线，定稿 §1-D2）：
- 验证附录 A SQL 骨架**逐句可执行**（双方言兼容写法）；
- 产出盘点报告的演练数字（全部标注"仿真值，不得进门禁"）。

用法：
  python sim_inventory_duckdb.py --report ../../../W1_数据盘点报告_模板.md \
      --raw ../out/sim-w1/raw --out ../out/sim-w1/inventory.json \
      --start 2026-07-01 --end 2026-09-30 --tenant t-sim
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import duckdb


def extract_sql_fence(md_path: Path) -> str:
    text = md_path.read_text(encoding="utf-8")
    m = re.search(r"```sql\n(.*?)```", text, flags=re.S)
    if not m:
        raise SystemExit(f"{md_path} 未找到 sql fence")
    return m.group(1)


def split_statements(sql: str) -> list[str]:
    """按"代码部分以分号结尾"切分（支持 `;` 后跟行尾注释；本骨架字符串内无 `--`）。"""
    out, buf = [], []
    for line in sql.splitlines():
        buf.append(line)
        code = line.split("--", 1)[0].rstrip()
        if code.endswith(";"):
            stmt = "\n".join(buf).strip()
            if any(l.split("--", 1)[0].strip() for l in stmt.splitlines()):
                out.append(stmt)
            buf = []
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True)
    ap.add_argument("--raw", required=True, help="仿真原始表目录（payments/invoices/recon_results）")
    ap.add_argument("--out", required=True)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--tenant", required=True)
    ap.add_argument("--doc-type", default="AP", help="AP 或 AR（D5 决策 4；Recon 生产为 AR）")
    a = ap.parse_args()

    sql = extract_sql_fence(Path(a.report))
    sql = (sql.replace("<start>", a.start)
           .replace("<end>", a.end)
           .replace("<tenant>", a.tenant)
           .replace("<doc_type>", a.doc_type))
    raw = Path(a.raw)
    for t in ("payments", "invoices", "recon_results"):
        if not (raw / f"{t}.csv").exists():
            raise SystemExit(f"缺 {raw / t}.csv")

    con = duckdb.connect()
    con.execute(f"CREATE TABLE payments AS SELECT * FROM read_csv_auto('{raw / 'payments.csv'}', all_varchar=true)")
    con.execute(f"CREATE TABLE invoices AS SELECT * FROM read_csv_auto('{raw / 'invoices.csv'}', all_varchar=true)")
    con.execute(f"CREATE TABLE recon_results AS SELECT * FROM read_csv_auto('{raw / 'recon_results.csv'}', all_varchar=true)")

    results, errors = {}, []
    for i, stmt in enumerate(split_statements(sql), 1):
        name = re.search(r"A\.(\d+[a-z]?)", stmt)
        label = name.group(0) if name else f"stmt{i}"
        try:
            rel = con.execute(stmt)
            cols = [d[0] for d in rel.description]
            rows = rel.fetchall()
            # 本骨架全部为单行聚合查询；若将来改为多行，此处会如实报 rows>1
            results[label] = {"n_rows": len(rows),
                              "row": dict(zip(cols, (_js(v) for v in rows[0]))) if rows else None}
        except Exception as e:  # noqa: BLE001
            errors.append({"label": label, "error": f"{type(e).__name__}: {e}",
                           "stmt": stmt[:200]})
            print(f"  [ERR] {label}: {e}", file=sys.stderr)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"duckdb": duckdb.__version__, "window": [a.start, a.end],
                               "tenant": a.tenant, "results": results, "errors": errors},
                              indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"ok": not errors, "n_results": len(results), "errors": errors,
                      "out": str(out)}, indent=2, ensure_ascii=False))
    sys.exit(1 if errors else 0)


def _js(v):
    if v is None:
        return None
    if isinstance(v, (int, str, bool)):
        return v
    if isinstance(v, float):
        return round(v, 6)
    return str(v)


if __name__ == "__main__":
    main()
