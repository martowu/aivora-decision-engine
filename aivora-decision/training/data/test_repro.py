"""管线可重跑性测试（定稿 §7-C 验收）：
同参数重跑 → 文件字节一致 + data_snapshot_hash 一致；参数变化 → hash 变化。
"""

import csv
import hashlib
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SALT = HERE / ".test-salt.txt"


def run(out: str, salt_content: str, start="2026-08-01", end="2026-10-31", n="50"):
    SALT.write_text(salt_content)
    return subprocess.run(
        [sys.executable, str(HERE / "extract.py"),
         "--start", start, "--end", end, "--tenant", "t-test",
         "--source", "synthetic", "--n", n,
         "--salt-file", str(SALT), "--output", out],
        capture_output=True, text=True)


def files_digest(out: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(out.iterdir()):
        if p.name == "snapshot.json":
            continue
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def main():
    import tempfile, json, shutil
    t = Path(tempfile.mkdtemp())
    try:
        o1, o2 = t / "out1", t / "out2"
        r1 = run(str(o1), "salt-A")
        r2 = run(str(o2), "salt-A")
        assert r1.returncode == 0, r1.stderr
        assert r2.returncode == 0, r2.stderr
        snap1 = json.loads((o1 / "snapshot.json").read_text(encoding="utf-8"))
        snap2 = json.loads((o2 / "snapshot.json").read_text(encoding="utf-8"))
        assert snap1["data_snapshot_hash"] == snap2["data_snapshot_hash"], "重跑 hash 不一致"
        assert files_digest(o1) == files_digest(o2), "重跑文件字节不一致"

        # 盐值变化 → 假名变化 → hash 变化
        o3 = t / "out3"
        r3 = run(str(o3), "salt-B")
        snap3 = json.loads((o3 / "snapshot.json").read_text(encoding="utf-8"))
        assert r3.returncode == 0, r3.stderr
        assert snap3["data_snapshot_hash"] != snap1["data_snapshot_hash"], "换盐后 hash 未变"

        # 时间窗变化 → hash 变化
        o4 = t / "out4"
        r4 = run(str(o4), "salt-A", end="2026-09-30")
        snap4 = json.loads((o4 / "snapshot.json").read_text(encoding="utf-8"))
        assert r4.returncode == 0, r4.stderr
        assert snap4["data_snapshot_hash"] != snap1["data_snapshot_hash"], "改时间窗后 hash 未变"

        # 脱敏抽查：输出中不得出现原始主体 ID 明文
        raw = ((o1 / "payments.csv").read_text(encoding="utf-8")
               + (o1 / "invoices.csv").read_text(encoding="utf-8"))
        for bad in ("CP-0", "SP-0"):
            assert bad not in raw, f"敏感字段未脱敏: {bad}"
        rows = list(csv.DictReader((o1 / "payments.csv").open(encoding="utf-8")))
        assert all(r["payer_id"].startswith("p_") for r in rows), "假名化格式错误"

        print("PASS: 可重跑性(同参 hash/字节一致) + 参数敏感性(盐/时间窗) + 脱敏抽查")
        print("snapshot_hash(sample):", snap1["data_snapshot_hash"][:16], "...")
    finally:
        shutil.rmtree(t, ignore_errors=True)
        SALT.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
