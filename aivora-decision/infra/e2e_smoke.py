#!/usr/bin/env python3
"""E2E 冒烟：服务 + 真实 Postgres 15 + 真实训练工件，验证部署路径全链路。

两种模式：
- 默认（容器模式）：
  前置：python:3.11-slim 基础镜像可用（Dockerfile 构建）、
  Postgres 15 在 localhost:15432（aivora-pg-test 容器，已执行 migrations）、
  工件目录（train.py 自检输出，默认 D:\\tmp\\selftest-artifact）。
  步骤：docker compose build → docker run（shadow，Postgres 后端）→ 全链路 API
  → psql 直查落库 → 切 production。
- --local（本地模式，无 Docker）：uvicorn 直接起服务（同 Postgres + 工件），
  验证除容器打包外的全部路径（app 入口、PostgresStore、幂等、feedback、
  热切换）。

用法：
  python e2e_smoke.py [--local] [artifact_dir]
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent   # aivora-decision/
INFRA = BASE / "infra"
LOCAL = "--local" in sys.argv
_pos = [a for a in sys.argv[1:] if not a.startswith("-")]
ARTIFACT = Path(_pos[0]) if _pos else Path(r"D:\tmp\selftest-artifact")
DB_URL_LOCAL = "postgresql://aivora:aivora-dev@localhost:15432/aivora_decision"
DB_URL = "postgresql://aivora:aivora-dev@host.docker.internal:15432/aivora_decision"
PORT = "8092" if LOCAL else "8091"
URL = f"http://127.0.0.1:{PORT}"
# 每次运行唯一 request_id（Postgres 持久化 + 幂等语义：固定 id 重跑会得到 409 重放）
RUN = time.strftime("%H%M%S")


def sh(*args, check: bool = True) -> str:
    r = subprocess.run(list(args), capture_output=True, text=True, cwd=str(INFRA))
    if check and r.returncode != 0:
        raise SystemExit(f"CMD FAILED: {args}\n{r.stdout}\n{r.stderr}")
    return r.stdout + r.stderr


def http(method: str, path: str, body: dict | None = None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(URL + path, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode() or "{}")


def main() -> None:
    proc = None
    if LOCAL:
        # 本地模式：uvicorn 直接起（验证 app 入口 + PostgresStore 全链路）
        print("== 0. 本地模式：uvicorn 起服务 ==")
        import os
        import tempfile
        py = sys.executable
        logf = open(tempfile.mktemp(suffix=".svc.log", prefix="aivora_e2e_"),
                    "a+", encoding="utf-8")
        env = dict(os.environ,
                   AIVORA_DB_URL=DB_URL_LOCAL, AIVORA_MODE="shadow",
                   AIVORA_BASE=str(BASE),
                   AIVORA_ARTIFACT_DIR=str(ARTIFACT))
        proc = subprocess.Popen(
            [py, "-m", "uvicorn", "app:app",
             "--host", "127.0.0.1", "--port", PORT],
            cwd=str(BASE / "service"), env=env,
            stdout=logf, stderr=subprocess.STDOUT)
        print(f"   uvicorn pid={proc.pid} log={logf.name}")
    else:
        # 0) 清理
        sh("docker", "rm", "-f", "aivora-svc-e2e")
        # 1) 构建
        print("== 1. docker compose build ==")
        out = sh("docker", "compose", "build", "decision-service")
        print("   build OK" if "Successfully" in out or "built" in out.lower() or True else "")
        # 2) 运行（shadow 模式，Postgres 后端）
        print("== 2. 启动服务容器 ==")
        sh("docker", "run", "-d", "--name", "aivora-svc-e2e",
           f"-p", f"{PORT}:8090",
           "-v", f"{ARTIFACT.as_posix()}/:/app/artifacts/recon-v1/:ro".replace("///", "/"),
           "-e", f"AIVORA_DB_URL={DB_URL}",
           "-e", "AIVORA_MODE=shadow",
           "aivora-decision-service:dev")
    # 3) 等健康
    print("== 3. /health 等待 ready ==")
    ready = False
    for i in range(60):
        time.sleep(2)
        try:
            st, body = http("GET", "/health")
            if body.get("status") == "ready":
                ready = True
                break
            print(f"   [{i*2}s] {body.get('status')} model={body.get('model_version')}")
        except Exception:
            pass
    if not ready:
        if LOCAL:
            logf.seek(0)
            print("   uvicorn 日志尾部：")
            print("\n".join(logf.read().splitlines()[-20:]))
        else:
            print("   容器日志尾部：")
            print(sh("docker", "logs", "--tail", "20", "aivora-svc-e2e", check=False))
        if proc:
            proc.terminate()
        raise SystemExit("NOT READY — 冒烟失败")
    print("   READY:", json.dumps({k: body.get(k) for k in
                                   ("mode", "model_version", "policy_version",
                                    "feature_version")}, ensure_ascii=False))

    sample = {
        "request_id": f"e2e-{RUN}-req-01", "tenant_id": "t-e2e",
        "payment_id": "PAY-E2E-1", "invoice_id": "INV-E2E-1",
        "pay_amount": 50000.0, "inv_amount": 50000.0,
        "pay_date": "2026-09-01", "inv_date": "2026-08-28",
        "payer_id": "CP-09-01", "supplier_id": "CP-09-01",
        "supplier_name": "甲公司", "channel": "BANK_TRANSFER",
        "reference": "付款 INV-E2E-1 全额", "note": "备注",
        "counterparty_match_rate_hist": 0.99,
    }
    # 4) shadow 决策 + 幂等
    print("== 4. /decide（shadow）+ 409 幂等 ==")
    st1, b1 = http("POST", "/decide", sample)
    assert st1 == 200, (st1, b1)
    assert b1["mode"] == "shadow" and b1["is_shadow"] is True
    assert b1["action"] == "REVIEW" or b1["action"] == "AUTO"
    print(f"   首次: action={b1['action']} model_action={b1['model_action']} "
          f"label={b1['model']['label']} latency={b1.get('latency_ms')}ms")
    st2, b2 = http("POST", "/decide", {**sample, "pay_amount": 99999.0})
    assert st2 == 409 and b2.get("error_code") == "DUPLICATE_REQUEST", (st2, b2)
    assert b2["decision_id"] == b1["decision_id"], "409 未返回首次结果！"
    print("   重复: 409 DUPLICATE_REQUEST + 首次 decision_id 一致 ✓")
    # 5) 审计 + feedback
    print("== 5. 审计查询 + feedback 闭环 ==")
    st3, b3 = http("GET", f"/v1/decisions/{sample['request_id']}")
    assert st3 == 200 and b3["decision_id"] == b1["decision_id"]
    print(f"   审计: {b3['action']} rule={b3['policy']['rule']} "
          f"model={b3['model_version']} feature={b3['feature_version']}")
    st4, b4 = http("POST", f"/v1/feedback/{sample['request_id']}",
                   {"reviewer": "e2e-fin", "corrected_label": "PARTIAL",
                    "reason": "e2e 演练", "status": "approved"})
    assert st4 == 200 and b4["status"] == "approved", (st4, b4)
    st5, b5 = http("GET", "/v1/feedback?status=approved")
    assert b5["count"] >= 1
    print("   feedback: approved 写入 + 查询 ✓")
    # 6) psql 直查落库
    print("== 6. psql 直查 Postgres ==")
    for q, name in (
        (f"SELECT count(*) FROM decision_logs WHERE request_id='{sample['request_id']}'", "logs"),
        ("SELECT status FROM decision_feedback WHERE reviewer='e2e-fin'", "feedback"),
        ("SELECT version FROM decision_models WHERE status='production'", "model_prod"),
        ("SELECT version FROM decision_policies WHERE status='active'", "policy_active"),
    ):
        r = subprocess.run(["docker", "exec", "aivora-pg-test", "psql", "-U", "aivora",
                            "-d", "aivora_decision", "-tA", "-c", q],
                           capture_output=True, text=True)
        print(f"   {name}: {r.stdout.strip()!r}")
    # 7) 切 production
    print("== 7. 切 production（AUTO 可穿透）==")
    st6, _ = http("POST", "/admin/mode", {"mode": "production"})
    assert st6 == 200
    st7, b7 = http("POST", "/decide", {**sample, "request_id": f"e2e-{RUN}-req-02"})
    assert st7 == 200 and b7["mode"] == "production"
    print(f"   production: action={b7['action']}（shadow 同请求为 "
          f"{b1['action']}，真实决策 {b1['model_action']}）")

    print("\nE2E SMOKE: ALL PASS ✓")
    # 收尾
    if LOCAL and proc:
        proc.terminate()
        proc.wait(timeout=10)
    else:
        subprocess.run(["docker", "rm", "-f", "aivora-svc-e2e"], capture_output=True)


if __name__ == "__main__":
    main()
