"""FastAPI 薄壳（W3 交付物）。

端点：
- POST /decide        决策（(付款,候选发票) → AUTO/REVIEW + 三态概率 + 命中规则）
- GET  /health        健康检查（模型/策略是否就绪；未就绪 → 503）
- GET  /status        详细状态
- GET  /decisions     决策审计（since_ts / limit）
- POST /admin/mode    切 production / shadow（W4 shadow 上线开关）
- POST /admin/policy/rollback   策略回滚（§2.2.1 可回滚）
- GET  /admin/policies          策略版本列表
- POST /admin/model/load        热加载新模型工件（失败保持旧版）
- POST /admin/policy/load       热加载新策略（黄金样例校验，失败保持旧版）

安全：任何 5xx 来自 ServiceUnavailableError → 503 全停（上游据此交人工，不静默）。

启动（uvicorn 由部署层拉起；此处提供 build_app 工厂供测试与生产复用）：
    uvicorn app:app --host 0.0.0.0 --port 8090
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "training" / "pipelines"))

from decision_service import DecisionService, ServiceUnavailableError  # noqa: E402
from inference import ArtifactLoadError  # noqa: E402
from storage import JsonlStore  # noqa: E402
from strategy import PolicyCompileError  # noqa: E402


# ---------------------------------------------------------------------------
# 请求/响应模型（与 PairInput 字段对齐）
# ---------------------------------------------------------------------------

class DecideRequest(BaseModel):
    payment_id: str
    invoice_id: str
    pay_amount: float = Field(ge=0)
    inv_amount: float = Field(ge=0)
    pay_currency: str = "CNY"
    inv_currency: str = "CNY"
    pay_date: str
    inv_date: str
    payer_id: str
    supplier_id: str
    supplier_name: str = ""
    channel: str = "BANK_TRANSFER"
    reference: str = ""
    note: str = ""
    fee: float = 0.0
    counterparty_match_rate_hist: Optional[float] = None
    is_first_counterparty: bool = False
    is_multi_mapping: bool = False
    has_split_merge: bool = False

    def to_pair_kwargs(self) -> dict:
        # counterparty_match_rate_hist 缺省给 0.5（无历史）—— 与特征侧 None→0.5 对齐
        d = self.model_dump()
        d["counterparty_match_rate_hist"] = (
            self.counterparty_match_rate_hist
            if self.counterparty_match_rate_hist is not None else 0.5)
        return d


class AdminModeIn(BaseModel):
    mode: str = Field(pattern="^(production|shadow)$")


class AdminRollbackIn(BaseModel):
    target_version: str


class AdminModelLoadIn(BaseModel):
    artifact_dir: str


class AdminPolicyLoadIn(BaseModel):
    yaml_path: str
    golden_path: Optional[str] = None


# ---------------------------------------------------------------------------
# app 工厂
# ---------------------------------------------------------------------------

def build_app(service: DecisionService) -> FastAPI:
    app = FastAPI(title="Aivora Decision Service", version="1.0",
                  description="recon.transaction_match 自动决策（AUTO/REVIEW）")

    @app.exception_handler(ServiceUnavailableError)
    async def _503(request, exc: ServiceUnavailableError):
        return JSONResponse(status_code=503, content={"error": str(exc), "action": "HUMAN"})

    @app.post("/decide")
    def decide(req: DecideRequest) -> dict:
        return service.decide(req.to_pair_kwargs())

    @app.get("/health")
    def health() -> dict:
        st = service.status()
        ready = st["model_loaded"] and st["policy_loaded"]
        return {"status": "ready" if ready else "not_ready", **st}

    @app.get("/status")
    def status() -> dict:
        return service.status()

    @app.get("/decisions")
    def decisions(since_ts: float = 0.0, limit: int = 1000) -> dict:
        rows = [log.to_row() for log in service._store.decisions(since_ts, limit)]
        return {"count": len(rows), "decisions": rows}

    @app.post("/admin/mode")
    def set_mode(body: AdminModeIn) -> dict:
        try:
            v = service.set_mode(body.mode)
        except ValueError as e:
            raise HTTPException(400, str(e))
        return {"mode": v}

    @app.post("/admin/policy/rollback")
    def rollback_policy(body: AdminRollbackIn) -> dict:
        try:
            v = service.rollback_policy(body.target_version)
        except ValueError as e:
            raise HTTPException(404, str(e))
        return {"policy_version": v}

    @app.get("/admin/policies")
    def list_policies() -> dict:
        return {"policy_versions": service._store.policy_versions(),
                "current": service.status()["policy_version"]}

    @app.post("/admin/model/load")
    def load_model(body: AdminModelLoadIn) -> dict:
        try:
            v = service.load_model(body.artifact_dir)
        except ArtifactLoadError as e:
            raise HTTPException(503, f"模型加载失败，保持旧版: {e}")
        return {"model_version": v}

    @app.post("/admin/policy/load")
    def load_policy(body: AdminPolicyLoadIn) -> dict:
        golden = None
        if body.golden_path and Path(body.golden_path).exists():
            golden = json.loads(Path(body.golden_path).read_text(encoding="utf-8"))
        try:
            v = service.load_policy(body.yaml_path, golden)
        except (PolicyCompileError, ArtifactLoadError, ValueError) as e:
            raise HTTPException(400, f"策略加载失败，保持旧版: {e}")
        return {"policy_version": v}

    return app


# ---------------------------------------------------------------------------
# 默认实例（生产入口）：从环境变量读路径
# ---------------------------------------------------------------------------

def _default_service() -> DecisionService:
    """构建生产 service：默认 shadow 模式，启动即加载工件 + 策略。

    加载失败**不抛**——让 service 保持 not-ready，/decide 返回 503 全停。
    这符合 §7-E 安全契约：未就绪 = 全停交人工，绝不静默降级。
    加载异常打到 stderr，便于排障。
    """
    import os
    import sys
    base = Path(os.environ.get("AIVORA_BASE",
                               str(Path(__file__).resolve().parent.parent)))
    artifact = os.environ.get("AIVORA_ARTIFACT_DIR", str(base / "artifacts" / "recon-v1"))
    policy = os.environ.get("AIVORA_POLICY_YAML", str(base / "policies" / "recon" / "engine.yaml"))
    golden = os.environ.get("AIVORA_GOLDEN_JSON",
                            str(base / "contracts" / "examples" / "recon.golden.json"))
    logs = os.environ.get("AIVORA_DECISION_LOGS", str(base / "logs" / "decision_logs.jsonl"))
    mode = os.environ.get("AIVORA_MODE", "shadow")  # 默认 shadow（最安全）

    svc = DecisionService(JsonlStore(logs), mode=mode)
    for step in (lambda: svc.load_model(artifact),
                 lambda: svc.load_policy(policy,
                                         _read_json_or_none(golden))):
        try:
            step()
        except Exception as e:  # noqa: BLE001
            print(f"[decision-service] 启动加载失败（保持 not-ready → 503 全停）: "
                  f"{type(e).__name__}: {e}", file=sys.stderr)
            break
    return svc


def _read_json_or_none(path: str) -> Optional[list]:
    p = Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


# 模块级单例：uvicorn app:app 直接可用。import 时构建（加载失败 → not-ready 503）。
# 测试请用 build_app(自己的 service)，不要 import 这个 `app`（避免起真实工件）。
app: FastAPI = build_app(_default_service())
