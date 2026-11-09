"""FastAPI 薄壳（W3 交付物，定稿 §2.2.7 API 契约）。

端点（定稿路径 + 兼容旧路径别名）：
- POST /v1/decisions（= /decide）  统一决策入口：(付款,候选发票) → AUTO/REVIEW
  + 三态校准概率 + 命中规则；body 含 request_id（幂等键）/ tenant_id
- GET  /v1/decisions/{request_id}（= /decisions/{request_id}）  历史决策查询（审计）
- GET  /decisions                  决策审计列表（since_ts / limit）
- POST /v1/feedback/{request_id}（= /feedback/{request_id}）    人工复核回写
  （pending→approved/rejected；仅 approved 修正进下一版训练集，定稿 §3-8）
- GET  /v1/feedback（= /feedback） 复核记录查询（status 过滤）
- GET  /health        健康检查（模型/策略是否就绪；未就绪 → 503）
- GET  /status        详细状态
- POST /admin/mode    切 production / shadow（W4 shadow 上线开关）
- POST /admin/policy/rollback   策略回滚（§2.2.1 可回滚）
- GET  /admin/policies          策略版本列表
- POST /admin/model/load        热加载新模型工件（失败保持旧版）
- POST /admin/policy/load       热加载新策略（黄金样例校验，失败保持旧版）

错误码与降级（定稿 §2.2）：
- 409 DUPLICATE_REQUEST  request_id 7 天内重复 → 返回首次结果（幂等），不重算
- 503                    ServiceUnavailableError → 全停（上游据此交人工，不静默）

启动（uvicorn 由部署层拉起；此处提供 build_app 工厂供测试与生产复用）：
    uvicorn app:app --host 0.0.0.0 --port 8090
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "training" / "pipelines"))

from decision_service import DecisionService, ServiceUnavailableError  # noqa: E402
from inference import ArtifactLoadError  # noqa: E402
from storage import JsonlStore, Store  # noqa: E402
from strategy import PolicyCompileError  # noqa: E402


# ---------------------------------------------------------------------------
# 请求/响应模型（与 PairInput 字段对齐）
# ---------------------------------------------------------------------------

class DecideRequest(BaseModel):
    # —— 契约字段（定稿 §2.2.7）——
    request_id: Optional[str] = Field(
        default=None, pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$",
        description="幂等键：7 天窗口内重复请求返回首次结果（409）；缺省服务端生成、不去重")
    tenant_id: str = Field(default="default", max_length=64,
                           description="租户（一期单租户，字段贯穿落库）")
    # —— 任务输入（与 PairInput 对齐）——
    payment_id: str
    invoice_id: str
    invoice_number: str = ""        # 发票业务编号（人眼可读，如 INV-1005；reference 携带的是它）
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
        d = self.model_dump(exclude={"request_id", "tenant_id"})
        d["counterparty_match_rate_hist"] = (
            self.counterparty_match_rate_hist
            if self.counterparty_match_rate_hist is not None else 0.5)
        return d


class FeedbackIn(BaseModel):
    reviewer: str = Field(min_length=1, max_length=128)
    corrected_label: Optional[str] = Field(
        default=None, pattern="^(MATCHED|PARTIAL|UNMATCHED)$",
        description="人工纠正后的三态标签（维持原预测则不填）")
    review_action: Optional[str] = Field(default=None, max_length=64,
                                         description="复核动作（如 MAINTAIN / 纠正说明）")
    reason: Optional[str] = Field(default=None, max_length=512,
                                  description="纠错原因")
    status: str = Field(default="pending", pattern="^(pending|approved|rejected)$",
                        description="pending 待审；仅 approved 修正可进下一版训练集")


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

    @app.exception_handler(RequestValidationError)
    async def _400_schema(request, exc: RequestValidationError):
        # 契约 §2.2.7 / README / Shadow 契约 §4.2 均约定 schema 校验失败 = 400 SCHEMA_INVALID
        # （覆盖 FastAPI 默认的 422，避免 worker 按 400 路由异常池时漏判）
        return JSONResponse(status_code=400,
                            content={"error_code": "SCHEMA_INVALID",
                                     "errors": jsonable_encoder(exc.errors())})

    @app.post("/decide")
    @app.post("/v1/decisions")
    def decide(req: DecideRequest):
        result = service.decide(req.to_pair_kwargs(),
                                request_id=req.request_id,
                                tenant_id=req.tenant_id)
        if result.get("duplicate"):
            # 409 DUPLICATE_REQUEST：返回首次结果（幂等），不重算（定稿 §2.2）
            first = {k: v for k, v in result.items() if k != "duplicate"}
            return JSONResponse(status_code=409,
                                content={"error_code": "DUPLICATE_REQUEST",
                                         **first})
        return result

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

    @app.get("/decisions/{request_id}")
    @app.get("/v1/decisions/{request_id}")
    def get_decision(request_id: str) -> dict:
        result = service.audit(request_id)
        if result is None:
            raise HTTPException(404, f"request_id {request_id} 无对应决策")
        return result

    @app.post("/feedback/{request_id}")
    @app.post("/v1/feedback/{request_id}")
    def submit_feedback(request_id: str, body: FeedbackIn) -> dict:
        try:
            return service.submit_feedback(
                request_id, body.reviewer,
                corrected_label=body.corrected_label,
                review_action=body.review_action, reason=body.reason,
                status=body.status)
        except KeyError:
            raise HTTPException(404, f"request_id {request_id} 无对应决策")
        except ValueError as e:
            raise HTTPException(400, str(e))

    @app.get("/feedback")
    @app.get("/v1/feedback")
    def list_feedback(status: Optional[str] = None, limit: int = 1000) -> dict:
        rows = service.feedback(status=status, limit=limit)
        return {"count": len(rows), "feedback": rows}

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

    存储后端：
    - 设置 AIVORA_DB_URL → PostgresStore（生产，定稿 §2.1；DDL = infra/migrations）
    - 未设置 → JsonlStore（MVP/本地）
    加载失败**不抛**——让 service 保持 not-ready，/decide 返回 503 全停。
    这符合 §7-E 安全契约：未就绪 = 全停交人工，绝不静默降级。
    加载异常打到 stderr，便于排障。
    """
    import os
    import sys
    import time
    base = Path(os.environ.get("AIVORA_BASE",
                               str(Path(__file__).resolve().parent.parent)))
    artifact = os.environ.get("AIVORA_ARTIFACT_DIR", str(base / "artifacts" / "recon-v1"))
    policy = os.environ.get("AIVORA_POLICY_YAML", str(base / "policies" / "recon" / "engine.yaml"))
    golden = os.environ.get("AIVORA_GOLDEN_JSON",
                            str(base / "contracts" / "examples" / "recon.golden.json"))
    logs = os.environ.get("AIVORA_DECISION_LOGS", str(base / "logs" / "decision_logs.jsonl"))
    mode = os.environ.get("AIVORA_MODE", "shadow")  # 默认 shadow（最安全）
    db_url = os.environ.get("AIVORA_DB_URL")

    if db_url:
        from postgres_store import PostgresStore
        store: Store = PostgresStore(db_url)
        # compose 已用 healthcheck 保证 Postgres 就绪；这里再做有限重试兜底
        for attempt in range(30):
            try:
                store.open(timeout_s=5)
                break
            except Exception as e:  # noqa: BLE001
                if attempt == 29:
                    print(f"[decision-service] Postgres 连接失败（保持 not-ready → 503 全停）: "
                          f"{type(e).__name__}: {e}", file=sys.stderr)
                time.sleep(1)
    else:
        store = JsonlStore(logs)

    svc = DecisionService(store, mode=mode)
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
