# ============================================================================
# Aivora Decision Service（定稿 §2.1/§2.5：Python 3.11 + FastAPI + LightGBM CPU）
# 构建上下文 = aivora-decision/（compose 中 context: ..）
#
#   docker build -f infra/Dockerfile -t aivora-decision-service:dev aivora-decision/
#
# 运行（本地冒烟，JsonlStore 后端）：
#   docker run --rm -p 8090:8090 -v <artifact-dir>:/app/artifacts/recon-v1 aivora-decision-service:dev
# 运行（Postgres 后端）：见 infra/docker-compose.yml（AIVORA_DB_URL）
#
# 注意：
# - 服务 import training/pipelines（特征清单）与 training/policies（策略引擎），
#   三者同仓库同版本发布（定稿 §2.2.3 特征版本化 / §7-H 策略与代码同评审）；
# - --workers 1：策略/模型注册表与幂等索引在进程内存（Postgres 持久化决策日志，
#   但热加载指针按单进程设计）；扩容前先拆 gateway（定稿 P4）。
# ============================================================================
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY service/requirements-service.txt ./requirements-service.txt
RUN pip install --no-cache-dir -r requirements-service.txt

COPY service/    ./service/
COPY training/   ./training/
COPY policies/   ./policies/
COPY contracts/  ./contracts/

# 非 root 运行；工件/日志目录预建（可被 volume 覆盖）
RUN useradd -m appuser \
    && mkdir -p /app/logs /app/artifacts/recon-v1 \
    && chown -R appuser:appuser /app
USER appuser

# service/ 模块互相以顶层模块导入（decision_service/inference/storage/...），
# WORKDIR 须为 service/（与本地 `uvicorn app:app` 同一启动方式；app.py 亦自带
# sys.path 兜底）
WORKDIR /app/service

EXPOSE 8090

# 存活探针：/health 恒 200（ready/not_ready 状态在 body；not_ready 时 /decide 503 全停）
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8090/health',timeout=3)" || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8090", "--workers", "1"]
