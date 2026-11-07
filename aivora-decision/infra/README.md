# infra/ —— 一期单机部署（定稿 §2.1 / §2.5 / v1.0 §2.6）

Docker Compose 单机部署（**不上 K8s**，预留 Helm 路径）：一个 decision-service
+ Postgres 15 + MinIO（预留）；W4 加 Prometheus + Grafana（`--profile w4`）。

## 文件

| 文件 | 作用 |
|---|---|
| `Dockerfile` | decision-service 镜像（python:3.11-slim，非 root，HEALTHCHECK） |
| `docker-compose.yml` | 一期编排（默认 3 服务；`--profile w4` 加 2 监控服务） |
| `env.example` | 环境变量样例（cp 为 .env；.env 已 gitignore） |
| `migrations/0001_init.sql` | 五表 DDL（tasks/models/policies/logs/feedback）+ recon 任务注册 |
| `monitoring/prometheus.yml` | W4 抓取配置（/metrics 端点落地前目标保持注释） |

**依赖版本注**：`lightgbm` 固定 `>=4.0,<4.6`（4.7.0 的 py3.10 wheel 在部分
Python 构建下原生崩溃；4.5.x 已全量验证）。`psycopg[binary,pool]` 为 Postgres
后端依赖（JsonlStore MVP 路径不需要，但镜像统一安装）。

## 快速开始

```bash
cd aivora-decision/infra
cp env.example .env          # 开发默认值；生产替换为强随机值或 secrets 注入
docker compose up -d --build

# 健康检查（未放工件时 = not_ready，/decide 503 全停——安全契约）
curl -s localhost:8090/health

# 放入训练工件（train.py 输出目录）后热加载（失败保持旧版，不中断）
docker compose cp <host>/artifacts/recon-v1/. decision-service:/app/artifacts/recon-v1/
curl -s -X POST localhost:8090/admin/model/load \
     -H 'content-type: application/json' \
     -d '{"artifact_dir": "/app/artifacts/recon-v1"}'
curl -s localhost:8090/health        # → ready
```

或直接挂载工件卷（训练机 → 服务机）：

```bash
docker run --rm -p 8090:8090 \
  -v /path/to/artifacts/recon-v1:/app/artifacts/recon-v1 \
  -e AIVORA_DB_URL="postgresql://aivora:<db-password>@<host>:15432/aivora_decision" \
  aivora-decision-service:dev
```

> `.env` 变量缺失时 `docker compose config/up` 会报
> `required variable ... is missing a value: set in infra/.env`（`:?` 必填校验）。
> 注：`AIVORA_DB_URL` 的主机名 `postgres` 仅在 compose 网络内有效。

## 存储后端

| 环境变量 | 行为 |
|---|---|
| `AIVORA_DB_URL`（compose 已设） | **PostgresStore**（生产路径）：decision_logs/feedback/策略与模型注册表落五表；`request_id` 主键幂等 |
| 未设置 | **JsonlStore**（MVP/本地）：JSONL 审计日志 + 内存注册表，重启自动重建索引 |

两者同接口（`service/storage.py::Store`），决策服务逻辑不变。
DDL 变更流程：新增 `migrations/000N_*.sql` 编号文件（initdb 目录按文件名序执行），
改表结构须同步 `postgres_store.py` 映射与 `test_postgres_store.py`。

## 安全契约（镜像行为）

- 模型/策略任一未就绪 → `/decide` 503 `{"action":"HUMAN"}`（全停交人工，不静默降级）；
- `AIVORA_MODE` 默认 **shadow**：AUTO 对外掩码 REVIEW、落库记真实决策（W4 门禁后切 production）；
- 紧急叫停：`POST /admin/mode {"mode":"shadow"}`（分钟级，§7-E runbook 场景 2/3）。

## 端口

| 服务 | 端口 | 用途 |
|---|---|---|
| decision-service | 8090 | 决策 API（Recon 内网调用） |
| postgres | 5432（默认不对外映射） | 五表 |
| minio | 9000 / 9001 | S3 API / 控制台（工件预留） |
| prometheus（w4） | 9090 | 抓取（/metrics 就绪后启用） |
| grafana（w4） | 3000 | 面板 |
