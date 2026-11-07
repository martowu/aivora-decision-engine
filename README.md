# Aivora Decision Engine

在业务系统旁挂一层**"模型无关、可验证、可拒绝"的决策服务**：把结构化业务状态转成
带校准置信度的类型化决策，**模型预测（prediction）与业务动作（decision）彻底分离**。

首任务：`recon.transaction_match@1.0`（Recon 对账核销的交易匹配三分类
MATCHED / PARTIAL / UNMATCHED + 策略路由 AUTO / REVIEW）。

> 执行基线：[`Aivora_Decision_Engine_最佳方案_定稿.md`](Aivora_Decision_Engine_最佳方案_定稿.md)。
> 三个门禁：W1 数据 → W2 离线 → W4 Shadow（拟 AUTO 推翻率 ≤2% 安全红线）。
> 红线：合成/仿真数据只验证 pipeline，**不得**进门禁、不得用于验收（定稿 §1-D2）。

---

## 一、仓库布局

```
aivora-decision-engine/
├── 【规划文档】（根目录，三层递进 + 会议/周交付物）
│   ├── Aivora_Decision_Engine_需求与技术方案.md        v0.1 讨论稿
│   ├── Aivora_Decision_Engine_需求分析与实施方案.md     v1.0 批判性分析 + 可执行方案（G1–G10 缺口补齐）
│   ├── Aivora_Decision_Engine_最佳方案_定稿.md         ★ 执行基线（冲突处以本文为准）
│   ├── D5决策会_会前材料.md                             立项 go/no-go 会议包（建议值已预填，附录 C 偏差清单）
│   ├── W1_数据盘点报告_模板.md                          模板（口径/SQL 骨架已预填）
│   ├── W1_数据盘点报告_模拟演练.md                      ★ 仿真数据演练版（红线：非门禁依据）
│   ├── W1_标签工作坊_引导材料.md                        模板（三态定义建议起点 + 20 边界案例已预填）
│   ├── W1_标签工作坊_模拟演练.md                        ★ 演练版（Kappa 0.85 实测，模拟裁决）
│   ├── W1_现状基线测量_模板.md                          模板（分层抽样/口径/收益公式已预填）
│   └── W1_现状基线测量_模拟演练.md                      ★ 演练版（200 笔分层 + CI 实测）
│
└── aivora-decision/                                     代码根（= 定稿 §2.6 仓库结构）
    ├── service/            【W3】决策服务（FastAPI，唯一新服务）
    │   ├── app.py            FastAPI 薄壳：/decide(/v1/decisions)、审计查询、
    │   │                     feedback、admin（模式切换/热加载/回滚）；503 全停映射
    │   ├── decision_service.py  决策核心：模型+策略+存储装配；幂等（409 重放）、
    │   │                     shadow 掩码、input_digest、latency_ms；安全契约（不静默降级）
    │   ├── inference.py      推理端：加载工件（model.txt+calibration.json）还原校准概率
    │   │                     （Platt/isotonic 纯加载，部署无状态）
    │   ├── storage.py        Store 接口 + JsonlStore（MVP：内存索引+JSONL，重启可恢复）
    │   ├── postgres_store.py PostgresStore（生产后端，同接口直接替换；五表映射）
    │   ├── shadow.py         Shadow 掩码 + ShadowReporter（W4 门禁：推翻率 ≤2%）
    │   ├── test_service.py   服务单测 11 例（503/落库/shadow/回滚/幂等/审计/feedback/重启）
    │   ├── test_postgres_store.py  Postgres 集成测试 8 例（需 AIVORA_TEST_DB_URL）
    │   └── requirements-service.txt
    │
    ├── training/
    │   ├── pipelines/        【W2】训练/校准/评估管线
    │   │   ├── features.py     特征 v1.0：26 个纯函数特征（point-in-time 安全、
    │   │   │                   事后信息黑名单 FORBIDDEN_INPUTS、结构 flag 透传）
    │   │   ├── train.py        切分（时间+主体 group）+ 规则基线 + LightGBM 三分类
    │   │   │                   + OvR 校准（≥2000 isotonic / 否则 Platt）+ 全量评估
    │   │   │                   + 门禁判定 + 工件导出（model.txt 为规范工件）
    │   │   ├── synthetic.py    合成数据生成器（主体人群"干净度"模型；仅验证流程）
    │   │   ├── demo_e2e.py     端到端演示：生成→特征→训练→校准→策略路由
    │   │   ├── verify_deploy.py 部署闭环验证（工件维度/类别数/确定性/概率合法）
    │   │   └── test_features.py 特征单测 11 例
    │   ├── policies/         【W3】策略引擎（与代码同仓库同评审，§7-H）
    │   │   ├── strategy.py     受限 AST 求值器（§7-A：禁裸 eval，白名单 f.*/calib.*/
    │   │   │                   payment.*）；first-match-wins；默认落 REVIEW；
    │   │   │                   求值失败保守 REVIEW（绝不静默 AUTO）；黄金样例干跑
    │   │   └── test_policy.py  策略单测 13 例（注入防御/硬规则优先/回滚/快照往返）
    │   └── data/             【W1】数据抽取/脱敏/质检管线（§7-C 可重跑）
    │       ├── pipe.py         核心：RunConfig（参数化）、假名化（加盐 sha256）、
    │       │                   文本截断、SENSITIVE_FIELDS、质检、data_snapshot_hash
    │       ├── extract.py      CLI：抽取→清洗→脱敏→质检→CSV/parquet+snapshot.json
    │       ├── csv_source.py   CsvReconSource(DataSource)：生产接入点（真实 Recon
    │       │                   导出同 schema 即零改动接入；清洗规则 C1–C5 固化）
    │       ├── test_repro.py   可重跑性验收（同参 hash/字节一致、参数敏感性、脱敏抽查）
    │       ├── simulate_w1.py        ★ W1 演练：仿真 Recon 原始表生成器
    │       ├── sim_inventory_duckdb.py ★ W1 演练：执行盘点模板附录 A SQL（DuckDB）
    │       └── sim_w1_reports.py     ★ W1 演练：基线 200 笔分层 + 工作坊 Kappa
    │
    ├── policies/recon/
    │   └── engine.yaml     策略 v1.0：硬规则（结构/主体/金额）优先于模型分数规则；
    │                       AUTO 仅"高置信 MATCHED+低金额+干净"交集；默认 REVIEW
    ├── contracts/examples/
    │   └── recon.golden.json  黄金样例 6 例（启动校验+策略热切换守卫，§7-A）
    │
    └── infra/              【W3/W4】部署（定稿 §2.1：单机 Compose，不上 K8s）
        ├── Dockerfile         python:3.11-slim + libgomp1（lightgbm 依赖）+ 非 root
        ├── docker-compose.yml decision-service + postgres:15（migrations 自动执行）
        │                      + minio（工件预留）；--profile w4 加 prometheus+grafana
        ├── env.example        环境变量样例（cp 为 .env；.env 已 gitignore）
        ├── migrations/0001_init.sql  五表 DDL（tasks/models/policies/logs/feedback）
        │                      硬约束：logs.request_id 幂等主键、models.data_snapshot_hash
        ├── monitoring/prometheus.yml W4 抓取配置（/metrics 端点为 W4 前置任务）
        ├── e2e_smoke.py       E2E 冒烟（--local 本地模式 / 容器模式）
        └── README.md          部署操作手册
```

**不进 git 的**（`training/.gitignore` + 根 `.gitignore`）：`out/`（数据产物）、
`*.csv/*.parquet`、盐值文件、`reports/`、`artifacts/`、`.venv/`。
数据不进仓库（定稿 §2.4）——只进脚本、配置与快照 hash。

---

## 二、一次决策的完整链路（数据流）

```
Recon（Go-zero）──HTTP──▶ POST /decide（body: request_id, tenant_id, 付款×候选发票）
  │
  ├─ 1 Schema 校验（Pydantic v2；400 SCHEMA_INVALID）
  ├─ 2 幂等短路（request_id 已存在 → 409 + 首次结果，零副作用）
  ├─ 3 特征构建（features.py 纯函数，26 维，PIT 安全）
  ├─ 4 模型推理（model.txt → booster.predict(raw_score=True) 原始分数）
  ├─ 5 校准还原（calibration.json → Platt/isotonic → 三态概率）
  ├─ 6 策略路由（engine.yaml 受限 AST 求值：硬规则优先，默认 REVIEW）
  ├─ 7 shadow 掩码（mode=shadow 时 AUTO→对外 REVIEW，落库记真实决策）
  └─ 8 落库（decision_logs 全量可追溯：四版本+digest+latency）
人工复核 ──▶ POST /v1/feedback/{request_id}（pending→approved；仅 approved 进下一版训练集）
```

**故障语义**：模型/策略未就绪、推理失败 → 一律 503 `{"action":"HUMAN"}` 全停，
上游交人工，绝不静默降级（§7-E）。紧急叫停：`POST /admin/mode {"mode":"shadow"}`。

---

## 三、快速开始

```bash
cd aivora-decision

# 1) 环境（Python 3.10+；lightgbm 固定 <4.6）
python -m venv .venv && .venv\Scripts\pip install -r service/requirements-service.txt

# 2) 测试（50+ 用例，全离线）
.venv\Scripts\python training\pipelines\test_features.py    # 11/11
.venv\Scripts\python training\policies\test_policy.py       # 13/13
.venv\Scripts\python training\data\test_repro.py            # 可重跑性
.venv\Scripts\python service\test_service.py                # 11/11（JsonlStore）
# Postgres 集成（需先起 PG 并执行 infra/migrations/0001_init.sql）：
set AIVORA_TEST_DB_URL=postgresql://user:pass@localhost:5432/aivora_decision
.venv\Scripts\python service\test_postgres_store.py         # 8/8

# 3) 训练（合成数据自检，仅验证流程）
.venv\Scripts\python training\pipelines\train.py            # → D:\tmp\selftest-artifact
.venv\Scripts\python training\pipelines\verify_deploy.py D:\tmp\selftest-artifact

# 4) 服务（JsonlStore MVP）
.venv\Scripts\python -m uvicorn app:app --port 8090   # cwd=service/
# 或 Postgres 后端：设置 AIVORA_DB_URL 后同上

# 5) 容器部署（W3 交付，已 E2E 验证）
cd infra && cp env.example .env
docker compose up -d --build
docker compose cp ../training/out/.../artifact/. decision-service:/app/artifacts/recon-v1/
curl -X POST localhost:8090/admin/model/load -d '{"artifact_dir":"/app/artifacts/recon-v1"}'
```

---

## 四、当前状态（2026-11-07）

| 层 | 状态 |
|---|---|
| 规划文档 | ✅ 定稿基线 + D5 会前材料/3 份 W1 模板建议值已预填 |
| W1 演练 | ✅ 仿真全流程跑通（盘点 SQL 实测、管线可重跑、Kappa 0.85、基线 200 笔） |
| W2 训练管线 | ✅ 特征/切分/校准/评估/工件导出，合成数据自检通过（**未做真实数据训练**） |
| W3 服务 | ✅ API 契约齐（幂等/租户/feedback/审计）、策略引擎、shadow、双存储后端、E2E 冒烟（本地+容器）全绿 |
| 待办（业务/外部依赖） | 真实 Recon 数据盘点、标签工作坊（业务签字）、D5 决策会、Shadow 对接契约（§7-B） |
| 待办（W4） | /metrics 端点 + Grafana 面板、Recon 测试环境接入、历史回放、go/no-go |
