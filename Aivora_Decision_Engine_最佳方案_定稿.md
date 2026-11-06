# Aivora Decision Engine 最佳方案（定稿）

> 版本：1.0（定稿基线）
> 日期：2026-11-05
> 依据：《需求与技术方案》v0.1 + 《需求分析与实施方案》v1.0
> 说明：本文是两稿合并后的**执行基线**。凡与 v0.1/v1.0 冲突处，以本文为准；
> 文中所有阈值、样本量、SLO 仍为**示例值**，须在 D5 决策会与拿到真实数据后标定。
> 本文不重复 v1.0 已完整给出的 DDL/Schema/YAML 全文，以引用为主，只固化决策与新增约束。

---

## 0. 一页结论

1. **定位不变**：业务系统旁挂一层"模型无关、可验证、可拒绝"的决策服务；模型预测（prediction）与业务动作（decision.action）彻底分离。
2. **执行基线 = v1.0 实施方案**：一个 FastAPI 服务 + Postgres 15 + MinIO + LightGBM（ONNX/CPU）+ YAML 策略引擎，Docker Compose 单机部署。
3. **首任务**：`recon.transaction_match@1.0`（三分类 + 内嵌候选排序），一期显式排除多对多/拆分/合并/跨币种结构（由策略 flag 直接路由 REVIEW）。
4. **三个门禁**：W1 数据门槛 → W2 离线门槛 → W4 Shadow 门槛（拟 AUTO 推翻率 ≤2% 为安全红线）。任一不达标：调参迭代 ≤1 次，仍不达标即暂停复盘，不硬推。
5. **北极星指标**：人工复核工作量↓ + 自动处理覆盖率↑（核销周期缩短为派生指标，不作 PoC 验收）。
6. **本文新增 8 项执行约束**（§7），补齐 v1.0 残留的歧义点，全部纳入 W1–W3 交付物。

---

## 1. 决策固化（原 v0.1 §14 六项待决策 → 定稿）

| # | 事项 | 定稿决策 | 备注 |
|---|---|---|---|
| D1 | 首任务 | **交易匹配分类**（含候选排序） | 重复付款识别列为 P2 第二任务（正例率 <1%，样本稀缺，但可复用一期全部基础设施） |
| D2 | 数据可得性 | 以 v1.0 §2.3.1 为**开工门槛**（总量 ≥3,000；MATCHED ≥500、PARTIAL ≥300、UNMATCHED ≥500；≥3 个月、≥10 个主体群；授权/脱敏记录完备） | **W1 末判定，不通过即触发停止条件 1**。合成数据只许验证 pipeline，不许用于验收 |
| D3 | 单/多任务 | **单任务** | 多任务统一模型是 v0.1 §8.4 第三阶段的事，有触发条件 |
| D4 | 部署范围 | **仅 Aivora 内部**，第一阶段不开放外部 API | 认证/计量/SLA/租户治理留到 P4 |
| D5 | 风险分级与责任 | 采用 v1.0 §2.2.6 二维框架（动作×金额×主体）；一期 AUTO 只开放"低风险+低金额+高置信+硬规则全过"交集；**数字由业务在 D5 决策会书面确认**，含 AUTO 责任与纠错成本归属 | 框架技术给，数字业务定 |
| D6 | 目标 | 北极星 = 降人工 + 提覆盖；W1 完成现状基线测量（抽样 ≥200 笔：人工时长、复核率、差错率） | 收益公式的输入 |

**REVIEW 语义（关键修正，v1.0 §1.3.4）**：
- 模型标签只有三类：`MATCHED / PARTIAL / UNMATCHED`；
- `REVIEW` 是策略层动作，不是训练标签；
- "证据不足/结构复杂"由特征层 flag（`is_multi_mapping`、`has_split_merge`、`currency_mismatch`…）表达，由策略路由；
- 若业务坚持模型输出 REVIEW，标签规范须定义为"结构性不可判定"并逐例裁决，且 Kappa ≥ 0.8 才可入训。**默认按前三条执行。**

---

## 2. 一期架构（定稿）

采用 v1.0 §2.1 最小可运行版，四个简化决策维持不变：

```
Aivora Recon（Go-zero，现有）
   │  真实核销请求（内部 HTTP）
   ▼
Decision Service（新增，唯一新服务，FastAPI / Python 3.11）
   ├── Schema Validator    Pydantic v2
   ├── Feature Builder     版本化纯函数（feature_version 常量 + 单测）
   ├── Model Runner        规则基线 + LightGBM（ONNX Runtime, CPU）
   ├── Calibrator          isotonic/Platt（随模型工件版本化）
   ├── Policy Engine       YAML 声明式（独立版本、独立回滚）
   └── Decision Logger     结构化日志 → PostgreSQL
PostgreSQL 15（单库：tasks / models / policies / decision_logs / feedback）
MinIO（模型工件、数据快照、评估报告）
Prometheus + Grafana（W4 上线；此前用结构化日志）
```

- 不单独建 Gateway（Go-zero 即网关）；
- 不做 Feature Store（特征 = 请求时纯函数）；
- 模型注册 = 表插一行 + 工件存 MinIO；
- 不上 K8s（单机 Compose；QPS 远低于单机能力，LightGBM 单条 <5ms）。

### 2.1 技术选型（定稿 = v1.0 §2.5）

| 模块 | 选型 |
|---|---|
| 语言 | Python 3.11 |
| API | FastAPI + Pydantic v2 |
| 表格模型 | LightGBM（主）；CatBoost / 逻辑回归（对照组） |
| 推理 | ONNX Runtime（CPU） |
| 校准 | scikit-learn isotonic/Platt |
| 存储 | PostgreSQL 15（行级租户隔离） |
| 工件 | MinIO（有云则云 S3） |
| 监控 | 结构化日志（W1–W3）→ Prometheus + Grafana（W4） |
| 业务侧 | Go-zero（现有） |
| 部署 | Docker Compose 单机，预留 Helm 路径 |
| 文本（P2） | Sentence Transformers（bge-small / text2vec 中文） |
| **一期不引入** | K8s、Feature Store、MLOps 平台、GPU、向量库、Agent 框架 |

### 2.2 API（定稿 = v1.0 §2.2.7，推翻 v0.1 六端点拆分）

一期统一入口，按 task_id 路由：

| Endpoint | 方法 | 功能 |
|---|---|---|
| `/v1/decisions` | POST | 统一决策入口（body 含 `tenant_id`、`request_id`；`mode: shadow` 走 Shadow） |
| `/v1/decisions/{request_id}` | GET | 历史决策查询（审计） |
| `/v1/feedback/{request_id}` | POST | 人工复核回写 |
| `/v1/tasks|models|policies/{id}` | GET | 版本与定义查询 |

分类型端点（/v1/choice、/v1/rank…）留到 P4 对外开放期。

错误码与降级（定稿）：

| 码 | 场景 | 调用方行为 |
|---|---|---|
| 400 SCHEMA_INVALID | 输入不合 Schema | 修正重试；计入监控 |
| 404 TASK/MODEL_NOT_FOUND | 配置错误 | 告警，人工介入 |
| 409 DUPLICATE_REQUEST | request_id 7 天内重复 | **返回首次结果（幂等）** |
| 422 FEATURE_ERROR | 特征构建失败 | 该笔走人工；入异常池 |
| 503 MODEL_UNAVAILABLE | 模型/依赖故障 | **Recon 回退现有规则/人工，业务不阻塞**；告警 |

响应契约：`prediction` 与 `decision` 分属两个字段域；同时返回 task/model/policy/feature 四个版本号 + `rules_triggered` + `degraded` 标志（全文见 v1.0 §2.2.7）。

### 2.3 数据模型

DDL 定稿 = v1.0 §2.2.8（decision_tasks / decision_models / decision_policies / decision_logs / decision_feedback 五表）。两条硬约束：
- `decision_logs.request_id` 为幂等唯一键；
- `decision_models` 必须带 `data_snapshot_hash`（与 §7-C 可重跑数据管线配套）。

### 2.4 仓库结构

定稿 = v1.0 §2.6（单服务 `services/decision-service/`，含 `features/ models/ policy/ storage/`；`training/` 管线与报告；`policies/recon/` 版本化策略 YAML；`infra/` Compose + migrations + grafana）。**数据不进 git**（仅脚本、配置、快照 hash）。

---

## 3. PoC 设计要点（recon.transaction_match@1.0）

定稿 = v1.0 §2.2，此处只列不可再议的要点：

1. **任务**：Choice（三分类）+ 内嵌 Rank（候选 ≤50，按 p(MATCHED) 排序取 Top-3 进分类；一期不做独立 Rank 模型——但见 §7-G 日志要求）。
2. **输入**：一笔付款 + N 张候选发票（Recon 已按主体/金额初筛）。
3. **标签规范 v1**：MATCHED/PARTIAL/UNMATCHED 三态定义 + 20 个边界案例逐条裁决（拆分、代付、合并开票、跨期、退补差…）写入附录；标注前 100 例双人 Kappa ≥ 0.75 方可全量；每条标签记 `label_source`（human_review / historical_confirmed / rule_inferred）。
4. **特征 v1**：约 25 个，五组（金额/时间/主体/单据/结构 flag + 渠道）；**全部 point-in-time 安全**；纯函数 + 单测；禁止把事后信息（历史核销结果、是否曾被人工处理）当特征。
5. **模型**：规则基线（必须做，是验收对照组）→ LightGBM 三分类（ONNX 导出）→ 可选逻辑回归/CatBoost 对照。切分：时间为主 + 主体 group split，test = 最后 30% 时间段。
6. **校准**：train/calibration/test 三分；calibration ≥2,000 条用 isotonic，否则 Platt；校准器（calibration.json）随模型工件发布。
7. **策略引擎**：first-match-wins 规则链；硬规则（黑名单/金额上限/结构 flag）优先级永远高于模型分数；默认落 REVIEW。策略表达式白名单：只许引用 `f.*`、`calib.*`、`payment.*`（求值机制见 §7-A）。
8. **数据闭环**：feedback 表 `status: pending → approved/rejected`，仅 approved 修正进入下一版训练集；规则推断标签只用于 baseline，不用于验收。

---

## 4. 验收门禁（定稿）

| 门禁 | 时点 | 标准（示例值，待标定） | 不达标处置 |
|---|---|---|---|
| **G-数据** | W1 末 | §1-D2 全部满足；100 例 Kappa ≥ 0.75 | 触发停止条件 1，止损 |
| **G-离线** | W2 末 | macro-F1 ≥ 规则基线 + 2pt；MATCHED precision ≥ 97%；ECE(10-bin) ≤ 0.03 且 p≥0.9 分段实际正确率 ≥ 97%；复杂结构样本被路由 REVIEW 比例 ≥ 95%（见 §7-D）；group split 与随机 split 指标差 < 3pt（泄漏检查） | 迭代 ≤1 次；仍不达标 → 暂停复盘 |
| **G-Shadow** | W4 末 | ≥10,000 笔真实流量或 ≥2 周（取先到）；模型建议 vs 人工最终结论一致率 ≥ 95%；**拟 AUTO 样本人工推翻率 ≤ 2%（安全红线）**；AUTO 覆盖率较基线 +10pt；P95 ≤ 100ms、错误率 ≤ 0.1% | go/no-go 决策会定；不达标不灰度 |

收益公式（go/no-go 输入）：
```
月节省人工 = 月交易量 × AUTO 覆盖率 × (1 − 推翻纠错率) × 单笔人工处理时长
```
单笔人工处理时长 = W1 现状基线测量值（抽样 ≥200 笔）。

上线后监控集（定稿 = v1.0 §2.3.4）：QPS、延迟直方图、action 分布、REVIEW 率、10-bin 校准 vs 实际正确率（月更）、特征 PSI（月更）、422/503 率。告警：REVIEW 率 7 日均值偏移 >2σ；503 任意发生；422 率 >1%。

---

## 5. 30 天计划（定稿 = v1.0 §2.4，含本文补充项）

资源：算法 1 / 后端 1 / 数据 0.5 / 业务财务 0.5 / 产品 0.2。

| 周 | 工作 | 交付物 | 验收 |
|---|---|---|---|
| **W1** | 标签工作坊；数据抽取/脱敏/质检脚本（**须可重跑，见 §7-C**）；现状基线测量（200 笔）；Decision Schema 冻结；**与 Recon 团队冻结 Shadow 对接契约（见 §7-B）** | 标签规范 v1（含 20 边界裁决）、脱敏样本集 v0、基线报告、任务定义、Shadow 接口契约 | Kappa ≥0.75；G-数据判定 |
| **W2** | 特征工程（纯函数+单测）；规则基线 + LightGBM（可复现：seed、配置、快照 hash 入库）；校准器拟合；评估 + 错误分析 | 训练/评估脚本、混淆矩阵、错误案例报告、校准报告 | G-离线达标 |
| **W3** | FastAPI 服务（decisions/feedback/版本查询/幂等/降级）；Policy Engine（**安全求值器，见 §7-A**）；Postgres 建表；Compose 部署；Recon 测试环境接入 Shadow；最小复核界面（列表+单条+回写，SLA 24h）；**事故回滚 runbook（见 §7-E）** | decision-service v0.1、策略 v1.0、复核界面 v0.1、runbook | 1,000 笔端到端回放；P95 ≤100ms；feedback 闭环跑通 |
| **W4** | 历史全量回放报告；Shadow 观察；监控面板（Prometheus+Grafana）；灰度方案（5%、低金额档、一键回滚）；**go/no-go 决策会** | Shadow 报告、延迟/成本报告、灰度方案、决策会纪要 | G-Shadow 判定；业务负责人书面确认风险边界与责任（R7） |

### 5.1 本周立即行动（D1–D5，定稿 = v1.0 §3.3）

1. D1–D2 数据盘点（对照 G-数据，全项目第一个 go/no-go）；
2. D3–D4 标签工作坊（冻结三态边界 + 20 边界裁决）；
3. D4–D5 现状基线测量（200 笔）；
4. D5 决策会（§1 六项 + 风险数字 + 责任归属 + 资源）；
5. W2 并行：仓库初始化（§2.4）、DDL 评审、FastAPI 骨架 + CI。

---

## 6. 演进路线（定稿 = v1.0 §3.1，含触发条件）

| 阶段 | 内容 | 进入条件 |
|---|---|---|
| **P1（1–4 周）** | Recon 匹配 MVP：Shadow → 5% 低金额灰度 | 立项 + G-数据通过 |
| **P2（2–4 月）** | 重复付款识别；独立 Rank 模型；文本 embedding 对照；PSI 月报 + 月度重训；灰度扩至 AUTO 覆盖 60–70% | P1 灰度 30 天：推翻率持续 ≤2% + 责任书面确认 |
| **P3（4–6 月）** | Supply（需求预测+优化算法+策略组合，非单分类）；Contract（小模型做分类/路由/评分，LLM 做证据抽取→特征，决策引擎出类型化结论+证据引用） | P2 稳定 + 目标任务数据门槛 |
| **P4（6 月+）** | 对外 API：认证/计量/配额/SLA/租户治理；拆 gateway；分类型端点 | 内部 ≥3 任务稳定 + 治理验收 |

LLM/Agent 边界（长期原则，定稿）：Agent 框架只能作为决策 API 的调用方；LLM 只出"证据抽取"当特征，不出业务动作；任何自动执行必须过 Policy Engine + 审计日志。

---

## 7. 本文新增的 8 项执行约束（补齐 v1.0 残留缺口）

以下条目为对 v1.0 的增量，全部纳入对应周的交付物：

### A. 策略表达式安全求值机制（W3）
- YAML 中 `when` 表达式**禁止裸 eval**。采用受限求值器：启动时解析为 AST，只允许比较/布尔运算与白名单标识符（`f.*`、`calib.*`、`payment.*`），拒绝一切内置名与属性链逃逸；
- 策略文件在启动时编译 + 校验（类型检查 + 干跑黄金样例），编译失败则拒绝启动；
- 每条规则有单元测试（含黄金样例 `contracts/examples/`），策略变更走代码评审 + 业务负责人审批（§J）。

### B. Shadow 对接是联调依赖，须 W1 冻结契约（W1）
- Shadow 模式要求 Go-zero 侧实现"只读建议"分支：调 decision-service、记对照日志、不影响真实流程；
- W1 末与 Recon 团队冻结接口契约（请求/响应/影子字段/对照日志格式），W3 联调时只实现不讨论；
- 若 Recon 团队 W3 前无法排期，Shadow 数据改用**离线回放 + 真实请求录放**兜底（G-Shadow 门槛中的"≥10,000 笔"以录放流量计入，但须在报告中标注非实时）。

### C. 数据管线必须可重跑（W1）
- 抽取/脱敏/质检脚本参数化（时间窗、租户、输出路径），重跑同参数产出同 `data_snapshot_hash`；
- 目的：P2 月度重训直接复用，且 `decision_models.data_snapshot_hash` 有实际锚点；
- 脱敏规则固化在脚本内（主体 ID 假名化、金额保留、文本截断），授权记录入表。

### D. 测试集保留复杂结构样本（W2）
- 复杂结构（多对多/拆分/合并/跨币种）样本**不进训练集**（R3），但**必须留在 test 集**并单独标记；
- G-离线的"复杂样本被路由 REVIEW ≥95%"在该子集上计算——否则拒绝能力无法验证；
- 该子集最小量：≥200 条；不足时放宽为"全部保留"并在报告标注置信区间。

### E. 事故回滚 Runbook（W3 交付）
书面 runbook，至少覆盖三个场景，各含操作命令 + 责任人 + 恢复验证：
1. **模型回滚**：`decision_models` 中 production 指针切回上一版本（工件已版本化，分钟级）；
2. **策略回滚**：同上切 `decision_policies`（改阈值是日常操作，策略回滚比模型回滚更常用）；
3. **503 全停**：decision-service 下线/故障时，Recon 侧确认走现有规则/人工流程，业务零阻塞（与 G4 降级一致）。
W4 灰度前做一次回滚演练并记录。

### F. 标注分期，避免一次性沉没（W1 起）
- 标注分两期：**先标 1,000 条**（覆盖三类 + 复杂结构子集）训 v0 模型，出错误分析后再决定全量标注范围；
- 错误分析若显示某类混淆集中（如 PARTIAL/UNMATCHED 边界），针对性补标该边界样本，而非盲目全量；
- 全量规模 >3,000 时，每 500 条做一次双人抽检，Kappa <0.75 立即暂停并回炉标签规范；
- 预算口径仍按 200–400 人时上限，分期只是把风险前置。

### G. Rank-分类耦合的日志支撑（W2/W3）
- 一期用 p(MATCHED) 排序取 Top-3 是已知妥协；为 P2 独立 Rank 模型留证据：
- `decision_logs` 的 `prediction` 字段除 Top-1 外，**记录 Top-5 候选及其校准概率**；
- Shadow 报告增加一项："Top-1 排序错误（正确发票在候选中但未被排到 Top-1）"的比例——若 >5%，P2 独立 Rank 的优先级上调。

### H. 政策变更审批流（W3 起生效）
- 策略 YAML 与代码同仓库、同评审；
- 生产策略变更流程：发起（业务或算法）→ 业务负责人书面确认风险数字变更 → 代码评审 → 版本发布（新 policy_version）→ 观察 48h REVIEW 率与推翻率；
- 任何绕过审批的紧急变更（如安全事件）须在 24h 内补审批并归档。

---

## 8. 风险清单（定稿 = v1.0 §1.5，无新增条目，执行注记如下）

| 风险 | 缓解（定稿） |
|---|---|
| R1 标签一致率低 | 标签规范先冻结 + 100 例 Kappa ≥0.75；§7-F 分期抽检机制 |
| R2 历史自动核销含错 | 只采纳人工确认记录为 gold；自动记录仅 baseline；抽检 5% |
| R3 复杂场景拖垮一期 | 一期显式排除 + 结构 flag 策略直路由 REVIEW；§7-D 测试集验证 |
| R4 模型/数据漂移 | PSI 月报 + action 分布监控 + 月度重训（依赖 §7-C 可重跑管线）；策略阈值按月可调、独立回滚 |
| R5 过度工程 | §2 四个简化决策 + "一期不引入"清单为硬约束 |
| R6 数据授权/脱敏 | §7-C 脱敏固化在抽取脚本；授权记录入表 |
| R7 自动核销责任不清 | 一期不自动执行真实核销；D5 决策会书面确认；§7-E runbook 兜底 |

---

## 9. 核心判断（收束）

> **先用 30 天在 Recon 交易匹配一个具体流程里，证明"小模型 + 校准置信度 + 策略分离 + 拒绝决策"能产生可度量、可追溯、可回滚的业务价值（人工推翻率 ≤2% 且 AUTO 覆盖率 +10pt），再谈扩展。**
>
> 项目的成败不取决于模型多强，而取决于三个门禁是否被诚实地执行：
> 数据不够就停（G-数据），离线不过就不上（G-离线），Shadow 推翻率超标就不灰度（G-Shadow）。
