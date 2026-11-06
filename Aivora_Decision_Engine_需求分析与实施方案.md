# Aivora Decision Engine —— 需求分析与实施方案

> 版本：1.0
> 日期：2026-11-05
> 基于：《Aivora_Decision_Engine_需求与技术方案.md》（v0.1 初稿）
> 说明：本文档分两部分。**第一部分**对 v0.1 需求做批判性分析（需求拆解、缺口、风险、6 项待决策的明确建议）；**第二部分**给出可直接执行的实施方案（一期架构、Recon PoC 详细设计、评估门槛、30 天计划、技术选型）。文中所有阈值、样本量、SLO 数字均为**示例值**，需在拿到真实数据后由业务与数据共同标定。

---

# 第一部分 需求分析

## 1.1 项目本质与总体判断

用一句话概括本项目：

> **在业务系统旁边加一层"模型无关、可验证、可拒绝"的决策服务：把结构化业务状态转成带校准置信度的类型化决策，并让模型预测与业务动作（Policy）彻底分离。**

对 v0.1 初稿的总体判断：

1. **方向正确，定位克制。** "Decision-first 而非 generation-first"、"先传统机器学习基线再上神经网络"、"不预训练、不复刻 Jev"、"30 天 MVP + 明确停止条件"——这些约束让项目避开了最常见的 AI 基础设施陷阱（先建平台后找场景、先上大模型后降成本）。
2. **架构分层合理**（Schema 校验 → 特征 → 模型 → 校准/策略 → 类型化结果），且"模型与策略分离""置信度必须校准""允许拒绝决策"三条原则是整个方案的灵魂，后续所有设计都应服务于这三条。
3. **主要问题不在方向，而在颗粒度**：v0.1 是一份优秀的"讨论稿"，但距离"可开工"还差一层——缺具体的 SLO、风险分级定义、最小数据量假设、降级策略、人工复核工作流和成本预算（详见 1.4 缺口清单）。
4. **一个容易在实现中走偏的点**：`REVIEW` 在 v0.1 中同时扮演"模型标签"（8.2 节四类标签之一）和"决策动作"（5.4 节策略输出），语义混用会导致标签污染，必须在标签规范中先行厘清（详见 1.3.4 与 2.2.2）。

## 1.2 需求拆解与优先级（MoSCoW）

### 功能需求

| 级别 | 需求 | 说明 |
|---|---|---|
| **Must** | 类型化任务定义（Choice/Score/Probability/Rank/Multi-label/Regression） | 一期只需**实现 Choice 与 Rank 两类**即可支撑 Recon PoC，其余类型在 Schema 层预留 |
| **Must** | 单条 + 批量推理、输入/输出 Schema 校验 | 校验是"结构化输出"承诺的基础 |
| **Must** | 模型注册与版本管理（artifact + 指标 + 数据版本） | 可追溯性的最小集合 |
| **Must** | 概率校准（isotonic/Platt）+ 校准指标 | 置信度可信的前提 |
| **Must** | Policy Engine（阈值 + 硬规则 → AUTO/REVIEW/REJECT） | 模型与业务动作分离 |
| **Must** | 拒绝决策（低置信/异常/结构复杂 → REVIEW） | 安全阀 |
| **Must** | 决策日志（模型/特征/策略版本 + 输入摘要 + 预测 + 动作） | 审计与闭环的数据基础 |
| **Must** | 人工复核回写接口（feedback API） | 闭环的最小一环 |
| Should | 离线回放、Shadow Mode | 验证手段，不是功能本身，但 30 天计划内必须有 |
| Should | 灰度/回滚（模型与策略各自可回滚） | 策略回滚比模型回滚更重要（阈值调整是日常操作） |
| Should | 漂移监测（一期仅做特征 PSI + 动作分布，不做完整 MLOps） | 防止过度工程 |
| Could | 文本特征（备注/摘要 embedding） | 二期 |
| Could | 小型 Transformer / 多任务统一模型 | 二、三期，且有明确触发条件 |
| **Won't（一期）** | 通用模型训练平台、模型市场、分布式训练、法律自动结论、全自动核销（未经业务验证） | 与 v0.1 §12 一致 |

### 非功能需求（v0.1 缺失，本文补充建议值）

| 维度 | 建议目标（示例值，待真实数据标定） |
|---|---|
| 在线延迟 | P50 ≤ 30ms，P95 ≤ 100ms，P99 ≤ 300ms（CPU，LightGBM，≤50 候选） |
| 批量吞吐 | 1,000 条 ≤ 2s（单节点，batch 推理） |
| 可用性 | 一期 ≥ 99.5%（内部服务）；**模型服务不可用时业务必须可降级**（见 G4） |
| 安全 | 多租户行级隔离（tenant_id 贯穿所有表与查询）、传输 TLS、敏感字段落库前脱敏/摘要 |
| 可追溯 | 任一历史决策可还原：task/model/policy/feature 版本 + 输入摘要 + 预测 + 动作 + 复核结论 |
| 可复现 | 固定随机种子、数据快照 hash、训练配置入库，同数据同配置可复现同模型 |

## 1.3 核心需求深度分析

### 1.3.1 模型与策略分离（最高优先级设计）

这是整个系统与"直接调 LLM/调模型"的本质区别。落地含义：

- API 响应中 `prediction`（模型输出）与 `decision.action`（策略输出）必须分属两个字段域，且响应中同时返回 `model_version` 与 `policy_version`；
- 策略必须是**独立版本化、可单独回滚、可被运营修改（经审批）的工件**，改阈值不需要重新部署模型；
- 硬规则（黑名单、金额上限、强制审批）优先级**永远高于**模型分数——v0.1 §5.4 已明确，实现上要保证规则顺序可审计（`reason_codes` 中列出触发的规则 id）。

### 1.3.2 置信度校准

v0.1 的原则（"置信度不能直接视为真实正确概率"）正确，需要补充执行细节：

- 数据三分：train / calibration（拟合校准器）/ test（只用于报告，不参与任何拟合）；
- 校准器选型：calibration 集 ≥ 2,000 条用 **isotonic regression**，不足用 **Platt scaling**；多分类逐类做 softmax 前校准或 per-class 二分类校准；
- 报告指标：ECE（10-bin）、Brier score、分位数可靠性曲线；**重点看高置信段**（p≥0.9 的样本中实际正确率是否 ≥ 95%——这是 AUTO 动作的安全前提）；
- 线上持续验证：按月把决策日志中"最终业务结果已知的样本"与当时的校准概率对比，监控校准漂移。

### 1.3.3 拒绝决策（REVIEW/ABSTAIN）

拒绝能力是"可验证"的关键：

- 拒绝的三类触发源：**低置信**（策略阈值）、**结构异常**（多对多/拆分/合并/币种不一致等 flag）、**分布外**（特征 PSI 超阈 / 输入校验失败）；
- 一期不做复杂的 OOD 检测模型（如 Mahalanobis / energy-based），用"结构 flag + 特征 PSI + 置信阈值"三件套即可覆盖 Recon 场景，避免过度工程；
- REVIEW 率本身是一个**要监控的业务指标**：过高说明模型能力不足或阈值过严，过低说明安全阀失效。

### 1.3.4 标签语义：REVIEW 是"过程标签"还是"状态标签"（v0.1 的隐藏矛盾）

v0.1 §8.2 把 `REVIEW` 列为与 `MATCHED/PARTIAL/UNMATCHED` 并列的第四类标签，但 §4.1 中匹配分类的输出只有三类，`REVIEW` 出现在核销建议（策略层）的输出中。这个混用有实际危害：

- 历史数据中"被人工复核过的样本"是**流程结果**（当时系统不敢自动处理），不是**状态事实**（这笔交易是否匹配）。把它当第四类教模型，等于让模型学会"把难样本扔出去"，而"难"的标准混杂了历史系统的规则缺陷；
- `UNMATCHED` 与 `REVIEW` 的边界（"确定不匹配" vs "证据不足"）在标注时极难一致，会拉低标注一致性（Kappa）。

**建议**：模型标签只用三类状态（MATCHED / PARTIAL / UNMATCHED）；"证据不足/结构复杂"由**特征层**表达（`is_multi_mapping`、`has_split_merge` 等 flag），由**策略层**路由到 REVIEW。若业务坚持要模型输出 REVIEW，则标签规范必须定义为"结构性不可判定（输入缺失/多对多）"并给出逐例裁决，且要求标注 Kappa ≥ 0.8 才允许进入训练。

### 1.3.5 数据闭环与标签治理

- 人工复核结论 ≠ 训练标签，中间必须有一道**审核**（v0.1 已明确）；实现上建议 feedback 表带 `status: pending → approved/rejected`，只有 approved 的修正才进入下一版训练集；
- 防止"模型输出被当标签"的循环污染：每次训练集构建脚本必须记录**标签来源**（人工复核 / 历史确认记录 / 规则推断），规则推断的标签只能用于 baseline，不能用于模型验收；
- 租户隔离：一期**只用单租户或已授权的少量租户数据训练**，跨租户训练必须等治理机制（授权记录、脱敏策略、保留期限）就位后再开放。

### 1.3.6 数据泄漏与 point-in-time 正确性

v0.1 提到"按时间或业务实体划分"，需强化为硬性规范：

- 切分键：**时间为主 + 交易主体（客户/供应商）为辅**。同一主体的样本不允许跨 train/test 出现（group split）；
- 所有特征必须是**决策时点可计算的**（point-in-time）：例如"供应商历史匹配率"只能取付款日期之前的统计；
- **禁止**把"历史核销结果""是否曾被人工处理"等事后信息作为特征（它们只能作为标签来源且需审计）。

## 1.4 v0.1 缺口清单与补齐建议

| # | 缺口 | 影响 | 补齐建议（本文给出的具体方案见第二部分） |
|---|---|---|---|
| G1 | 无性能 SLO 数字 | 无法验收、无法做容量规划 | 给出 P50/P95/P99、批量吞吐、可用性示例目标（§1.2） |
| G2 | 无"动作风险等级"定义 | Policy 引擎无从写起——"高置信可自动执行"的边界依赖风险分级 | 提出二维风险分级框架：动作类型 × 金额档 × 主体敏感度（§2.2.6） |
| G3 | 无最小数据量假设 | "能否训练"在拿到数据前无法回答，PoC 可能卡死 | 给出最小可行数据集标准：总量 ≥ 3,000、每类 ≥ 300、≥ 3 个月历史、≥ 10 个交易主体群（§2.3.1） |
| G4 | 无服务降级策略 | 模型服务故障会阻塞核销流程（业务不能接受） | 明确"决策服务是建议性质"：API 返回 503 时 Recon 回退现有规则/人工流程，业务不阻塞（§2.2.7 错误码） |
| G5 | 人工复核工作流不具体 | 闭环断在"谁来核、多久核、怎么核" | 定义最小复核界面 + 复核 SLA（24h 内）+ feedback API 回写（§2.2.7、§2.4 第 3 周） |
| G6 | 无成本预算 | 停止条件 5（成本 > 收益）无法评估 | 一期全 CPU（无需 GPU）；主要成本 = 标注（约 200–400 人时）+ 工程（2 人月）；推理成本可忽略（§2.3.3 给出收益测算公式） |
| G7 | 幂等/请求追踪只提了 request_id | 重试、对账、审计都依赖它 | request_id 7 天去重窗口，重复请求返回首次结果；决策日志以 request_id 为唯一键（§2.2.7 DDL） |
| G8 | 可观测性无指标清单 | 上线后无法回答"模型变笨了吗" | 一期指标集：QPS、延迟直方图、action 分布、REVIEW 率、每类校准概率 vs 实际正确率、特征 PSI；告警：REVIEW 率突增 > 2σ、错误率 > 1%（§2.3.2） |
| G9 | 目录结构缺部署与脚本 | 无法直接开工 | 补充 `infra/`（Dockerfile、compose）、`scripts/`（数据抽取/回放），并明确**数据不进 git**（§2.5） |
| G10 | API 按类型拆 6 个端点（/v1/choice、/v1/score…） | 一期只需 Choice+Rank，拆端点增加校验与维护成本 | 一期统一入口 `POST /v1/decisions` + task_id 路由；分类型端点留到对外开放期（§2.2.7） |

## 1.5 风险清单与缓解

| 风险 | 概率 | 影响 | 缓解措施 |
|---|---|---|---|
| R1 标签一致率低（MATCHED/PARTIAL/UNMATCHED 边界模糊） | 高 | 高 | 标签规范先冻结 + 100 例双人标注 Kappa ≥ 0.75 才开训；边界案例逐条裁决并归档 |
| R2 历史"自动核销"记录含错，当标签用会学错 | 高 | 高 | 只采纳"人工确认过"的记录为 gold；自动记录仅作 baseline 参照；抽检 5% |
| R3 复杂场景（多对多/拆分/合并/汇差/跨期）拖垮一期 | 中 | 高 | **一期显式排除**：结构 flag 命中 → 策略直接 REVIEW，模型只处理 1:1 干净场景；这同时是模型的安全阀 |
| R4 模型/数据漂移（新供应商、新支付渠道、财务政策变化） | 中 | 中 | 特征 PSI 月报 + action 分布监控 + 月度重训节奏；策略阈值可按月调整（策略回滚独立） |
| R5 过度工程（30 天内做 Feature Store / 完整 MLOps / 漂移平台） | 中 | 中 | 一期架构刻意最小化（§2.1）：特征=版本化纯函数、策略=YAML、存储=Postgres 单库；漂移只做 PSI |
| R6 数据授权/脱敏不到位 | 低 | 高（合规） | 数据抽取脚本内置脱敏（主体 ID 假名化、金额保留、文本截断）；授权记录入表 |
| R7 业务方对"自动核销"责任归属不清 | 中 | 高 | 一期**不自动执行真实核销**，Shadow → 小流量灰度前必须开责任认定会议（谁对自动动作负责、纠错成本谁承担）——列入 §4.2 决策会议 |

## 1.6 对 v0.1 §14 六项待决策的建议

| # | 待决策 | 建议 | 理由 |
|---|---|---|---|
| 1 | 首任务：交易匹配 vs 重复付款识别 | **交易匹配分类** | ① 数据量最大、标签最客观（付款-发票关系有业务事实依据）；② 直接命中 Recon 核心流程，价值最直接；③ 重复付款正例率极低（<1%），样本稀缺、评估困难，适合作为**第二个任务**（可复用一期基础设施） |
| 2 | 能否取得足够脱敏数据 | 以 §2.3.1 的最小数据集标准为**开工门槛**；拿不到就触发停止条件 1，不要先用合成数据"把流程跑通"来安慰自己——合成数据可以验证 pipeline，但**不能**用于验收模型效果 | 30 天计划第 1 周的前置任务就是数据盘点，早失败早止损 |
| 3 | 单任务 vs 多任务 | **单任务** | 一期只有一个任务（匹配分类），多任务统一模型是 §8.4 第三阶段的事，有明确触发条件 |
| 4 | 内部部署 vs 外部 API | **仅内部** | 与 §2.2 三阶段路线一致；对外开放的额外工作（认证/计量/SLA/租户治理）不应挤占 30 天 |
| 5 | 风险等级、审批边界、复核责任 | 采用 §2.2.6 的二维分级框架（动作 × 金额 × 主体），**一期 AUTO 动作只开放给"低风险 + 低金额 + 高置信 + 硬规则全过"的交集**；审批边界与责任归属在第 4 周灰度决策会上由业务负责人书面确认 | 数字由业务定，框架由技术给 |
| 6 | 目标：降人工 / 提覆盖 / 缩周期 | **北极星 = 降低人工工作量（复核率↓）+ 自动处理覆盖率↑**；核销周期缩短是其派生指标 | 前两者可直接由决策日志计算，第三个依赖端到端流程改造，不宜作为 PoC 验收指标；第 1 周先测现状基线（§2.4） |

---

# 第二部分 实施方案

## 2.1 一期总体架构（最小可运行版）

``` text
 Aivora Recon (Go-zero 业务服务，现有)
   │  ① 真实核销请求（内部 HTTP）
   ▼
 Aivora Decision Service（新增，单个 FastAPI 服务，Python 3.11）
   ├── Schema Validator      Pydantic v2（输入/输出强类型）
   ├── Feature Builder       版本化纯函数（feature_version 固定）
   ├── Model Runner          规则基线 + LightGBM（ONNX Runtime 加载）
   ├── Calibrator            isotonic/Platt（随模型工件一起版本化）
   ├── Policy Engine         YAML 声明式策略（独立版本，独立回滚）
   └── Decision Logger       结构化日志 → PostgreSQL
   ▼
 PostgreSQL 15（单库：tasks / models / policies / decision_logs / feedback）
 MinIO（S3 兼容）（模型工件、数据快照、评估报告）
 Prometheus + Grafana（监控；Grafana 可第 4 周再上，先用结构化日志）
```

**相对 v0.1 的简化决策（及理由）：**

1. **一期只有一个新服务**（FastAPI），不单独建 "Decision API / Gateway" 层——Go-zero 业务服务本身就是网关（鉴权、配额、编排），与 v0.1 §6.2 的分工一致。等到对外 API 期再拆 gateway。
2. **不做 Feature Store / 独立特征服务**——特征 = 请求时计算的纯函数（输入 JSON 进，特征向量出），版本化靠 `feature_version` 常量 + 单测。Recon 场景的特征全部来自请求输入，不需要离线特征库。
3. **不做模型注册中心平台**——模型注册 = 往 `decision_models` 表插一行 + 工件存 MinIO。
4. **部署 = Docker Compose 单机**（1 个 decision 容器 + Postgres + MinIO），不上 K8s。一期 QPS 远低于单机能力（LightGBM 单条 < 5ms）。

## 2.2 PoC 详细设计：Recon 交易匹配（recon.transaction_match@1.0）

### 2.2.1 任务定义

- 类型：Choice（三分类）+ 内嵌 Rank（对候选发票排序，取 Top-1 进入分类；候选数 ≤ 50）
- 输入：一笔付款 + N 张候选发票（Recon 按主体/金额初筛后的候选集）
- 输出：`MATCHED | PARTIAL | UNMATCHED` + 校准概率 + 命中发票 id
- 决策动作（策略输出）：`AUTO_RECONCILE | REVIEW | REJECT`（REJECT = 判定无匹配关系，不核销，返回未匹配池）

输入 Schema（JSON Schema 摘要，完整版入 `contracts/schemas/`）：

``` json
{
  "payment": {
    "payment_id": "string",
    "amount": "number > 0",
    "currency": "string (ISO-4217)",
    "date": "date",
    "payer_id": "string",
    "channel": "enum: BANK_TRANSFER|CHECK|ONLINE|OTHER",
    "reference": "string | null   // 付款附言/回单摘要"
  },
  "candidate_invoices": [
    {
      "invoice_id": "string",
      "amount": "number > 0",
      "currency": "string",
      "date": "date",
      "supplier_id": "string",
      "supplier_name": "string",
      "note": "string | null"
    }  // 1..50 项
  ]
}
```

### 2.2.2 标签规范（v1 草案要点）

| 标签 | 定义 | 典型例 |
|---|---|---|
| MATCHED | 付款与发票一一对应，主体一致，金额在可解释容差内（手续费/汇差已扣除或符合财务规则），业务上应核销 | 发票 10,000 元，付款 10,000 元，附言含发票号 |
| PARTIAL | 对应关系成立但存在可解释差异（部分付款、含手续费、小额尾差），需财务确认口径 | 发票 10,000，付款 8,000（余款 2,000 在途） |
| UNMATCHED | 无对应关系，或差异不可解释（主体不符且无代付协议、金额严重不符） | 付款主体与全部候选发票主体均无关联 |

**硬性规则：**
- `REVIEW` 不作为模型标签（见 1.3.4）；
- 标注前先裁决 20 个边界案例（拆分付款、代付、合并开票、跨期、退补差等）并写入规范附录；
- 100 例双人标注，Cohen's Kappa ≥ 0.75 方可全量标注；
- 每条标签记录 `label_source`：`human_review | historical_confirmed | rule_inferred`。

### 2.2.3 特征列表（feature_version 1.0，一期约 25 个）

| 分组 | 特征 | 说明 |
|---|---|---|
| 金额 | `amount_diff`、`amount_diff_ratio`、`amount_log10`、`partial_ratio` | 付款-发票差额、差比例（容差判断核心） |
| 时间 | `date_diff_days`、`is_cross_period` | 日期差、是否跨月/跨年（财务口径敏感） |
| 主体 | `supplier_exact_match`、`supplier_fuzzy_score`、`counterparty_match_rate_hist` | 精确匹配 / 名称归一化相似度 / **付款日前**主体历史匹配率 |
| 单据 | `invoice_number_in_reference`、`invoice_number_fuzzy_hit` | 附言中是否含（精确/模糊）发票号——强信号 |
| 结构 flag | `is_multi_mapping`、`has_split_merge`、`currency_mismatch`、`has_fee`、`is_first_counterparty` | **多对多/拆分/币种不符 → 策略直接 REVIEW，不进模型 AUTO 通道** |
| 渠道 | `channel_onehot`、`amount_bucket` | 支付渠道、金额分桶 |

约束：全部特征 point-in-time 安全（§1.3.6）；特征构建函数纯函数化、带单测，输入是任务 Schema 对象，输出是固定顺序的 numpy 向量 + 命名特征 dict（策略引擎消费 dict，模型消费向量）。

### 2.2.4 模型与基线

1. **规则基线（必须做，且是验收对照组）**：确定性规则集——附言含发票号且主体一致且金额差 ≤ 容差 → MATCHED；主体不一致 → UNMATCHED；其余 → 交人工（基线"覆盖率"）。基线的覆盖率与错误率就是一期要打败的对象。
2. **LightGBM**：3 分类（multiclass），结构化特征；训练配置入库（learning_rate、num_leaves、early stopping on calibration 集）；导出 ONNX 供服务加载。
3. **对照实验**：LightGBM vs 规则基线 vs（可选）逻辑回归，指标口径统一；CatBoost 作为备选（类别特征多时试一版）。
4. **排序**：候选集 > 1 时，用 LightGBM 的 MATCHED 概率做候选排序取 Top-K（K=3）进入分类，一期不做独立 Rank 模型。

### 2.2.5 校准

- 三分：train / calibration / test（group by 交易主体 + 按时间，test 为最后 30% 时间段）；
- calibration 集 ≥ 2,000 条 → isotonic，否则 → Platt；
- test 集报告：ECE(10-bin)、Brier、分位数可靠性；**验收重点：p(MATCHED)≥0.9 分段的实际正确率 ≥ 97%**（示例门槛，与 G2 风险分级联动）；
- 校准器作为模型工件的一部分（`calibration.json`）随版本发布。

### 2.2.6 风险分级与策略引擎

**风险分级框架（示例，数字由业务定）：**

| 维度 | 低风险 | 中风险 | 高风险 |
|---|---|---|---|
| 动作 | AUTO_RECONCILE ≤ 5 万 | AUTO_RECONCILE 5–50 万 | > 50 万 / 冲销 / 退款 |
| 主体 | 历史匹配率 ≥ 95% 的成熟供应商 | 新供应商 / 匹配率 80–95% | 黑名单 / 有异常历史 / 首次交易 |
| 结构 | 1:1、币种一致、无拆分 | 1:1 含手续费 | 多对多、拆分、合并、跨币种 |

**策略文件示例（`policies/recon/match_policy@1.0.yaml`）：**

``` yaml
task: recon.transaction_match
version: "1.0"
order: priority        # 自上而下，first-match-wins
defaults:
  action: REVIEW
  reason: DEFAULT_REVIEW
rules:
  - id: r01-complex-structure
    when: "any(f.is_multi_mapping, f.has_split_merge, f.currency_mismatch)"
    action: REVIEW
    reason: COMPLEX_STRUCTURE
  - id: r02-risk-entity
    when: "f.counterparty_risk_flag == 1 or f.is_first_counterparty == 1"
    action: REVIEW
    reason: RISK_ENTITY
  - id: r03-high-amount
    when: "payment.amount > 200000"
    action: REVIEW
    reason: HIGH_AMOUNT
  - id: r04-auto-match
    when: "calib.p_MATCHED >= 0.95 and payment.amount <= 50000
            and f.supplier_exact_match == 1 and f.invoice_number_in_reference == 1"
    action: AUTO_RECONCILE
    reason: HIGH_CONFIDENCE
  - id: r05-reject-unmatched
    when: "calib.p_UNMATCHED >= 0.90"
    action: REJECT
    reason: LOW_MATCH_PROBABILITY
  # 其余全部落入 defaults → REVIEW
```

实现要点：策略在启动时编译 + 校验（when 表达式白名单：只能引用 `f.*` 特征、`calib.*` 校准概率、`payment.*` 原始字段，**不能**引用模型内部量）；每次决策返回 `rules_triggered` 列表；策略版本独立于模型版本，可单独回滚。

### 2.2.7 API 契约（一期）

| Endpoint | 方法 | 功能 |
|---|---|---|
| `/v1/decisions` | POST | 统一决策入口（task_id 路由；body 同 v0.1 §7.2，加 `tenant_id`） |
| `/v1/decisions` | POST（`"mode": "shadow"`） | Shadow Mode：全链路执行但不落 AUTO 动作，Recon 侧只读建议 |
| `/v1/decisions/{request_id}` | GET | 查询历史决策（审计） |
| `/v1/feedback/{request_id}` | POST | 人工复核回写（`corrected_label`、`review_action`、`reason`、`reviewer`） |
| `/v1/tasks/{task_id}`、`/v1/models/{model_id}`、`/v1/policies/{policy_id}` | GET | 版本与定义查询 |

响应（在 v0.1 §7.3 基础上补充）：

``` json
{
  "request_id": "req-unique-id",
  "task_id": "recon.transaction_match",
  "task_version": "1.0",
  "model_version": "recon-match-0.1.0",
  "policy_version": "1.0",
  "feature_version": "1.0",
  "prediction": {
    "value": "MATCHED",
    "matched_invoice_id": "INV-2026-000123",
    "probabilities": {"MATCHED": 0.973, "PARTIAL": 0.021, "UNMATCHED": 0.006}
  },
  "calibration": {"method": "isotonic", "version": "0.1.0",
                   "calibrated_probabilities": {"MATCHED": 0.982, "PARTIAL": 0.014, "UNMATCHED": 0.004}},
  "decision": {"action": "AUTO_RECONCILE", "reason_codes": ["HIGH_CONFIDENCE"],
                "rules_triggered": ["r04-auto-match"]},
  "metadata": {"latency_ms": 8, "degraded": false}
}
```

**错误码与降级：**

| 码 | 场景 | 调用方行为 |
|---|---|---|
| 400 `SCHEMA_INVALID` | 输入不符合任务 Schema | 修正重试；计入监控 |
| 404 `TASK_NOT_FOUND` / `MODEL_NOT_FOUND` | 配置错误 | 告警，人工介入 |
| 409 `DUPLICATE_REQUEST` | request_id 7 天内重复 | **返回首次结果**（幂等） |
| 422 `FEATURE_ERROR` | 特征构建失败（输入异常） | 该笔走人工；样本进入异常池 |
| 503 `MODEL_UNAVAILABLE` | 模型/依赖故障 | **Recon 回退现有规则/人工流程，业务不阻塞**（G4）；告警 |

### 2.2.8 数据模型（PostgreSQL DDL 摘要）

``` sql
CREATE TABLE decision_tasks (
  task_id        text,
  version        text,
  spec_jsonb     jsonb NOT NULL,        -- 输入/输出 Schema、标签规范引用
  label_spec_uri text,                  -- MinIO: 标签规范文档
  PRIMARY KEY (task_id, version)
);

CREATE TABLE decision_models (
  model_id   text,
  task_id    text,
  version    text,
  artifact_uri text NOT NULL,           -- MinIO: onnx + calibration.json + config
  data_snapshot_hash text,              -- 训练数据快照指纹
  metrics_jsonb jsonb,                  -- F1/AUC/ECE/Brier + 切片
  status     text CHECK (status IN ('draft','canary','production','retired')),
  created_by text, notes text, created_at timestamptz DEFAULT now(),
  PRIMARY KEY (model_id, version)
);

CREATE TABLE decision_policies (
  policy_id   text,
  task_id     text,
  version     text,
  spec_jsonb  jsonb NOT NULL,           -- YAML 序列化
  status      text CHECK (status IN ('draft','active','retired')),
  effective_at timestamptz,
  PRIMARY KEY (policy_id, version)
);

CREATE TABLE decision_logs (
  request_id    text PRIMARY KEY,       -- 幂等键
  tenant_id     text NOT NULL,
  task_id       text, task_version text,
  model_version text, policy_version text, feature_version text,
  mode          text CHECK (mode IN ('online','shadow','replay')),
  input_digest  text NOT NULL,          -- 输入 hash；明细按脱敏策略存 input_uri
  prediction    jsonb, calibrated_prob jsonb,
  action        text, reason_codes jsonb, rules_triggered jsonb,
  latency_ms    int, degraded bool,
  created_at    timestamptz DEFAULT now()
);
CREATE INDEX ON decision_logs (tenant_id, task_id, created_at DESC);

CREATE TABLE decision_feedback (
  decision_log_id text REFERENCES decision_logs(request_id),
  reviewer        text, reviewed_at timestamptz,
  review_action   text,                 -- 维持 / 纠正为 MATCHED|PARTIAL|UNMATCHED
  correction_reason text,
  status          text CHECK (status IN ('pending','approved','rejected')),
  PRIMARY KEY (decision_log_id, reviewer)
);
```

## 2.3 评估体系与验收门槛

### 2.3.1 开工门槛（数据，第 1 周内判定，不满足即触发停止条件 1）

- 脱敏样本总量 ≥ 3,000 笔，其中 MATCHED ≥ 500、PARTIAL ≥ 300、UNMATCHED ≥ 500（示例）；
- 覆盖 ≥ 3 个月历史、≥ 10 个交易主体群（group split 后有足够 test 主体）；
- 数据授权与脱敏记录完备。

### 2.3.2 离线验收门槛（第 2 周）

| 指标 | 门槛（示例，真实数据后标定） |
|---|---|
| 3 类 macro-F1 | ≥ 规则基线 + 2pt（绝对值） |
| MATCHED 类 precision | ≥ 97%（AUTO 动作安全前提） |
| 校准 ECE（10-bin） | ≤ 0.03；p≥0.9 分段实际正确率 ≥ 97% |
| 拒绝能力 | test 集中"结构复杂/主体存疑"样本被策略路由 REVIEW 的比例 ≥ 95% |
| 泄漏检查 | group split 下 test 指标与随机 split 差距 < 3pt（否则泄漏） |

### 2.3.3 Shadow Mode 验收门槛（第 4 周，≥ 10,000 笔真实流量或 ≥ 2 周，取先到）

| 指标 | 门槛（示例） |
|---|---|
| 模型建议 vs 人工最终结论一致率 | ≥ 95% |
| **模型拟 AUTO 样本中人工推翻率** | **≤ 2%**（最关键安全指标） |
| 规则基线 vs 模型：AUTO 覆盖率提升 | ≥ +10pt（证明模型有增量价值） |
| 延迟 / 错误率 | P95 ≤ 100ms；错误率 ≤ 0.1% |

**业务收益测算公式**（用于 go/no-go 决策会）：

```
月节省人工 = 月交易量 × AUTO 覆盖率 × (1 − 推翻纠错率) × 单笔人工处理时长
示例：50,000 × 40% × 0.98 × 4min ≈ 78,400 min ≈ 1,307 人时/月 ≈ 1.6 FTE
```

（单笔人工处理时长 = 第 1 周现状基线测量值，抽样 ≥ 200 笔。）

### 2.3.4 监控指标集（上线后）

QPS、延迟直方图、action 分布、REVIEW 率、每 10-bin 校准概率 vs 实际正确率（月更）、特征 PSI（月更）、422/503 率。告警：REVIEW 率 7 日均值偏移 > 2σ；503 任意发生；422 率 > 1%。

## 2.4 30 天实施计划（修订版）

**资源**：算法工程师 1、后端工程师 1、数据工程 0.5、业务/财务专家 0.5、产品 0.2。

| 周 | 工作 | 交付物 | 验收 |
|---|---|---|---|
| **W1 定义与数据** | 标签规范工作坊（业务+财务）；数据抽取/脱敏/质检脚本；现状基线测量（抽样 200 笔测人工时长/复核率/差错率）；Decision Schema 冻结 | 标签规范 v1（含 20 边界案例裁决）、脱敏样本集 v0、现状基线报告、任务定义文档 | 100 例双人标注 Kappa ≥ 0.75；**开工门槛（§2.3.1）判定通过/不通过——不通过即止损** |
| **W2 基线模型** | 特征工程（纯函数+单测）；规则基线 + LightGBM 训练（可复现）；校准器拟合；评估 | 训练/评估脚本（配置入库）、混淆矩阵、错误案例分析报告、校准报告 | 离线门槛（§2.3.2）达标；同数据同配置可复现 |
| **W3 服务化** | FastAPI 服务（decisions/feedback/版本查询 + 幂等 + 降级）；Policy Engine（YAML 编译+校验）；Postgres 建表；Docker Compose 部署；Recon 测试环境接入 + Shadow 展示；最小复核界面（列表+单条复核+回写） | 可部署的 decision-service v0.1、策略文件 v1.0、复核界面 v0.1 | 端到端 1,000 笔回放成功；P95 ≤ 100ms（测试环境）；feedback 闭环可跑通 |
| **W4 接入验证** | 历史全量离线回放报告；Shadow 观察（真实流量）；监控面板；灰度方案（5%、低金额档、可一键回滚）；**go/no-go 决策会** | Shadow 报告、延迟/成本报告、灰度方案、决策会纪要 | Shadow 门槛（§2.3.3）判定；业务负责人书面确认风险边界与责任归属（R7） |

**里程碑门禁**：W1 末（数据门槛）、W2 末（离线门槛）、W4 末（Shadow 门槛 + 灰度决定）——任一不达标，按 v0.1 §12 停止条件处理（调参迭代 ≤ 1 次，仍不达标则暂停并复盘，不硬推）。

## 2.5 技术选型（定稿）

| 模块 | 选型 | 理由 / 备选 |
|---|---|---|
| 语言/运行时 | Python 3.11 | 与现有算法栈一致 |
| API 框架 | FastAPI + Pydantic v2 | Schema 校验即类型系统；异步 |
| 表格模型 | **LightGBM**（主） | 小样本表格数据首选；可解释（特征重要性）；备选 CatBoost、逻辑回归（对照组） |
| 推理 | ONNX Runtime（CPU） | 单条 < 5ms；无需 GPU |
| 校准 | scikit-learn isotonic/Platt | 成熟、可版本化 |
| 存储 | PostgreSQL 15（单库） | 元数据+日志+反馈一体；行级租户隔离 |
| 工件存储 | MinIO（S3 兼容） | 自建；有云则用云 S3 |
| 监控 | Prometheus + Grafana（W4）；一期先结构化日志 | 轻量 |
| 业务侧 | Go-zero（现有） | 鉴权/配额/编排，调用 decision-service |
| 部署 | Docker Compose（单机） | 一期 QPS 无需 K8s；预留 Helm 化路径 |
| 文本（二期） | Sentence Transformers（bge-small / text2vec 中文） | 备注/摘要 embedding，与结构化特征拼接 |
| **不引入**（一期） | K8s、Feature Store、MLOps 平台、GPU、向量数据库、Agent 框架 | 防止过度工程（R5） |

## 2.6 仓库结构（在 v0.1 §13 基础上修订）

``` text
aivora-decision/
├── docs/                  # PRD、架构、标签规范、评估计划（本分析文档归档于此）
├── contracts/
│   ├── schemas/           # JSON Schema（任务输入/输出、API 请求/响应）
│   └── examples/          # 黄金样例（含边界案例）
├── services/
│   └── decision-service/  # 一期唯一新服务
│       ├── app/           # FastAPI 应用
│       │   ├── api/       # 路由
│       │   ├── features/  # 版本化纯函数特征（feature_version 常量）
│       │   ├── models/    # runner + calibrator
│       │   ├── policy/    # YAML 编译、校验、执行
│       │   └── storage/   # Postgres 仓库、MinIO 客户端
│       ├── Dockerfile
│       └── tests/
├── training/
│   ├── data/              # 抽取/脱敏/质检脚本（数据本身不进 git）
│   ├── pipelines/         # 训练/校准/评估（配置入库，seed 固定）
│   └── reports/           # 评估报告模板与产出（git-ignored，存 MinIO）
├── policies/recon/        # 策略 YAML（版本化，与代码同评审）
├── infra/
│   ├── docker-compose.yml
│   ├── migrations/        # Postgres DDL（按版本编号）
│   └── grafana/           # 监控面板 JSON
├── scripts/               # 回放、shadow 对比、收益测算
└── README.md
```

---

# 第三部分 演进路线与执行

## 3.1 分阶段路线与触发条件

| 阶段 | 内容 | 进入触发条件 |
|---|---|---|
| **P1（第 1–4 周）** | Recon 匹配分类 MVP：Shadow → 5% 小流量灰度（低金额档） | 立项 + 数据门槛通过 |
| **P2（第 2–4 月）** | ① 重复付款识别（第二任务，复用基础设施）；② 独立 Rank 模型（候选排序）；③ 文本 embedding 特征上线对照；④ 漂移监测（PSI 月报）+ 月度重训节奏；⑤ 灰度扩量至 AUTO 覆盖率 60–70% | P1 灰度 30 天：推翻率持续 ≤ 2%、责任归属已书面确认 |
| **P3（第 4–6 月）** | Aivora Supply（注意：补货 = 需求预测 + 优化算法 + 策略组合，不是单分类问题，需引入预测/优化模块）；Aivora Contract（小模型做条款分类/路由/风险评分，**LLM 做条款证据抽取与解释**，决策引擎输出类型化结论+证据引用——machine-native 与生成式能力的正确分工） | P2 稳定 + 目标任务数据门槛通过 |
| **P4（6 月+）** | 对外开放 API：认证/计量/配额/SLA/多租户治理；拆独立 gateway；类型化端点（/v1/choice、/v1/rank…） | 内部 ≥ 3 个任务稳定运行 + 治理机制验收 |

## 3.2 与 LLM/Agent 的边界（长期原则）

- 决策引擎输出**类型化决策**，LangGraph 等 Agent 框架可以作为**调用方**（把决策当工具），但 Agent 不替代策略层——任何自动执行动作必须经过 Policy Engine 与审计日志；
- 需要"理解"的场景（合同条款、邮件、备注）用 LLM 做**证据抽取**，把抽取结果作为**特征**喂给决策模型/策略，而不是让 LLM 直接出业务动作——这保持了可验证性与低成本的平衡。

## 3.3 立即行动清单（本周）

1. **数据盘点**（D1–D2，数据工程+业务）：对照 §2.3.1 门槛，确认 Recon 脱敏数据可得性——这是全项目第一个 go/no-go；
2. **标签工作坊**（D3–D4，业务/财务主导）：冻结 MATCHED/PARTIAL/UNMATCHED 边界 + 20 个边界案例裁决（§2.2.2）；
3. **现状基线测量**（D4–D5，业务）：抽样 200 笔核销，测人工处理时长、复核率、差错率——收益公式的输入；
4. **责任与风险会议**（D5，产品+业务负责人+财务）：确认 §1.6 六项决策、风险分级数字（§2.2.6）、灰度责任归属（R7）；
5. **工程启动**（W2 并行准备）：仓库初始化（§2.6）、Postgres DDL 评审、FastAPI 骨架 + CI。

## 3.4 决策会议议程建议（对应 v0.1 §14）

1. 首任务确认：交易匹配分类（建议，附 §1.6 理由）；
2. 数据门槛判定结果与授权/脱敏确认；
3. 风险分级数字：AUTO 金额上限、高置信阈值（0.95 示例值）、黑名单口径；
4. 北极星指标确认：人工工作量↓ + 自动覆盖率↑，及基线数值；
5. 复核责任：谁复核、SLA（24h）、AUTO 动作的责任与纠错成本归属；
6. 资源确认：算法 1 / 后端 1 / 数据 0.5 / 业务 0.5 / 产品 0.2；
7. 停止条件再确认：W1/W2/W4 三个门禁不达标时的处置（迭代 1 次 vs 暂停复盘）。

---

## 附：本文对 v0.1 的净增结论（TL;DR）

1. **v0.1 方向与原则正确，可开工的前提是补齐颗粒度**：SLO、风险分级、最小数据量、降级策略、复核工作流、成本——本文 §1.4 已逐条给出。
2. **修正一个隐藏矛盾**：REVIEW 不做模型标签，只做策略动作；模型三类状态 + 结构 flag 表达"不可判定"（§1.3.4）。
3. **一期刻意做小**：一个 FastAPI 服务 + Postgres + LightGBM + YAML 策略 + Compose 部署；特征=纯函数、漂移=PSI、无 K8s/Feature Store/MLOps（§2.1、§2.5）。
4. **三个门禁代替模糊验收**：W1 数据门槛 → W2 离线门槛（MATCHED precision ≥ 97% 等）→ W4 Shadow 门槛（**拟 AUTO 推翻率 ≤ 2%** 是安全红线）（§2.3）。
5. **六项待决策全部给出建议**（§1.6）：首任务=交易匹配；单任务；仅内部部署；风险二维分级框架；北极星=降人工+提覆盖；数据门槛不满足即止损。
6. **核心判断与 v0.1 一致并强化**：先在一个具体流程（Recon 匹配）中证明"小模型 + 校准置信度 + 策略分离 + 拒绝决策"能稳定产生可度量的业务价值，再谈扩展。
