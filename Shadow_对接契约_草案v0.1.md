# Shadow 对接契约（草案 v0.1）—— Recon ↔ Decision Service

> 依据：定稿 §7-B（"Shadow 对接是联调依赖，须 W1 冻结契约；W3 联调时只实现不讨论"）
> 实测基础：2026-11-06/09 对 Recon 生产系统（`ivora-reconciliation`，recon.ivlacloud.com）的只读 API 探查
> 状态：**草案**——待 Recon 团队 + 业务负责人确认后冻结为 v1.0（D5 会后 T+3 动作项）
> 红线：Shadow 期对 Recon **零写操作、零流程影响**；全部对照离线计算（定稿 §7-B）

---

## 1. 目标与范围

在 W4 用真实流量验证"模型建议 vs 人工最终结论"，产出 G-Shadow 门禁所需的：
一致率 ≥95%、**拟 AUTO 样本人工推翻率 ≤2%（安全红线）**、AUTO 覆盖率较规则基线 +10pt、P95 ≤100ms。

- **目标人群**：Recon 现有引擎的 `PENDING_REVIEW` 匹配结果（score 0.5–0.9 的歧义/差异候选）——
  这正是现有规则基线"不敢自动放"的集合（实测 reason："Ambiguous: several invoices match the payer and amount" 等）；
- **不做**：不改变 Recon 任何状态（不调 `PATCH /matching/{id}/{action}`、不触发 `/matching/run`、不改 settings）；
  AUTO 动作在 Shadow 期由决策服务掩码为 REVIEW（`AIVORA_MODE=shadow`，已实现并单测覆盖）；
- **前提**：Shadow 流量需来自有真实交易量的租户（≥10,000 对或 ≥2 周，取先到）——
  当前 "Ivora Seed Demo" 工作区（6 付款）仅用于联调验证，**不计入门禁**。

## 2. 拓扑与阶段

### 阶段一（W3 联调 + W4 Shadow）：独立 Shadow Worker（**Recon 零改造**）

```
┌─────────────────────────┐         ┌──────────────────────────────┐
│ Recon (ivora-reconciliation) │        │ shadow-worker（新增，只读消费者） │
│  /payments /invoices    │◀──GET──│  轮询 PENDING_REVIEW 新对      │
│  /customers             │◀──GET──│  构建 DecideRequest            │
│  /matching/results      │◀──GET──│  记录对照日志                  │
│  （人工确认 → CONFIRMED）│         │  回填 human truth → 推翻率     │
└─────────────────────────┘         └──────────────┬───────────────┘
                                                   │ POST /decide (mode=shadow)
                                                   ▼
                                    ┌──────────────────────────────┐
                                    │ decision-service（已部署）      │
                                    │  503 全停 / 409 幂等 / 四版本落库 │
                                    │  AUTO 掩码 REVIEW 对外           │
                                    └──────────────────────────────┘
```

- Worker 是**只读** API 消费者：登录（专用 shadow 服务账号）+ GET 五个端点；
- 决策服务以 `AIVORA_MODE=shadow` 运行：照常推理落库（`is_shadow=true`、记真实 `model_action`），对外动作掩码 REVIEW；
- 推翻率离线计算：worker 轮询到人工确认（`status→CONFIRMED` + `confirmed_by`）后回填 truth，
  交 `ShadowReporter`（已实现，边界用例已单测：同意率/推翻率 0/1 正确）。

### 阶段二（灰度期，G-Shadow 通过后）：Recon 内嵌实时调用（需 Recon 团队开发）

- Recon 在匹配引擎产出 PENDING_REVIEW 时**同步**调 `POST /v1/decisions`（mode 由决策服务侧控制），
  响应只作为"建议"展示/记录，**不执行**任何自动动作（5% 低金额灰度才解锁 AUTO，走 D5 批准的灰度方案）；
- 本契约同样适用（请求/响应/日志格式不变），差异仅在调用位置（worker 轮询 → Recon 内嵌同步）。

### 兜底（定稿 §7-B）：离线回放 + 录放

若 worker 开发延期：用 `GET /matching/export`（17 列 CSV）+ payments/invoices 导出做离线回放，
决策日志 `mode='replay'`，**报告须标注"非实时"**（门槛中 ≥10,000 笔以录放流量计入，定稿已允许）。

## 3. 字段映射（Recon API → DecideRequest）

> 决策服务输入 = 单对单（一笔付款 × 一张候选发票）；候选集（1 付款 × N 发票）口径见 D5 附录 C"另注"，
> Shadow 期由 worker 逐对调用（先前者上 Shadow 的既定决策）。

| DecideRequest 字段 | Recon 来源 | 映射规则 | 备注 |
|---|---|---|---|
| `request_id` | 派生 | `shadow-{tenant_id}-{payment_id}-{invoice_id}` | 幂等键：重放/重轮询天然去重（409 返回首次结果） |
| `tenant_id` | `/auth/me` | 原样（UUID） | 行级隔离贯穿 |
| `payment_id` / `invoice_id` | `payments.id` / `invoices.id` | 原样（UUID） | 对照日志 join 键 |
| `pay_amount` / `inv_amount` | `payments.amount` / `invoices.amount` | **string decimal → Decimal → float**（Recon 用字符串传金额，worker 用 Decimal 中转防精度丢失） | |
| `pay_currency` / `inv_currency` | 各自 `currency` | 原样 | 不一致 → `currency_mismatch` 特征 → 策略 REVIEW |
| `pay_date` / `inv_date` | `payments.payment_date` / `invoices.invoice_date` | 原样（YYYY-MM-DD） | |
| `payer_id` | 解析 | `payer_name` → `customers`（`normalized_name`/`aliases` 精确匹配）→ `customer.id`；**未解析 → `UNK-{sha8(payer_name)}`** | 未解析 = 主体存疑，策略侧应 REVIEW（见 §6 扩展 flag） |
| `supplier_id` | `invoices.customer_id` | 原样 | |
| `supplier_name` | `invoices.customer_name` | 原样 | 特征 `supplier_fuzzy_score` 依赖 |
| `channel` | `payments.source` | `BANK_FEED→BANK_TRANSFER`，`MANUAL/API→OTHER`（other 渠道特征=0，登记影响） | |
| `reference` | `payments.payment_reference` | 原样（含票号文本，如 `INV-1005`）；空串兜底 | 仅喂 `reference_len_norm`；票号命中判定见下一行 `invoice_number` |
| `invoice_number` | `invoices.invoice_number` | 原样（人眼可读票号，如 `INV-1005`）；无则空串 | **`invoice_number_in_reference`/`invoice_number_fuzzy_hit` 特征输入**：与 `reference` 比对；`invoice_id` 是 UUID，**不能**用于此匹配（历史 bug 已修，见文末"变更记录"） |
| `note` | —（Recon 发票无备注字段） | `""` | **特征缺口**：`note_len_norm` 恒 0（一期可接受，登记） |
| `fee` | —（Recon 无手续费字段） | `0.0` | **特征缺口**：`has_fee` 恒 0；差异类型（FX/SHORT_PAY）在 matching results 侧，不在请求输入（见 §6 扩展） |
| `counterparty_match_rate_hist` | worker 计算（PIT） | 该 `payer_id` 在 `pay_date` **之前**的付款中，有胜出对（CONFIRMED/AUTO_MATCHED）的占比；无历史 → 0.5 | 与 `features.py` None→0.5 缺省一致 |
| `is_first_counterparty` | worker 计算（PIT） | `pay_date` 之前无该客户任何付款 | |
| `is_multi_mapping` | worker 计算 | 该付款在 matching results 中关联 ≥2 张发票（实测存在：Zeta 双发票） | 策略硬规则 → REVIEW |
| `has_split_merge` | —（Recon 无拆合信号） | `false` | **特征缺口**：一期靠多对多 flag 兜底，登记 |

## 4. 请求 / 响应 / 日志格式

### 4.1 请求（worker → decision-service）

`POST /v1/decisions`，body = §3 映射后的 DecideRequest（JSON）。示例（真实 seed 数据）：

```json
{
  "request_id": "shadow-8e5ea0e4…-5350d075…-9be47d70…",
  "tenant_id": "8e5ea0e4-8651-4bd6-8411-e47b3fe2d645",
  "payment_id": "5350d075-2d30-4778-924e-58c5e55f7167",
  "invoice_id": "9be47d70-3425-45cb-b699-d15896fe5386",
  "invoice_number": "INV-1005",
  "pay_amount": 500.0, "inv_amount": 500.0,
  "pay_currency": "USD", "inv_currency": "EUR",
  "pay_date": "2026-09-14", "inv_date": "2026-08-28",
  "payer_id": "ba2ec591-5ef7-4f9b-b05c-235ff7bd32f9",
  "supplier_id": "ba2ec591-5ef7-4f9b-b05c-235ff7bd32f9",
  "supplier_name": "Epsilon Company",
  "channel": "BANK_TRANSFER", "reference": "INV-1005", "note": "", "fee": 0.0,
  "counterparty_match_rate_hist": 0.5, "is_first_counterparty": true,
  "is_multi_mapping": false, "has_split_merge": false
}
```

### 4.2 响应（决策服务 → worker，已实现）

```json
{
  "decision_id": "DEC-…", "request_id": "shadow-…", "tenant_id": "…",
  "action": "REVIEW",            // Shadow 期对外恒为掩码后动作（AUTO→REVIEW）
  "model_action": "AUTO",        // 真实决策（worker 记入对照日志——推翻率计算用）
  "mode": "shadow", "is_shadow": true,
  "model": {"label": "MATCHED", "conf": 0.98, "probs": {"MATCHED": 0.98, "PARTIAL": 0.01, "UNMATCHED": 0.01}},
  "policy": {"rule": "auto_matched_high_conf", "reason": "…", "version": "1.0"},
  "model_version": "recon-v1:…", "feature_version": "1.1", "latency_ms": 2, "duplicate": false
}
```

> 说明：§4.1 示例是 USD/EUR 跨币种对，真实路由会被 `struct_currency_mismatch` 硬规则判 **REVIEW**；
> §4.2 则是**另一条**"高置信干净对"的通用 AUTO 示例（二者刻意不是同一样本）。§7 联调请分别构造对应样例。

错误码（worker 行为）：

| 码 | 场景 | worker 行为 |
|---|---|---|
| 409 | 同 request_id 重放 | 幂等：采用返回的首次结果，不重复计 |
| 400 | Schema 校验失败 | 该对进异常池，计数监控，不重试 |
| 503 | 决策服务未就绪/推理失败 | **整批跳过**（Recon 零影响），记 skip 日志，下轮重试 |

### 4.3 对照日志（worker 侧，JSONL/Postgres，W4 报告的数据源）

```json
{
  "request_id": "shadow-…", "payment_id": "…", "invoice_id": "…",
  "model_action": "AUTO", "model_label": "MATCHED",
  "probs": {"MATCHED": 0.98, "PARTIAL": 0.01, "UNMATCHED": 0.01},
  "policy_rule": "auto_matched_high_conf",
  "model_version": "…", "policy_version": "1.0", "feature_version": "1.1",
  "latency_ms": 2, "decided_at": "…",
  "human_truth": null,          // 回填前为 null
  "truth_filled_at": null
}
```

human truth 回填（轮询 `/matching/results`，按 payment_id+invoice_id join）：

| Recon 最终状态 | human_label | human_action | 说明 |
|---|---|---|---|
| CONFIRMED（`confirmed_by` 人工） | 按 difference_type：NONE→MATCHED，FX/SHORT_PAY/…→PARTIAL | AUTO（人工放行自动核销口径） | gold |
| CONFIRMED 但纠正到另一发票 | 该对 → UNMATCHED（人工否掉此候选） | REVIEW | 推翻样本 |
| 保持 PENDING_REVIEW 超 SLA（24h，D5 §4.5） | 未回填 | — | 不计入分母（ShadowReporter 已实现该语义） |
| 进 exceptions（UNKNOWN_CUSTOMER 等） | 未回填 | REVIEW | 计入"人工未放行"侧 |

**推翻判定**（`ShadowReporter`，已实现）：`model_action==AUTO` 且（`human_action!=AUTO` 或 `human_label!=model_label`）→ 推翻。

## 5. 运行参数

| 项 | 值 | 依据 |
|---|---|---|
| 轮询间隔 | 5 min（新对水位线 = 上次 `reconciled_at`/`created_at` 最大值） | 平衡延迟与 API 压力 |
| 单对决策超时 | 3 s（决策服务侧 P95 目标 ≤100ms，实测 2–3ms） | G-Shadow P95≤100ms |
| `/matching/results` 客户端超时 | 120 s（**实测该端点可 30s+ 才返回**，90s 成功） | 2026-11-09 实测 |
| 批量 | 单轮 ≤200 对，批内串行（决策服务单 worker 设计） | 容量保护 |
| 重试 | 503 整批跳过下轮再来；网络错误指数退避 3 次 | |
| Shadow 观察量 | **≥10,000 对 或 ≥2 周（取先到）**，须来自真实交易租户 | G-Shadow 门槛 |
| 监控 | worker 心跳、决策成功率、409/400/503 计数、对照日志积压（truth 回填率） | 定稿 §4 监控集 |

## 6. 需 Recon 团队确认/支持的事项（冻结前必须闭环）

| # | 事项 | 建议 | 状态 |
|---|---|---|---|
| R1 | 专用 **shadow 只读服务账号**（MEMBER 角色即可，API 消费） | 独立账号、凭证入密钥管理、审计可区分（actor_email 前缀 `shadow-`） | ⏳ 待确认 |
| R2 | Shadow 流量租户选择 + 授权记录（盘点 1.7 口径） | 选月交易量最大的生产租户；授权文档编号入契约附录 | ⏳ 待确认 |
| R3 | 轮询 vs 推送：Recon 是否有 webhook/事件出口可替代轮询 | 探查未见 webhook 端点；阶段一用轮询，若有事件出口优先 | ⏳ 待确认 |
| R4 | 扩展输入（可选，提升特征完整度）：请求随附 `difference_type` / 真实手续费 / **`invoice_number`** | **`fee` 保持 0**（仅当有可验证手续费才填非零）——差异类型不得塞进 `fee`，否则会误触发 `has_fee` 特征：FX 走既有 `currency_mismatch`，SHORT_PAY 走 PARTIAL 标签口径；**票号特征依赖 `invoices.invoice_number`**（人眼可读号，非 UUID `invoice_id`），阶段一 worker 从 invoices 侧带入、阶段二内嵌调用请随 matching result 一并传出 `invoice_number` | ⏳ 待确认 |
| R5 | `/matching/results` 慢查询（30s+）原因与 SLA | Shadow 期可接受（轮询异步）；灰度期内嵌调用时**必须**给出 P95 SLA（≤200ms 目标） | ⏳ 待确认 |
| R6 | 阶段二内嵌调用的开发排期（W4 灰度前） | 按 §7-B 兜底：排期不到位 → 灰度入口用"worker 近实时轮询（30s）"顶替，报告标注 | ⏳ 待排期 |

## 7. W3 联调验收清单（契约冻结后执行）

- [ ] worker 以 R1 账号登录 Recon，只读拉取五个端点成功（audit 日志中无写操作记录——**零改造证据**）
- [ ] 全链路：PENDING_REVIEW 新对 → DecideRequest 构建 → `POST /v1/decisions` → 对照日志落盘（四版本齐全、`is_shadow=true`）
- [ ] 幂等：同一对重复轮询 → 409 首次结果，对照日志不重复
- [ ] 503：决策服务停止 → worker 整批跳过、Recon 侧无任何变化（audit 无新增）
- [ ] truth 回填：人工确认一笔后，对照日志 `human_truth` 正确回填，`ShadowReporter` 推翻率计算正确（含 0/1 边界）
- [ ] 延迟：worker 单对端到端（拉取→决策→落日志）P95 记录在案；决策服务侧 P95 ≤100ms
- [ ] 安全：凭证不出现在 worker 日志/对照日志；tenant 隔离（只处理 R2 租户）

## 8. 与既有实现的关系（已具备，无需新开发的部分）

| 契约要求 | 现有实现 | 状态 |
|---|---|---|
| 决策入口 + 幂等 + 四版本 + 503 全停 | `service/app.py`（`/v1/decisions`、409、`latency_ms`、`input_digest`） | ✅ 已实现并 E2E 验证 |
| Shadow 掩码（AUTO→对外 REVIEW，落库记真实） | `decision_service.decide()` + `shadow.py`，11 例单测覆盖 | ✅ |
| 推翻率计算（含"未回填不计分母"语义） | `ShadowReporter`，0/1 边界单测 | ✅ |
| 策略硬规则（多对多/币种不符/主体未解析→REVIEW） | `policies/recon/engine.yaml` | ✅（`UNK-*` 主体走 `supplier_exact_match=0` → REVIEW） |
| 对照日志持久化（跨重启） | `PostgresStore` / `JsonlStore`（`decision_logs` + `mode='shadow'`） | ✅ 8 例集成测试 |
| 数据转储/字段映射参考实现 | `training/data/recon_api_source.py`（阶段一 worker 的映射逻辑可直接复用） | ✅ |
| **shadow-worker 进程本体**（轮询+水位线+truth 回填调度） | 需新开发（约 1 人周，映射逻辑复用 recon_api_source） | ⏳ W3 交付 |

---

## 附：冻结流程

1. 本草案发 Recon 团队 + 业务负责人（D5 会后 T+3 前）；
2. §六 R1–R6 逐项闭环（书面确认）；
3. 版本号升 v1.0、双方签字 → 进入 W3 联调（只实现不讨论，定稿 §7-B）；
4. 任何变更走 §7-H 审批流（策略/契约同仓库评审，24h 内归档）。

---

## 变更记录（冻结前，代码走查发现的契约↔实现不一致已就地修正）

- **v0.1 补丁 ①（票号特征）**：`invoice_number_in_reference`/`fuzzy` 原实现误用 UUID `invoice_id` 比对 `reference`，对真实数据恒为 0（合成/单测因 `invoice_id` 恰为人类可读号而掩盖）。已改为消费新增的 `invoice_number` 字段（`DecideRequest`/`PairInput`），`FEATURE_VERSION` 1.0 → **1.1**。§3 已加 `invoice_number` 映射行、R4 已并入该扩展请求。
- **v0.1 补丁 ②（错误码）**：`/decide` schema 校验失败此前返回 FastAPI 默认 **422**，与本契约 §4.2 及 README 约定的 **400 SCHEMA_INVALID** 不符。已在 `app.py` 加 `RequestValidationError` 处理器统一发 400；worker 按 §4.2 表路由异常池成立。
- **v0.1 补丁 ③（fee 语义）**：R4 原写法把 `difference_type` 映射进 `fee`，会误触发 `has_fee`。已改为 `fee` 保持 0，FX 走 `currency_mismatch`、SHORT_PAY 走 PARTIAL 标签。

> 遗留（待 Recon 走查会一并定调）：§4.3 推翻率 join 口径 —— 契约写 `payment_id+invoice_id`，
> 现 `ShadowReporter` 仅按 `payment_id`（因多对多恒被硬规则路由 REVIEW 而未暴雷）。冻结前二选一：
> 代码改元组键，或本文注明"多对多→REVIEW 故 payment_id 足够"。
