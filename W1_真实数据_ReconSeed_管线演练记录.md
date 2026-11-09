# W1 真实数据（Recon seed）管线演练记录

> 执行时间：2026-11-09
> 数据源：**Aivora Recon 生产系统**（`https://recon.ivlacloud.com`，服务名 `ivora-reconciliation`）
> 工作区：**"Ivora Seed Demo"**（tenant 8e5ea0e4…，ENTERPRISE 套餐，2026-10-04 创建，用户 Seed Admin/OWNER）
>
> ## ⚠️ 红线（定稿 §1-D2，全文适用）
>
> 该工作区为 **seed 演示数据**（`external_txn_id=SEED-P*`、租户名 "Ivora Seed Demo"）：
> 6 付款 / 7 发票 / 6 客户 / 6 匹配结果 / 1 异常。**只用于验证"真实 API → 我方管线"的
> 适配与链路正确性，不得进入 G-数据门禁、不得用于 D5 go/no-go、不得用于模型验收。**
> 真实验收需生产租户数据（≥3,000 笔、≥3 个月、≥10 主体群）。

---

## 一、系统形态确认（对 D5/方案的实质修正）

| 项 | 方案假设 | 实测 | 处置 |
|---|---|---|---|
| 业务方向 | AP/AR 待 D5 决策 4 确认 | **AR**（客户银行流水 ↔ 应收发票） | D5 决策 4 落 AR；盘点 SQL 加 `<doc_type>` 占位符 |
| 初筛候选 | 未明确 | `GET /matching/results` 每行 = 一个 (付款,候选发票) 对，含 match_type/score/difference/reason/status | §7-B Shadow 契约的输入面已明确 |
| 现有规则基线 | 抽象描述 | 3 类确定性规则实测：参考号匹配（"Deterministic reference match"，score 1.0）、客户+金额精确（"Customer + exact outstanding amount"，1.0）、歧义候选（"Ambiguous: several invoices match…"，0.6–0.9 → PENDING_REVIEW） | W2 规则基线对照目标有了真实语义；**歧义候选正是决策引擎目标人群** |
| 多对多 | 结构 flag 假设 | 真实存在：付款 5cb03b13 → INV-1006 + INV-1007 两条结果（1 张被 CONFIRMED，1 张 PENDING_REVIEW） | `is_multi_mapping` 有真实形态 |
| 币种差异 | 结构 flag 假设 | 真实存在：5350d075(USD) ↔ INV-1005(EUR)，difference_type=FX | `currency_mismatch` 有真实形态 |
| 异常池 | 未明确 | `GET /matching/exceptions`：UNKNOWN_CUSTOMER（"Mystery Payer Ltd" 无法解析到客户） | 主体未解析 = UNK-* 假名，进主体存疑处理 |
| 数据导出通道 | 未明确 | `GET /matching/export` → 17 列 CSV（含 reason/score/diff） | 大批量抽取时的生产通道（比逐条 API 高效） |
| 延迟 | 未明确 | `/matching/results` 首探 30s 超时、90s 成功 | Shadow 联调延迟预算须考虑 |

## 二、执行路径（全部真实跑通）

```
ReconApiSource（新增 training/data/recon_api_source.py，只读：login + 5 个 GET）
  → 4 份原始 CSV（与 simulate_w1.py 同 schema，含 recon_results 的 role/is_correct/final_label/confirmed_by）
  → extract.py --source csv（§7-C 管线零改动复用）
      清洗 0（真实数据干净）/ 脱敏假名化 / 质检 PASS
      data_snapshot_hash = 58003b2b7f97e2d3…
  → 盘点 SQL 11/11 在真实 schema 上执行通过（sim_inventory_duckdb.py --doc-type AR）
  → verify_real_link.py：6 个真实对 → PairInput → 26 维特征 → 校准推理 → top-1
```

**实测 A.1（doc_type=AR）**：total_pairs 6 = MATCHED 3 / PARTIAL 2 / UNMATCHED 1；
标签来源 human_review 1 / rule_inferred 2 / 未确认 3（PENDING_REVIEW）；
gold 池 = 1（CONFIRMED 行）——seed 规模下 gold 远小于门槛，符合预期。

**推理链路演示**（工件=合成自检模型，仅证链路）：
- 部分付款对（2600 vs 3000）→ p(PARTIAL)=0.691 正确命中
- 币种不符对（USD/EUR）→ 模型 p(MATCHED)=1.0，但 `currency_mismatch=1` → **策略硬规则路由 REVIEW**
- 多对多付款 → 模型 p(MATCHED)=1.0，但 `is_multi_mapping=1` → **策略硬规则路由 REVIEW**
  （后两例正是"模型预测与业务动作分离"设计意图的真实数据演示）

## 三、标签映射（演练默认，**须标签工作坊确认**）

| Recon 状态 | final_label | confirmed_by | 说明 |
|---|---|---|---|
| CONFIRMED + EXACT + NONE | MATCHED | HUMAN | 人工确认的胜出对（gold） |
| AUTO_MATCHED + EXACT + NONE | MATCHED | RULE_AUTO | 系统自动（仅 baseline，定稿 §3-8） |
| 任一 match_type=PARTIAL 或 difference∈{FX,SHORT_PAY,OVER_PAY,FEE,ROUNDING,TAX} | PARTIAL | 按状态 | 可解释差异 |
| 同付款未胜出对（EXACT+NONE 但 status=PENDING_REVIEW） | UNMATCHED | NONE | **假设**：干扰候选；歧义对（如 Zeta 双发票）的真实状态是"业务裁决"，工作坊定口径 |
| 无结果行付款 | —（不产对） | —（unlabeled） | 如 Mystery Payer（EXCEPTION） |

## 四、产出物（均在 gitignored `training/out/` 下）

| 文件 | 说明 |
|---|---|
| `recon-seed/{payments,invoices,recon_results,counterparty_stats}.csv` | 真实 API 转储（UUID 原始 id，未脱敏——仅本地） |
| `recon-seed/masked/` | 脱敏产物 + snapshot.json（hash 58003b2b…） |
| `recon-seed/inventory.json` | 盘点 SQL 11 条结果 |
| `_recon_probe/` | API 探查缓存（bundle.js、各端点 JSON、export CSV） |

## 五、下一步

1. **D5 决策 4 落 AR**；盘点 SQL 模板已参数化 `<doc_type>`；
2. **§7-B Shadow 契约草案**（输入 = `/matching/results` 的 PENDING_REVIEW 集；
   输出 = 决策服务 AUTO/REVIEW 建议 + 四版本；回写 = 只读 shadow 日志先行，
   `PATCH /matching/{id}/{action}` 留到灰度）——可起草给 Recon 团队；
3. **要真实数据**：向业务要一个有交易量的租户（或大 seed 灌入）+ 授权记录（盘点 1.7），
   按本记录的路径直接换租户重跑（`recon_api_source.py --out …` 一行命令）；
4. 标签映射（§三）带入标签工作坊逐条确认/修改。
