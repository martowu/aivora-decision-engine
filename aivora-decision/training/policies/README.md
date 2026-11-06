# 合成数据与端到端演示（无真实 Recon 时）

> 真实 Recon DDL 未到位前，用贴近生产结构的合成数据把整条决策链路跑通、
> 拿到**真实非平凡指标**（而非旧自检里被特征偶然编码的 1.0）。

| 文件 | 作用 |
|---|---|
| `synthetic.py` | 合成数据集生成器：主体人群（内在"干净度"）→ (付款, 候选发票, 真值) 三元组。固定 seed 可复现（同 hash）。刻意制造 MATCHED/PARTIAL/UNMATCHED 特征重叠 + 10% 复杂结构 |
| `demo_e2e.py` | 端到端：生成→特征→LightGBM→校准→**策略路由**。输出模型指标 + AUTO/REVIEW 分布 + **AUTO 精度** + 复杂结构拦截验证 |

```bash
../../.venv/bin/python synthetic.py            # 生成器自检（分布 + hash + 样例）
../../.venv/bin/python demo_e2e.py --n 1500 --seed 42   # 端到端演示
```

**最近一次 demo（seed=42, n=1500）实测**：
- 模型 macro_F1(test)=**0.9601**（真实、非 1.0）；p≥0.9 分段正确率 **0.9773**
- 策略路由：453 条 test → **158 AUTO（35%）/ 295 REVIEW**；复杂结构 **0 条漏进 AUTO**
- AUTO 精度（AUTO 样本里真值确为 MATCHED 的比例）= 1.0（策略足够保守）

**红线（定稿 §1-D2）**：合成数据仅用于验证流程 + 演示决策链路，
**指标不得进入三硬门禁计数**。真实验收必须用脱敏 Recon 数据。

与 `extract.py` 里那个 `SyntheticSource` 的区别：那个是"管线验证用"
（付款/发票互相独立随机，只验证脱敏 + hash 稳定）；`synthetic.py` 是
"训练/演示用"（有真实匹配关系 + 可学习信号 + 重叠噪声）。

# 策略引擎（W3 核心件）

> 定稿 `Aivora_Decision_Engine_最佳方案_定稿.md` §7-A（受限求值）、§2.2.1（first-match-wins）、
> §2.2.6（风险分级 AUTO 交集）。把**模型三态 + 校准概率 + 特征 flag + 付款事实**
> 路由成**业务动作**（AUTO / REVIEW）。

## 文件

| 文件 | 作用 |
|---|---|
| `strategy.py` | 受限 AST 求值器 + first-match-wins 引擎 + 上下文构建 + 黄金样例校验 |
| `../../policies/recon/engine.yaml` | 策略 v1.0（版本化，与代码同仓库同评审，§J） |
| `../../contracts/examples/recon.golden.json` | 黄金样例（启动期干跑，§7-A） |
| `test_policy.py` | 单测（注入防御 + 硬规则优先 + first-match + 默认 REVIEW + 版本化） |

## 核心设计（与定稿逐条对应）

1. **禁止裸 eval**（§7-A）：`when` 启动时解析为 AST，白名单只许
   比较 / 布尔 / `in` / 负数字面量 + 根标识符 `f.*`、`calib.*`、`payment.*`。
   拒绝一切：`__import__`、属性链双下划线逃逸、算术 `+ - * / // % **`、
   下标 `[]`、`lambda`、条件表达式、函数调用、非白名单标识符。
   → 13 个注入用例全部被拒（见 `test_injection_rejected`）。

2. **first-match-wins，硬规则优先**（§2.2.1）：规则按 YAML 顺序匹配，
   结构 flag / 主体不一致 / 金额上限排在模型分数规则**之前**，
   优先级永远高于模型分数。无命中落 `default_action: REVIEW`。

3. **REVIEW 是策略动作，不是模型标签**（§1.3.4）：模型只输出三态 + 校准概率；
   "结构复杂 / 证据不足"由特征 flag 触发策略路由 REVIEW，不污染训练标签。

4. **求值失败保守 REVIEW**（绝不静默 AUTO）：运行期缺键 / 类型不可比较 →
   该规则按求值失败处理，整体落 REVIEW，保证安全侧失败。

5. **阈值可配，默认示例值**（§2.2.6 / D5）：金额上限 100k、置信 0.9 等
   写在 `engine.yaml`，D5 决策会书面确认后改数字即可，**不改代码**。

6. **版本化可回滚**（§2.2.1 / §7-E）：`engine.snapshot()` 产出
   `decision_policies` 落库字段（policy_id / version / rules）；
   策略回滚 = 切回上一 version（比模型回滚更常用）。

7. **启动期编译 + 黄金样例干跑**（§7-A）：`PolicyEngine.from_yaml(path, golden)`
   任一规则编译失败或黄金样例 action 不符 → 抛错拒绝启动。

## 快速验证

```bash
cd training/policies
.venv 在 aivora-decision/.venv
../.venv/bin/python test_policy.py
# → 13/13 passed（含注入防御 + 硬规则优先 + 黄金样例干跑）
```

## 变量域（策略表达式可引用）

| 根 | 字段 | 来源 |
|---|---|---|
| `f.*` | `is_multi_mapping` `has_split_merge` `currency_mismatch` `supplier_exact_match` `amount_abs_diff` … | `features.build_features` |
| `calib.*` | `matched` `partial` `unmatched`（校准概率）`label`（预测三态）`conf`（最大概率） | 模型 + 校准器 |
| `payment.*` | `amount` `currency` `channel` `payer_id` | 付款事实 |

## 接入推理服务（W3 decision-service）

```python
from strategy import PolicyEngine, build_context
eng = PolicyEngine.from_yaml("policies/recon/engine.yaml", golden)  # 启动期
ctx = build_context(features, calib_probs, label, payment)
decision = eng.decide(ctx)   # → Decision(action=..., rule=..., reason=...)
decision.to_log()            # → decision_logs 落库字段
```

## 策略变更流程（§J / §7-H）

发起（业务/算法）→ 业务负责人书面确认风险数字 → 代码评审 →
版本发布（新 `policy_version`）→ 观察 48h REVIEW 率与推翻率。
紧急变更 24h 内补审批归档。
