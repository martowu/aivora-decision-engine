# W2 训练 / 校准 / 评估管线

> 定稿 `Aivora_Decision_Engine_最佳方案_定稿.md` §3-4/5、§2.2.4-5、§2.3.2 的落地。
> 首任务：`recon.transaction_match` 三分类（MATCHED / PARTIAL / UNMATCHED）+ 逐候选打分。
> **红线**：本管线用合成数据仅验证流程正确性；真实验收必须用脱敏 Recon 数据
> （合成数据不得进入任何门禁计数，定稿 §2.3.2 / R3）。

## 文件

| 文件 | 作用 |
|---|---|
| `features.py` | 特征工程 v1.0（**纯函数**、版本化、point-in-time 安全、结构 flag 透传） |
| `train.py` | 切分 + 规则基线 + LightGBM 三分类 + OvR 校准 + 全量评估 + 工件导出 |
| `test_features.py` | 特征单测（11 项：顺序/纯函数/黑名单/flag 透传/边界） |
| `verify_deploy.py` | 部署闭环验证（加载 model.txt：维度/类别数/确定性/概率合法） |

## 快速自检（合成数据，仅验证流程）

```bash
cd training/pipelines
.venv 在 aivora-decision/.venv（Python 3.14）

# 1) 特征单测
../.venv/bin/python test_features.py

# 2) 训练 + 校准 + 评估 + 导出工件
../.venv/bin/python train.py
#    → 打印宏观指标；写出 /tmp/selftest-artifact/{model.txt,calibration.json,config.json,metrics.json}

# 3) 部署闭环验证
../.venv/bin/python verify_deploy.py /tmp/selftest-artifact
```

## 模型工件（定稿 §1.2：模型 = 特征函数 + 配置 + 模型 + 校准器）

| 工件 | 内容 |
|---|---|
| `model.txt` | LightGBM 原生模型（**规范推理工件**，CPU 无状态，`lgb.Booster(model_file=...)` 即载） |
| `calibration.json` | 校准方法（sigmoid/isotonic）+ 校准集规模 + 特征版本 |
| `config.json` | lgbm 参数 + seed + test 比例 + 数据 hash + split 规模 + best_iteration |
| `metrics.json` | 全量评估指标 + 各门禁判定 + 泄漏检查 |

> **ONNX**：部署端可选项。当前 `skl2onnx>=1.20` 已移除 `convert_lightgbm`，
> 导出会 best-effort 降级并在 `metrics.json.onnx_note` 如实记录；
> `model.txt` 同样满足定稿「LightGBM(ONNX/CPU) 无状态 CPU 推理」的核心诉求。
> 若确需 ONNX，部署环境用 Python 3.10-3.12 + `skl2onnx<1.20`。

## 关键设计点（与定稿逐条对应）

1. **切分 = 时间为主 + 主体 group split**（§1.3.6/§2.2.5）
   `temporal_group_split`：最后 30% 时间段为 test；同一 `payer_id` 不跨 train/test；
   train/calib 再按主体 group 二分（calib 仅用于拟合校准器，绝不碰 test）。

2. **校准器随工件版本化**（§2.2.5）：calibration 集 ≥2000 → isotonic，否则 Platt。
   自定义 `OvRCalibrator`（逐类 logit → 二值校准 → 归一化），不依赖 sklearn
   `CalibratedClassifierCV(cv='prefit')`（该 API 在 sklearn 1.9 已移除）。

3. **规则基线是对照组**（§2.2.4-1）：`rule_baseline` 确定性规则，
   模型必须在 macro-F1 上 +2pt 才过门禁；**未超基线 = 停止条件 3**（§2.3.4）。

4. **复杂样本策略路由**（§1.3.4/§3-6）：`is_complex()` 命中结构 flag 的样本
   不强行出模型三态，由策略层路由 REVIEW；评估里 `complex_reject_rate` 确认策略生效。

5. **泄漏检查**（§2.3.2）：group split 与随机 split 的 macro-F1 差 < 3pt，
   否则判泄漏（`metrics.json.leakage`）。

6. **事后信息黑名单**（§1.3.6/§2.2.3）：`FORBIDDEN_INPUTS`，任何字段出现即抛
   `FeatureLeakageError`，从源头堵"用未来结果预测过去"。

## 验收门禁（示例值，真实数据后标定 — §2.3.2）

| 门禁 | 阈值 |
|---|---|
| 规则基线对照 | macro-F1 ≥ 基线 +2pt |
| MATCHED precision | ≥ 97% |
| ECE（10-bin） | ≤ 0.03 |
| p(预测类)≥0.9 分段实际正确率 | ≥ 97% |
| 复杂样本策略拒绝率 | ≥ 95% |
| 泄漏检查 | group vs 随机 macro-F1 差 < 3pt |

全过 → 进 W3 Shadow。任一不过且无法 1 轮内解决 → 触发停止条件（§2.3.4）。

## 接入真实数据（下一步）

`train.py` 当前 `__main__` 用合成数据自检。接真实数据时：
1. 用 `training/data/extract.py`（已落地）抽 Recon → `PairInput` 列表 + `data_snapshot_hash`；
2. 把 `inputs/labels` 与 `data_snapshot_hash` 传入 `train_and_evaluate()`；
3. 校准/评估/门禁逻辑不变——**同一套代码跑合成（验证流程）和真实（验收）**。
