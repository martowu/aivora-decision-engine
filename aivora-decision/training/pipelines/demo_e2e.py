"""端到端演示（合成数据）：生成 → 特征 → LightGBM → 校准 → 策略路由。

目的：在**没有真实 Recon 数据**时，把整条决策链路跑通并展示"非平凡"指标：
  1. 模型三分类指标（macro-F1 / 各类 precision / p≥0.9 分段正确率）—— 真实、非 1.0；
  2. 策略路由后 AUTO / REVIEW 分布；
  3. **AUTO 动作的实际精度**（§2.2.6 核心：自动核销必须高精度，宁可 REVIEW 不可错 AUTO）；
  4. 复杂结构样本是否被策略一律拦到 REVIEW（§7-D 拒绝能力）。

红线（定稿 §1-D2）：合成数据**仅验证流程 + 演示决策链路**，指标不得进入三硬门禁计数。
真实验收必须用脱敏 Recon 数据。

运行：
  ../../.venv/bin/python demo_e2e.py [--n 1500] [--seed 42]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "policies"))

from features import LABELS, build_features, is_complex, to_vector   # noqa: E402
from train import (LGBM_PARAMS, SEED, calibrated_probs,                  # noqa: E402
                   fit_calibrator, temporal_group_split)
from synthetic import dataset_hash, generate_dataset, to_pair_inputs      # noqa: E402
from strategy import PolicyEngine, build_context                          # noqa: E402

import lightgbm as lgb

HERE = Path(__file__).parent
POLICY_YAML = HERE.parent.parent / "policies" / "recon" / "engine.yaml"
GOLDEN = HERE.parent.parent / "contracts" / "examples" / "recon.golden.json"


def run(n: int, seed: int) -> dict:
    # 1. 生成 + 转 PairInput
    recs = generate_dataset(n, seed)
    inputs, labels = to_pair_inputs(recs)
    h = dataset_hash(recs)

    # 2. 特征 + 切分
    X_all = np.vstack([to_vector(build_features(i)) for i in inputs])
    y = np.array([LABELS.index(l) for l in labels])
    split = temporal_group_split(inputs, labels)

    # 3. 训练 + 校准
    model = lgb.LGBMClassifier(**LGBM_PARAMS)
    model.fit(X_all[split["train"]], y[split["train"]],
              eval_set=[(X_all[split["calib"]], y[split["calib"]])],
              callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
    cal = fit_calibrator(model, X_all[split["calib"]], y[split["calib"]])

    # 4. test 集推理 + 策略路由
    te = split["test"]
    proba = calibrated_probs(cal, model, X_all[te])
    pred = proba.argmax(axis=1)

    eng = PolicyEngine.from_yaml(POLICY_YAML,
                                 json.loads(GOLDEN.read_text(encoding="utf-8")))

    auto_true, auto_pred, review_true = Counter(), Counter(), Counter()
    complex_auto = 0
    for k, gi in enumerate(te):
        inp = inputs[gi]
        feats = build_features(inp)
        ctx = build_context(
            features=feats,
            calib_probs=[float(p) for p in proba[k]],
            label=LABELS[pred[k]],
            payment={"amount": inp.pay_amount, "currency": inp.pay_currency,
                     "channel": inp.channel, "payer_id": inp.payer_id})
        d = eng.decide(ctx)
        truth, pr = labels[gi], LABELS[pred[k]]
        if d.action == "AUTO":
            auto_true[truth] += 1
            auto_pred[pr] += 1
            if is_complex(feats):
                complex_auto += 1
        else:
            review_true[truth] += 1

    auto_total = sum(auto_true.values())
    auto_precision = (auto_true.get("MATCHED", 0) / auto_total) if auto_total else None
    # AUTO 精度定义：被 AUTO 的样本里，真值确实是 MATCHED 的比例
    # （AUTO 动作语义 = 自动核销为 MATCHED；若真值是 PARTIAL/UNMATCHED 则核销错误）

    macro_f1 = float(_macro_f1(y[te], pred))
    return {
        "dataset_hash": h[:16], "n": n, "seed": seed,
        "label_dist": dict(Counter(labels)),
        "split_sizes": {k: len(v) for k, v in split.items()},
        "model": {
            "macro_f1_test": round(macro_f1, 4),
            "per_class": {
                LABELS[c]: _class_pr(y[te], pred, c) for c in range(3)},
            "high_conf_p0.9": _high_conf(proba, y[te], pred),
        },
        "strategy": {
            "auto_total": auto_total,
            "review_total": sum(review_true.values()),
            "auto_true_label_dist": dict(auto_true),
            "AUTO_precision_is_MATCHED": None if auto_precision is None
            else round(auto_precision, 4),
            "auto_pred_label_dist": dict(auto_pred),
            "complex_routed_to_AUTO": complex_auto,
            "review_true_label_dist": dict(review_true),
        },
        "safety": {
            "auto_high_precision": (auto_precision is not None and auto_precision >= 0.97),
            "complex_never_auto": complex_auto == 0,
        },
    }


def _macro_f1(y_true, y_pred) -> float:
    from sklearn.metrics import f1_score
    return f1_score(y_true, y_pred, average="macro")


def _class_pr(y_true, y_pred, c) -> dict:
    tp = int(np.sum((y_true == c) & (y_pred == c)))
    fp = int(np.sum((y_true != c) & (y_pred == c)))
    fn = int(np.sum((y_true == c) & (y_pred != c)))
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    return {"precision": round(prec, 4), "recall": round(rec, 4), "support": int((y_true == c).sum())}


def _high_conf(proba, y_true, y_pred, thr=0.9) -> dict:
    from train import high_conf_accuracy
    return high_conf_accuracy(proba, y_true, y_pred, thr=thr)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    res = run(a.n, a.seed)
    print(json.dumps(res, ensure_ascii=False, indent=2))
    # 红线提示
    print("\n⚠ 合成数据仅演示流程/决策链路，指标不得进入门禁计数（定稿 §1-D2）。")
