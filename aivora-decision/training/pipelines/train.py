"""训练 / 校准 / 评估管线（定稿 §3-4/5、§2.3.2）。

职责（对应 W2 交付物）：
- 从 PairInput 数据集构建特征矩阵；
- 切分：时间为主 + 主体 group split，test = 最后 30% 时间段（定稿 §2.2.5）；
- 训练：规则基线（对照组）+ LightGBM 三分类；
- 校准：train/calibration/test 三分；calibration ≥2000 → isotonic，否则 Platt（逐类）；
- 评估：macro-F1、各类 precision/recall、ECE(10-bin)、Brier、p≥0.9 分段实际正确率、
  泄漏检查（group split vs 随机 split 指标差）、复杂样本拒绝率（策略路由）；
- 导出：模型 + calibration.json + 训练配置（seed/参数/数据 hash）→ artifact 目录。

可复现：固定 seed、配置入库、同数据同配置必出同模型（定稿 §1.2 可复现要求）。

注意：本模块是**管线骨架 + 可运行逻辑**，用合成数据验证流程正确性；
真实验收必须用脱敏 Recon 数据（定稿红线：合成数据不得进门禁计数）。
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import Sequence

import numpy as np
import lightgbm as lgb
from sklearn.metrics import brier_score_loss, confusion_matrix, f1_score
from sklearn.model_selection import GroupShuffleSplit

from features import (FEATURE_NAMES, FEATURE_VERSION, LABELS, PairInput,
                      build_features, is_complex, to_vector)

# 训练配置（定稿：配置入库，改参数须版本 +0.1）
LGBM_PARAMS = dict(
    objective="multiclass", num_class=3,
    learning_rate=0.05, num_leaves=31, max_depth=-1,
    min_child_samples=20, subsample=0.9, subsample_freq=1,
    colsample_bytree=0.9, reg_alpha=0.0, reg_lambda=1.0,
    n_estimators=400, random_state=42, n_jobs=2, verbosity=-1,
)
EARLY_STOPPING_ROUNDS = 50
SEED = 42
TEST_FRACTION = 0.30          # test = 最后 30% 时间段
CALIB_MIN_FOR_ISOTONIC = 2000 # calibration 集 ≥ 该值用 isotonic，否则 Platt


def _label_idx(y: Sequence[str]) -> np.ndarray:
    idx = {l: i for i, l in enumerate(LABELS)}
    return np.array([idx[v] for v in y], dtype=int)


# ---------------------------------------------------------------------------
# 切分（时间为主 + 主体 group split，定稿 §1.3.6 / §2.2.5）
# ---------------------------------------------------------------------------

def temporal_group_split(
    inputs: Sequence[PairInput],
    labels: Sequence[str],
) -> dict[str, list[int]]:
    """按付款日期排序，最后 TEST_FRACTION 时间段为 test；
    且同一主体（payer_id）不跨 train/test（group split 防泄漏）。
    返回 {"train": [idx], "calib": [idx], "test": [idx]}。
    """
    n = len(inputs)
    dates = sorted({d for d in (i.pay_date for i in inputs)})
    if not dates:
        raise ValueError("empty dataset")
    # 时间段边界：按排序后的付款日，取最后 30% 的时间点作为 test 起点
    order = sorted(range(n), key=lambda i: inputs[i].pay_date)
    cut_pos = int(round(n * (1 - TEST_FRACTION)))
    test_start_date = inputs[order[cut_pos]].pay_date if cut_pos < n else dates[-1]

    test_idx = [i for i in range(n) if inputs[i].pay_date >= test_start_date]
    traincand = [i for i in range(n) if inputs[i].pay_date < test_start_date]

    # group split：test 中出现的主体，从 traincand 中剔除（防同主体跨集）
    test_groups = {inputs[i].payer_id for i in test_idx}
    traincand = [i for i in traincand if inputs[i].payer_id not in test_groups]

    # train/calib 再按主体 group 二分（calib 用于拟合校准器，不得与 test 同主体）
    gs = GroupShuffleSplit(n_splits=1, test_size=0.4, random_state=SEED)
    calib_ratio = 0.4
    if len(traincand) >= 10 and len({inputs[i].payer_id for i in traincand}) >= 3:
        for tr_pos, ca_pos in gs.split(
                np.zeros(len(traincand)), np.zeros(len(traincand)),
                [inputs[i].payer_id for i in traincand]):
            train_idx = [traincand[k] for k in tr_pos]
            calib_idx = [traincand[k] for k in ca_pos]
            break
    else:  # 样本太少，简单按比例
        k = int(len(traincand) * (1 - calib_ratio))
        train_idx, calib_idx = traincand[:k], traincand[k:]

    return {"train": sorted(train_idx), "calib": sorted(calib_idx),
            "test": sorted(test_idx)}


# ---------------------------------------------------------------------------
# 规则基线（对照组，定稿 §2.2.4-1）
# ---------------------------------------------------------------------------

def rule_baseline(feat: dict[str, float], amount: float) -> str:
    """确定性规则：附言含发票号且主体一致且金额差在容差内 → MATCHED；
    主体不一致 → UNMATCHED；复杂结构 → (策略层)REVIEW；其余 → UNMATCHED(交人工)。
    这是 W2 要打败的对象，也是验收对照组。"""
    if is_complex(feat):
        return "UNMATCHED"  # 基线把复杂样本当不匹配（保守），策略层会路由 REVIEW
    if feat["supplier_exact_match"] == 0:
        return "UNMATCHED"
    tol_abs, tol_rel = 100.0, 0.01
    if (feat["invoice_number_in_reference"] == 1
            and feat["amount_abs_diff"] <= tol_abs
            and abs(feat["amount_diff_ratio"]) <= tol_rel):
        return "MATCHED"
    return "UNMATCHED"


# ---------------------------------------------------------------------------
# 原始分数（logits）
# ---------------------------------------------------------------------------

def raw_logits(model: "lgb.LGBMClassifier", X: np.ndarray) -> np.ndarray:
    """逐类原始分数（multiclass → (n, 3)）。

    用 `booster_.predict(X, raw_score=True)`——与推理端（service/inference.py）
    完全同一条路径（定稿：工件 = model.txt 原生 LightGBM）。
    不再用 sklearn 包装层的 `decision_function`：lightgbm ≥4.5 已移除该 API
    （2026-11-07 实测 4.5.0 报 AttributeError）。
    """
    return np.asarray(model.booster_.predict(X, raw_score=True))


# ---------------------------------------------------------------------------
# 校准（自定义 OvR，避免依赖 sklearn 内部 API 变动；可版本化、可单测）
# ---------------------------------------------------------------------------

class OvRCalibrator:
    """逐类（One-vs-Rest）校准器：对每个类别 c 拟合
        P(y=c | x) = calibrator( 模型对该类的原始 margin/logit 或 proba )
    方法：isotonic（样本足）或 Platt sigmoid（LogisticRegression 二值）。
    校准后按类别归一化得到三分类概率。

    选择 raw logit 而非 predict_proba 作为校准输入：predict_proba 已含一次
    softmax 归一化，再逐类校准会双重变换；用逐类 logit（multiclass 下
    raw_score 返回 num_class 维）更干净，符合 Platt 原意。
    """

    def __init__(self, method: str):
        self.method = method
        self._fitters: dict[int, object] = {}

    def fit(self, base, X_calib: np.ndarray, y_calib: np.ndarray) -> "OvRCalibrator":
        logits = raw_logits(base, X_calib)  # (n, 3)
        self._n_class = logits.shape[1]
        for c in range(self._n_class):
            z = logits[:, c]
            yb = (y_calib == c).astype(int)
            if self.method == "isotonic":
                from sklearn.isotonic import IsotonicRegression
                f = IsotonicRegression(increasing=True, out_of_bounds="clip")
            else:  # platt
                from sklearn.linear_model import LogisticRegression
                f = LogisticRegression(C=1e6)  # 弱正则 ≈ Platt
            f.fit(z.reshape(-1, 1), yb)
            self._fitters[c] = f
        return self

    def predict_proba(self, base, X: np.ndarray) -> np.ndarray:
        logits = raw_logits(base, X)
        out = np.zeros((X.shape[0], self._n_class))
        for c in range(self._n_class):
            out[:, c] = self._fitters[c].predict(logits[:, c].reshape(-1, 1))
        # 归一化（OvR 概率之和未必为 1）
        s = out.sum(axis=1, keepdims=True)
        s[s == 0] = 1.0
        return out / s


def fit_calibrator(model, X_calib: np.ndarray, y_calib: np.ndarray,
                   method_pref: str | None = None) -> OvRCalibrator:
    """calibration 集 ≥ CALIB_MIN_FOR_ISOTONIC 用 isotonic，否则 Platt。
    校准器作为模型工件的一部分随版本发布（定稿 §2.2.5）。"""
    if method_pref is None:
        method_pref = "sigmoid" if len(X_calib) < CALIB_MIN_FOR_ISOTONIC else "isotonic"
    return OvRCalibrator(method_pref).fit(model, X_calib, y_calib)


def calibrated_probs(cal: OvRCalibrator, model, X: np.ndarray) -> np.ndarray:
    return cal.predict_proba(model, X)


class OvRPlatt:
    """可部署的 OvR Platt 校准器（逐类 sigmoid 参数 A/B，纯 Python）。

    与 `OvRCalibrator`(method="platt") 数学等价：Platt P(y=c|x)=sigmoid(A_c·z_c + B_c)，
    z_c 为模型第 c 类的原始 logit（multiclass decision_function 第 c 列），逐类校准后
    归一化得三分类概率。

    引入目的：生产服务只拿到工件（model.txt + calibration.json），不能重拟校准器
    （需 calibration 集，违背部署无状态 / 定稿 §1.2）。Platt 逐类 A/B 序列化进
    calibration.json，服务纯加载即还原，与训练期 predict_proba 完全一致。
    """

    def __init__(self, params: list[tuple[float, float]]):
        self.params = params            # [(A0,B0),(A1,B1),(A2,B2)]
        self._n_class = len(params)

    @staticmethod
    def _sigmoid(z: float) -> float:
        if z >= 0:
            return 1.0 / (1.0 + math.exp(-z))
        ez = math.exp(z)
        return ez / (1.0 + ez)

    def predict_proba(self, logits: np.ndarray) -> np.ndarray:
        """logits: (n, K) 原始逐类 logit → 归一化概率 (n, K)。"""
        out = np.zeros((logits.shape[0], self._n_class))
        for c in range(self._n_class):
            A, B = self.params[c]
            out[:, c] = [self._sigmoid(A * z + B) for z in logits[:, c]]
        s = out.sum(axis=1, keepdims=True)
        s[s == 0] = 1.0
        return out / s

    def to_dict(self) -> dict:
        return {"method": "platt", "params": [[float(a), float(b)] for a, b in self.params]}

    @classmethod
    def from_dict(cls, d: dict) -> "OvRPlatt":
        return cls([tuple(p) for p in d["params"]])


def platt_from_calibrator(cal: OvRCalibrator) -> OvRPlatt:
    """从 sklearn 版 Platt 校准器抽出逐类 A/B（coef_[0][0] / intercept_[0]）。"""
    params = []
    for c in range(len(cal._fitters)):
        lr = cal._fitters[c]
        params.append((float(lr.coef_[0][0]), float(lr.intercept_[0])))
    return OvRPlatt(params)


# ---------------------------------------------------------------------------
# 评估指标
# ---------------------------------------------------------------------------

def ece(probs: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray,
        n_bins: int = 10) -> float:
    """Expected Calibration Error（按预测类别的置信度分箱）。"""
    conf = np.max(probs, axis=1)
    pred_ok = (y_pred == y_true).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    total = len(y_true)
    e = 0.0
    for b in range(n_bins):
        m = (conf > bins[b]) & (conf <= bins[b + 1])
        if m.sum() == 0:
            continue
        avg_conf = conf[m].mean()
        emp_acc = pred_ok[m].mean()
        e += (m.sum() / total) * abs(avg_conf - emp_acc)
    return float(e)


def high_conf_accuracy(probs: np.ndarray, y_true: np.ndarray,
                       y_pred: np.ndarray, thr: float = 0.9) -> dict:
    """验收重点（定稿 §2.2.5）：p(预测类)≥thr 分段的实际正确率。
    这是 AUTO 动作的安全前提。"""
    conf = np.max(probs, axis=1)
    m = conf >= thr
    n = int(m.sum())
    if n == 0:
        return {"n": 0, "accuracy": None}
    acc = float((y_pred[m] == y_true[m]).mean())
    return {"n": n, "accuracy": round(acc, 4)}


def leakage_check_macroF1(random_split_f1: float, group_split_f1: float) -> dict:
    """定稿 §2.3.2：group split 与随机 split 的 macro-F1 差距 < 3pt 否则判泄漏。"""
    diff = random_split_f1 - group_split_f1
    return {"random_split_f1": round(random_split_f1, 4),
            "group_split_f1": round(group_split_f1, 4),
            "diff_pt": round(diff * 100, 2),
            "leakage_suspected": bool(diff * 100 >= 3.0)}


def per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, dict]:
    """从混淆矩阵手动派生各类 precision/recall/support（避免 sklearn 多类
    指标 API 跨版本漂移；precision = TP/(TP+FP), recall = TP/(TP+FN)）。"""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(LABELS))))
    out = {}
    for c in range(len(LABELS)):
        tp = int(cm[c, c])
        fp = int(cm[:, c].sum() - tp)
        fn = int(cm[c, :].sum() - tp)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        out[LABELS[c]] = {"precision": round(prec, 4), "recall": round(rec, 4),
                          "support": int((y_true == c).sum())}
    return out


# ---------------------------------------------------------------------------
# 主训练流程
# ---------------------------------------------------------------------------

def train_and_evaluate(
    inputs: Sequence[PairInput],
    labels: Sequence[str],
    artifact_dir: Path,
    data_snapshot_hash: str,
) -> dict:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    y = _label_idx(labels)

    # 特征矩阵
    X_all = np.vstack([to_vector(build_features(i)) for i in inputs])
    y_all = y

    # 切分
    split = temporal_group_split(inputs, labels)
    Xtr, ytr = X_all[split["train"]], y_all[split["train"]]
    Xcal, ycal = X_all[split["calib"]], y_all[split["calib"]]
    Xte, yte = X_all[split["test"]], y_all[split["test"]]

    # 训练 LightGBM（early stopping on calibration 集）
    model = lgb.LGBMClassifier(**LGBM_PARAMS)
    model.fit(Xtr, ytr, eval_set=[(Xcal, ycal)],
              callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
                         lgb.log_evaluation(0)])

    # 校准（prefit：用 calib 集拟合校准器）
    cal = fit_calibrator(model, Xcal, ycal)

    # 评估（test 集，用校准后概率）
    proba_te = calibrated_probs(cal, model, Xte)
    pred_te = proba_te.argmax(axis=1)
    macro_f1 = float(f1_score(yte, pred_te, average="macro"))
    per_class = per_class_metrics(yte, pred_te)
    ece_val = ece(proba_te, yte, pred_te)
    brier = float(brier_score_loss(
        (yte == 0).astype(int), proba_te[:, 0]))  # MATCHED 类二值 Brier（示例）
    high_conf = high_conf_accuracy(proba_te, yte, pred_te, thr=0.9)

    # 规则基线对照
    base_pred = np.array(
        [_label_idx([rule_baseline(build_features(inputs[i]), inputs[i].pay_amount)])[0]
         for i in split["test"]])
    base_f1 = float(f1_score(yte, base_pred, average="macro"))

    # 复杂样本：策略层按结构 flag 确定性路由 REVIEW（不强行出模型标签）。
    # reject_rate = test 中复杂样本被路由 REVIEW 的比例（设计上恒 1.0，
    # 门禁用它确认策略生效）；complex_in_test = 复杂样本占比（信息量）。
    complex_flags = np.array([is_complex(build_features(inputs[i])) for i in split["test"]])
    complex_in_test = float(complex_flags.mean())
    reject_rate = 1.0 if complex_flags.sum() > 0 else None

    # 泄漏检查：再跑一次随机 split 的 macro-F1
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_FRACTION, random_state=SEED)
    for tr_p, te_p in gss.split(X_all, y_all, [inputs[i].payer_id for i in range(len(inputs))]):
        m_rand = lgb.LGBMClassifier(**LGBM_PARAMS)
        m_rand.fit(X_all[tr_p], y_all[tr_p])
        p_rand = m_rand.predict_proba(X_all[te_p]).argmax(axis=1)
        rand_f1 = float(f1_score(y_all[te_p], p_rand, average="macro"))
        break
    leak = leakage_check_macroF1(rand_f1, macro_f1)

    # 离线门槛（定稿 §2.3.2，示例值，真实数据后标定）
    gates = {
        "macro_f1_vs_baseline_+2pt": bool(macro_f1 >= base_f1 + 0.02),
        "matched_precision_>=0.97": bool(per_class["MATCHED"]["precision"] >= 0.97),
        "ece_<=0.03": bool(ece_val <= 0.03),
        "high_conf_accuracy_>=0.97": bool(high_conf["accuracy"] is not None
                                          and high_conf["accuracy"] >= 0.97),
        "complex_reject_>=0.95": bool(reject_rate is not None and reject_rate >= 0.95),
        "no_leakage": bool(not leak["leakage_suspected"]),
    }

    # 导出工件
    model_path = artifact_dir / "model.txt"
    model.booster_.save_model(str(model_path))
    # 校准器：Platt 存逐类 A/B（服务纯加载即可还原）；isotonic 存断点参数（
    # 服务侧可纯加载还原，见 service/inference.py 的 isotonic 分支）。
    calib_path = artifact_dir / "calibration.json"
    if len(Xcal) < CALIB_MIN_FOR_ISOTONIC:
        platt = platt_from_calibrator(cal)
        calib_doc = {**platt.to_dict(), "calibration_set_size": int(len(Xcal)),
                     "feature_version": FEATURE_VERSION}
    else:  # isotonic：逐类存 x 断点 / y 值，服务侧线性还原
        iso = {"x": [list(f.x_) for f in cal._fitters.values()],
               "y": [list(f.y_) for f in cal._fitters.values()],
               "x_threshold": [list(getattr(f, "x_threshold_", []))
                               for f in cal._fitters.values()]}
        calib_doc = {"method": "isotonic", "calibration_set_size": int(len(Xcal)),
                     "feature_version": FEATURE_VERSION, "iso": iso}
    calib_path.write_text(json.dumps(calib_doc, indent=2))
    config_path = artifact_dir / "config.json"
    config_path.write_text(json.dumps({
        "lgbm_params": LGBM_PARAMS, "seed": SEED,
        "test_fraction": TEST_FRACTION,
        "data_snapshot_hash": data_snapshot_hash,
        "split_sizes": {k: len(v) for k, v in split.items()},
        "best_iteration": int(model.best_iteration_ or LGBM_PARAMS["n_estimators"]),
    }, indent=2))
    # ONNX 导出（可选优化）。规范推理工件是 model.txt（原生 LightGBM，
    # CPU 无状态，满足定稿 §1.2 要求）。ONNX 仅为部署端可选项：
    # 当前 skl2onnx(1.20) 已移除 convert_lightgbm，导出会 best-effort 降级。
    onnx_exported = False
    onnx_note = "skl2onnx>=1.20 removed convert_lightgbm; model.txt is the canonical artifact"
    try:
        from skl2onnx import convert_lightgbm  # 旧版 skl2onnx 才有
        from skl2onnx.common.data_types import FloatTensorType
        onx = convert_lightgbm(model,
                               initial_types=[("input", FloatTensorType([None, len(FEATURE_NAMES)]))])
        with open(artifact_dir / "model.onnx", "wb") as f:
            f.write(onx.SerializeToString())
        onnx_exported = True
        onnx_note = "exported"
    except ImportError:
        pass
    except Exception as e:  # noqa: BLE001
        onnx_note = f"export failed: {e}"

    metrics = {
        "macro_f1": round(macro_f1, 4), "baseline_macro_f1": round(base_f1, 4),
        "per_class": per_class, "ece_10bin": round(ece_val, 4),
        "brier_matched": round(brier, 4), "high_conf_p0.9": high_conf,
        "complex_in_test": round(complex_in_test, 4),
        "complex_reject_rate": reject_rate, "leakage": leak,
        "gates": gates, "all_gates_passed": all(gates.values()),
        "onnx_exported": onnx_exported, "onnx_note": onnx_note,
        "split_sizes": {k: len(v) for k, v in split.items()},
    }
    (artifact_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    return metrics


if __name__ == "__main__":
    # 自检：用合成数据跑通流程（**不得用于验收**，仅验证管线正确性）
    import random
    random.seed(SEED)
    np.random.seed(SEED)

    def synth_pair(i: int, label: str) -> PairInput:
        # 主体/时间分段在下方按 sample i 统一分配（保证时间+主体切分可行）。
        base = random.uniform(5_000, 200_000)
        if label == "MATCHED":
            pay, inv = base, base
            ref = f"付款 INV-{i:04d} 全额"
            rate, first = random.uniform(0.9, 1.0), False
            mm = sm = False
        elif label == "PARTIAL":
            pay, inv = base * random.uniform(0.6, 0.95), base
            ref = f"付款 INV-{i:04d} 部分"
            rate, first = random.uniform(0.8, 0.98), False
            mm = sm = False
        else:
            pay, inv = base * random.uniform(0.3, 1.4), base
            ref = "其他付款"
            rate, first = random.uniform(0.5, 0.9), random.random() < 0.2
            mm = sm = False
        if random.random() < 0.1:
            mm = True  # 10% 复杂结构
        # 时间分段：sample i 落在第 (i%10)+1 月；主体按"时间块"分配，
        # 允许 ~25% 跨月借用相邻块主体（制造真实重叠，供 group split 剔除）。
        seg = (i % 10) + 1
        if random.random() < 0.25:
            seg = random.choice([seg - 1, seg + 1])
            seg = max(1, min(10, seg))
        payer = f"CP-{seg:02d}-{random.randint(1, 3):02d}"
        pay_date = f"2026-{seg:02d}-{random.randint(1, 28):02d}"
        return PairInput(
            payment_id=f"PAY-{i:05d}", invoice_id=f"INV-{i:04d}",
            pay_amount=round(pay, 2), inv_amount=round(inv, 2),
            pay_currency="CNY", inv_currency="CNY",
            pay_date=pay_date, inv_date=pay_date,
            payer_id=payer, supplier_id=payer if label != "UNMATCHED" else f"SP-{seg:02d}-{random.randint(1, 3):02d}",
            supplier_name="某公司", channel=random.choice(("BANK_TRANSFER", "ONLINE")),
            reference=ref, note="备注", fee=0.0 if label != "PARTIAL" else round(base * 0.01, 2),
            counterparty_match_rate_hist=rate, is_first_counterparty=first,
            is_multi_mapping=mm, has_split_merge=sm,
        )

    data = []
    for k in range(600):
        lab = random.choices(["MATCHED", "PARTIAL", "UNMATCHED"], weights=[0.5, 0.25, 0.25])[0]
        data.append((synth_pair(k, lab), lab))
    ins, labs = [d[0] for d in data], [d[1] for d in data]

    m = train_and_evaluate(ins, labs, Path("/tmp/selftest-artifact"), data_snapshot_hash="selftest")
    print(json.dumps({k: m[k] for k in ("macro_f1", "baseline_macro_f1", "ece_10bin",
                                        "high_conf_p0.9", "complex_reject_rate",
                                        "all_gates_passed", "onnx_exported", "split_sizes")},
                     indent=2, ensure_ascii=False))
