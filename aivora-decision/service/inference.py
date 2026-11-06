"""推理端：纯加载工件（model.txt + calibration.json）还原校准概率。

与训练端（training/pipelines/train.py）对齐的命门：
- LightGBM 原始 logit 用 booster.predict(X, raw_score=True) 取，
  等价于训练端 model.decision_function(X)（已实测 50/50 决策一致，diff<1e-4）；
- Platt 校准：逐类 sigmoid(A_c·z_c + B_c)，A/B 从 calibration.json 直接读，
  与 train.OvRPlatt 数学一致；
- Isotonic 校准：逐类用 calibration.json 存的断点 (x_, y_) 做分段线性插值，
  超出断点范围取端点值（= sklearn out_of_bounds="clip"）。

服务**无状态**：不依赖 calibration 集，不重拟校准器（定稿 §1.2 部署无状态）。
加载失败抛 ArtifactLoadError → app 层映射 503 全停。
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import lightgbm as lgb

# 复用版本化特征清单（FEATURE_NAMES）与特征函数
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "training" / "pipelines"))
from features import (FEATURE_NAMES, LABELS, PairInput, build_features,  # noqa: E402
                      to_vector)


class ArtifactLoadError(RuntimeError):
    """工件缺失 / 格式错误 / 特征维不符。app 层 → 503 全停。"""


@dataclass(frozen=True)
class InferenceBundle:
    booster: lgb.Booster
    calibrator: "DeployCalibrator"
    model_version: str
    feature_version: str
    num_feature: int


class DeployCalibrator:
    """从 calibration.json 还原的校准器（Platt 或 Isotonic）。"""

    def __init__(self, doc: dict):
        self.method = doc.get("method")
        self.feature_version = doc.get("feature_version", "unknown")
        if self.method in ("platt", "sigmoid"):
            if "params" not in doc:
                raise ArtifactLoadError(f"calibration.json method={self.method} 缺 params")
            self.params = [(float(a), float(b)) for a, b in doc["params"]]
            if len(self.params) != len(LABELS):
                raise ArtifactLoadError(f"Platt 参数类数 {len(self.params)} != {len(LABELS)}")
        elif self.method == "isotonic":
            iso = doc.get("iso")
            if not iso or "x" not in iso or "y" not in iso:
                raise ArtifactLoadError("calibration.json method=isotonic 缺 iso.x/iso.y")
            if len(iso["x"]) != len(LABELS) or len(iso["y"]) != len(LABELS):
                raise ArtifactLoadError("isotonic 断点类数不符")
            self.iso_x = np.array(iso["x"], dtype=float)
            self.iso_y = np.array(iso["y"], dtype=float)
        else:
            raise ArtifactLoadError(f"未知校准方法: {self.method}")

    @staticmethod
    def _sigmoid(z: float) -> float:
        if z >= 0:
            return 1.0 / (1.0 + math.exp(-z))
        ez = math.exp(z)
        return ez / (1.0 + ez)

    def calibrate(self, logits: np.ndarray) -> np.ndarray:
        """logits (n,3) 原始逐类 logit → 归一化校准概率 (n,3)。"""
        out = np.zeros((logits.shape[0], len(LABELS)))
        if self.method in ("platt", "sigmoid"):
            for c, (A, B) in enumerate(self.params):
                out[:, c] = [self._sigmoid(A * z + B) for z in logits[:, c]]
        else:  # isotonic
            for c in range(len(LABELS)):
                # np.interp 超出范围取端点 = sklearn out_of_bounds="clip"
                out[:, c] = np.interp(logits[:, c], self.iso_x[c], self.iso_y[c])
        s = out.sum(axis=1, keepdims=True)
        s[s == 0] = 1.0
        return out / s


def load_bundle(artifact_dir: str | Path) -> InferenceBundle:
    """加载工件 → 推理包。任何不合法都抛 ArtifactLoadError（→ 503 全停）。"""
    ad = Path(artifact_dir)
    model_path = ad / "model.txt"
    calib_path = ad / "calibration.json"
    config_path = ad / "config.json"
    if not model_path.exists():
        raise ArtifactLoadError(f"缺 {model_path}")
    if not calib_path.exists():
        raise ArtifactLoadError(f"缺 {calib_path}")

    booster = lgb.Booster(model_file=str(model_path))
    n_feat = booster.num_feature()
    if n_feat != len(FEATURE_NAMES):
        raise ArtifactLoadError(
            f"特征维不符: model={n_feat} vs spec={len(FEATURE_NAMES)}"
            f"（工件与当前特征版本不匹配，禁止推理）")

    cal = DeployCalibrator(json.loads(calib_path.read_text(encoding="utf-8")))

    model_version, feature_version = "unknown", cal.feature_version
    if config_path.exists():
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
        model_version = str(cfg.get("data_snapshot_hash", "unknown"))[:16]
    if ad.name:
        model_version = f"{ad.name}:{model_version}"

    return InferenceBundle(booster, cal, model_version, feature_version, n_feat)


def predict(bundle: InferenceBundle, pair: PairInput) -> dict:
    """单条 (付款, 候选发票) → 校准三分类概率 + top-1 标签。

    返回 {"probs": [pM,pP,pU], "label": str, "conf": float, "features": dict}。
    任何异常向上抛（app 层 → 503）。
    """
    feats = build_features(pair)
    x = to_vector(feats).reshape(1, -1)
    if x.shape[1] != bundle.num_feature:
        raise ArtifactLoadError(f"输入特征维 {x.shape[1]} != 工件 {bundle.num_feature}")
    logits = np.asarray(bundle.booster.predict(x, raw_score=True))
    probs = bundle.calibrator.calibrate(logits)[0]
    idx = int(probs.argmax())
    return {"probs": [float(p) for p in probs], "label": LABELS[idx],
            "conf": float(probs[idx]), "features": feats}
