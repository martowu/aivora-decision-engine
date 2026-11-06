"""部署闭环验证（W2 交付物自检）：加载导出的 model.txt 规范工件，
确认 (1) 可加载 (2) 特征维/类别数正确 (3) 同输入推理确定 (4) 概率形状合法。

这是对定稿 §1.2「模型 = 特征构建函数(版本化)+训练配置+模型工件+校准器」
部署侧的验收：生产 FastAPI 容器加载 model.txt 后必须通过这些检查。

运行（对任意一次 train.py 产出的 artifact 目录）：
    .venv/bin/python verify_deploy.py /tmp/selftest-artifact
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import lightgbm as lgb

from features import FEATURE_NAMES, PairInput, build_features, to_vector

LABELS = ["MATCHED", "PARTIAL", "UNMATCHED"]


def main(artifact_dir: str) -> int:
    ad = Path(artifact_dir)
    model_path = ad / "model.txt"
    if not model_path.exists():
        print(f"FAIL: 找不到 {model_path}（先跑 train.py）")
        return 1

    booster = lgb.Booster(model_file=str(model_path))

    # (1) 特征维必须等于版本化特征清单
    n_feat = booster.num_feature()
    ok_feat = n_feat == len(FEATURE_NAMES)

    # 构造一个「干净应匹配」样本：主体一致 + 附言含发票号 + 金额一致
    p = dict(
        payment_id="PAY-9", invoice_id="INV-0009",
        pay_amount=50_000.0, inv_amount=50_000.0,
        pay_currency="CNY", inv_currency="CNY",
        pay_date="2026-09-01", inv_date="2026-08-28",
        payer_id="CP-09-01", supplier_id="CP-09-01", supplier_name="甲公司",
        channel="BANK_TRANSFER", reference="付款 INV-0009 全额", note="备注",
        fee=0.0, counterparty_match_rate_hist=0.99, is_first_counterparty=False,
        is_multi_mapping=False, has_split_merge=False,
    )
    ins = PairInput(**p)
    X = to_vector(build_features(ins)).reshape(1, -1)

    r1 = booster.predict(X)[0]
    r2 = booster.predict(X)[0]

    # (2) 类别数 = 3
    ok_class = len(r1) == 3
    # (3) 推理确定（同输入同输出）
    ok_deter = bool(np.allclose(r1, r2))
    # (4) 概率合法：和为 1、非负
    ok_prob = bool(abs(r1.sum() - 1.0) < 1e-6) and bool((r1 >= 0).all())

    pred = LABELS[int(r1.argmax())]

    checks = {
        "feature_dim_matches": (ok_feat, f"model={n_feat} vs spec={len(FEATURE_NAMES)}"),
        "num_classes_is_3": (ok_class, f"got {len(r1)}"),
        "inference_deterministic": (ok_deter, ""),
        "probs_valid": (ok_prob, f"sum={r1.sum():.6f}"),
    }
    print(f"artifact      : {ad}")
    print(f"probs (MATCHED/PARTIAL/UNMATCHED) = {np.round(r1, 4).tolist()}")
    print(f"argmax        : {pred}")
    all_ok = True
    for name, (ok, detail) in checks.items():
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_ok = False
        print(f"  [{status}] {name}" + (f"  ({detail})" if detail else ""))

    # 干净应匹配样本，模型应给出 MATCHED 最高（合成数据下通常成立；
    # 真实数据以 test 集 per-class 为准，这里仅作工件健全性检查）
    print(f"\n部署闭环：{'全部通过 ✓' if all_ok else '存在失败 ✗'}")
    return 0 if all_ok else 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/selftest-artifact"))
