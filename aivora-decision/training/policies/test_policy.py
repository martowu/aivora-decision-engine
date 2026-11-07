"""策略引擎单测（定稿 §7-A / §2.2.1 / §2.2.6）。

覆盖：
- 受限 AST：注入 / 逃逸 / 算术 / 下标 / lambda / 非白名单标识符全部拒绝
- 合法表达式：比较、布尔、in 列表、负数字面量
- first-match-wins + 硬规则优先于模型分数
- 无命中落默认 REVIEW；求值失败保守 REVIEW（绝不静默 AUTO）
- 黄金样例干跑（启动期校验）
- 版本化快照（decision_policies 可回滚）

运行：../.venv/bin/python test_policy.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from strategy import (PolicyCompileError, PolicyEngine, PolicyEvalError,
                      build_context, compile_expression, evaluate)

HERE = Path(__file__).parent
POLICY_YAML = HERE.parent.parent / "policies" / "recon" / "engine.yaml"
GOLDEN = HERE.parent.parent / "contracts" / "examples" / "recon.golden.json"

# ---------------------------------------------------------------------------
# 注入 / 逃逸防御（§7-A：禁止裸 eval）
# ---------------------------------------------------------------------------

INJECTION_CASES = [
    "__import__('os').system('id')",          # 内置导入逃逸
    "().__class__.__bases__",                 # 属性链逃逸（双下划线）
    "f.amount + 1",                           # 算术
    "f['amount']",                            # 下标
    "lambda: 1",                              # lambda
    "open('/etc/passwd')",                    # 非白名单函数
    "x if True else y",                       # 条件表达式
    "eval('1+1')",                            # 裸 eval
    "f.amount - f.amount",                    # 减号算术
    "'a'.upper()",                            # 方法调用
    "payment.amount ** 2",                    # 幂
    "f.amount // 2",                          # 整除
]


def test_injection_rejected():
    for expr in INJECTION_CASES:
        try:
            compile_expression(expr)
        except PolicyCompileError:
            continue
        raise AssertionError(f"未拒绝危险表达式: {expr!r}")


# 合法表达式必须通过
VALID_CASES = [
    "f.is_multi_mapping > 0",
    "calib.label == 'MATCHED' and calib.conf >= 0.9",
    "payment.amount <= 100000",
    "calib.label in ['MATCHED', 'PARTIAL']",
    "f.currency_mismatch == 0 and f.supplier_exact_match == 1",
    "not (f.is_multi_mapping > 0)",
    "calib.conf >= -0.5",           # 负数字面量（USub 允许）
    "payment.channel == 'BANK_TRANSFER' or payment.channel == 'ONLINE'",
]


def test_valid_expressions_compile():
    for expr in VALID_CASES:
        compile_expression(expr)  # 不抛即通过


def test_unknown_root_rejected():
    for expr in ("foo.amount > 1", "model.conf > 1"):
        try:
            compile_expression(expr)
        except PolicyCompileError:
            continue
        raise AssertionError(f"未拒绝非白名单根: {expr!r}")


def test_eval_basic():
    ctx = build_context(
        features={"is_multi_mapping": 0, "has_split_merge": 0,
                  "currency_mismatch": 0, "supplier_exact_match": 1,
                  "amount_abs_diff": 0.0},
        calib_probs=[0.98, 0.01, 0.01], label="MATCHED",
        payment={"amount": 50000, "currency": "CNY",
                 "channel": "BANK_TRANSFER", "payer_id": "CP-01"})
    assert evaluate("calib.conf >= 0.9", ctx) is True
    assert evaluate("calib.label == 'MATCHED'", ctx) is True
    assert evaluate("payment.amount > 100000", ctx) is False
    assert evaluate("f.is_multi_mapping in [0, 1]", ctx) is True


# ---------------------------------------------------------------------------
# 引擎行为：first-match-wins + 硬规则优先 + 默认 REVIEW
# ---------------------------------------------------------------------------


def load_engine() -> PolicyEngine:
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    return PolicyEngine.from_yaml(POLICY_YAML, golden)


def test_engine_loads_and_validates_golden():
    eng = load_engine()
    assert eng.policy_id == "recon.transaction_match"
    assert eng.version == "1.0"


def _ctx(f_over=None, pay_over=None, calib=None):
    f = {"is_multi_mapping": 0, "has_split_merge": 0, "currency_mismatch": 0,
         "supplier_exact_match": 1, "amount_abs_diff": 0.0}
    f.update(f_over or {})
    pay = {"amount": 50000, "currency": "CNY", "channel": "BANK_TRANSFER",
           "payer_id": "CP-01"}
    pay.update(pay_over or {})
    if calib is None:
        calib = (0.98, 0.01, 0.01, "MATCHED")
    pm, pp, pu, label = calib
    return build_context(f, [pm, pp, pu], label, pay)


def test_hard_rule_beats_model_score():
    eng = load_engine()
    # 高置信 MATCHED 但金额超上限 → 必须 REVIEW（金额硬规则先于 AUTO 规则）
    d = eng.decide(_ctx(pay_over={"amount": 150000},
                        calib=(0.99, 0.005, 0.005, "MATCHED")))
    assert d.action == "REVIEW", f"硬规则应优先, got {d.action} ({d.rule})"
    assert d.rule == "amount_above_cap"


def test_clean_matched_auto():
    eng = load_engine()
    d = eng.decide(_ctx(calib=(0.98, 0.01, 0.01, "MATCHED")))
    assert d.action == "AUTO"
    assert d.rule == "auto_matched_high_conf"


def test_no_hit_falls_to_default_review():
    eng = load_engine()
    # 高置信 UNMATCHED，无 AUTO 规则 → 默认 REVIEW
    d = eng.decide(_ctx(calib=(0.1, 0.0, 0.9, "UNMATCHED")))
    assert d.action == "REVIEW"
    assert d.rule == "default"


def test_struct_flag_review():
    eng = load_engine()
    d = eng.decide(_ctx(f_over={"is_multi_mapping": 1},
                        calib=(0.99, 0.005, 0.005, "MATCHED")))
    assert d.action == "REVIEW"
    assert d.rule == "struct_multi_mapping"


def test_eval_error_conservative_review():
    """求值失败（如缺键）必须保守 REVIEW，绝不静默 AUTO。"""
    eng = PolicyEngine.from_dict({
        "policy": {"id": "t", "version": "1", "default_action": "REVIEW"},
        "rules": [{"name": "r1", "when": "f.missing_key > 0",
                   "action": "AUTO"}],
    })
    d = eng.decide(_ctx())  # f 里没有 missing_key
    assert d.action == "REVIEW"
    assert d.rule.startswith("error:")


def test_first_match_order():
    """规则顺序决定命中（first-match-wins）。"""
    eng = PolicyEngine.from_dict({
        "policy": {"id": "t", "version": "1", "default_action": "REVIEW"},
        "rules": [
            {"name": "first", "when": "payment.amount >= 0", "action": "AUTO"},
            {"name": "second", "when": "payment.amount >= 0", "action": "REVIEW"},
        ],
    })
    d = eng.decide(_ctx())
    assert d.rule == "first"


def test_snapshot_versioned():
    eng = load_engine()
    snap = eng.snapshot()
    assert snap["policy_id"] == "recon.transaction_match"
    assert snap["version"] == "1.0"
    assert len(snap["rules"]) == 6
    # 快照可序列化（供 decision_policies 落库）
    json.dumps(snap, ensure_ascii=False)
    # 快照 → 引擎无损往返（决策服务回滚重建路径依赖此性质）
    eng2 = PolicyEngine.from_dict(snap)
    assert eng2.policy_id == eng.policy_id and eng2.version == eng.version
    assert eng2.default_action == eng.default_action
    assert [(r.name, r.when, r.action) for r in eng2.rules] == \
           [(r.name, r.when, r.action) for r in eng.rules]


def test_invalid_action_rejected():
    try:
        PolicyEngine.from_dict({
            "policy": {"id": "t", "version": "1"},
            "rules": [{"name": "r", "when": "f.is_multi_mapping > 0",
                       "action": "BLOCK"}],
        })
    except PolicyCompileError:
        return
    raise AssertionError("非法 action 未被拒绝")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"  PASS {fn.__name__}")
        except Exception as e:
            failed += 1
            print(f"  FAIL {fn.__name__}: {e!r}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
