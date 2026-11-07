"""策略引擎（定稿 §7-A 受限求值 + §2.2.1 first-match-wins + §2.2.6 风险分级）。

核心原则：
- **禁止裸 eval**。`when` 表达式启动时解析为 AST，白名单校验，拒绝一切
  内置名、属性链逃逸、算术、下标、lambda、条件表达式等（§7-A）。
- **first-match-wins 规则链**，硬规则（结构 flag / 主体不一致 / 金额上限 /
  黑名单）排在模型分数规则之前 → 优先级永远高于模型分数；无命中落默认 REVIEW。
- **REVIEW 是策略动作，不是模型标签**（§1.3.4）：模型只出三态 + 校准概率，
  "证据不足 / 结构复杂"由特征 flag 触发策略路由 REVIEW。
- **阈值可配、默认用定稿示例值**（D5 决策会书面确认后改 YAML 数字即可，不改代码）。
- **编译 + 干跑黄金样例**，编译失败 / 黄金样例不符 → 拒绝启动（§7-A）。

变量域（白名单根标识符，§7-A）：
- ``f.*``      特征（build_features 输出，point-in-time 安全）
- ``calib.*``  校准结果：``matched/partial/unmatched`` 三类概率、``label`` 预测三态、
               ``conf`` 最大概率
- ``payment.*`` 付款事实：``amount/currency/channel/payer_id``
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

# ---------------------------------------------------------------------------
# 异常
# ---------------------------------------------------------------------------


class SafePolicyError(Exception):
    """策略引擎错误基类。"""


class PolicyCompileError(SafePolicyError):
    """表达式编译 / 白名单校验失败（启动期，须拒绝启动）。"""


class PolicyEvalError(SafePolicyError):
    """运行时求值错误（未知键、类型不可比较等）。"""


# ---------------------------------------------------------------------------
# 白名单
# ---------------------------------------------------------------------------

ROOTS = {"f", "calib", "payment"}

_ALLOWED_NODES = {
    ast.Expression,
    ast.BoolOp, ast.And, ast.Or,
    ast.UnaryOp, ast.Not, ast.USub,
    ast.Compare, ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
    ast.In, ast.NotIn,
    ast.Constant, ast.Name, ast.Attribute,
    ast.List, ast.Tuple,  # 仅用于 `in` 的常量列表
    ast.Load,             # 表达式上下文（Name/Attribute 携带），无害
}

# 显式禁止清单（用于给出更友好的报错）
_FORBIDDEN = {
    ast.BinOp: "算术运算（+ - * / // % **）",
    ast.Call: "函数调用（如 __import__ / eval / open）",
    ast.Subscript: "下标访问 []",
    ast.Lambda: "lambda",
    ast.IfExp: "条件表达式 a if c else b",
    ast.Starred: "* 展开",
    ast.Dict: "字典字面量",
    ast.Set: "集合字面量",
    ast.NamedExpr: "海象 :=",
}


def _forbidden_name(node: ast.AST) -> str | None:
    for cls, why in _FORBIDDEN.items():
        if isinstance(node, cls):
            return why
    if type(node) not in _ALLOWED_NODES:
        return f"语法节点 {type(node).__name__}"
    return None


# ---------------------------------------------------------------------------
# 编译（AST 白名单校验）
# ---------------------------------------------------------------------------


def compile_expression(expr: str) -> ast.Expression:
    """把 `when` 表达式解析为已白名单校验的 AST。失败抛 PolicyCompileError。"""
    if not isinstance(expr, str) or not expr.strip():
        raise PolicyCompileError("表达式为空")
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise PolicyCompileError(f"语法错误: {e.msg} (第 {e.lineno} 行)") from e

    for node in ast.walk(tree):
        why = _forbidden_name(node)
        if why:
            raise PolicyCompileError(f"禁止使用 {why}（§7-A 白名单）")
        if isinstance(node, ast.Name):
            if node.id not in ROOTS:
                raise PolicyCompileError(
                    f"标识符 {node.id!r} 不在白名单 {sorted(ROOTS)}")
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("_"):
                raise PolicyCompileError(f"属性链逃逸: {node.attr!r}")
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            if not isinstance(node.operand, ast.Constant):
                raise PolicyCompileError("负号只允许用于数值字面量")
    return tree


# ---------------------------------------------------------------------------
# 求值
# ---------------------------------------------------------------------------


def _truthy(v: Any) -> bool:
    return bool(v)


def _resolve(node: ast.AST, ctx: Mapping[str, Any]) -> Any:
    """解析 Name / 属性链 → 点路径 → 在 ctx 的嵌套 dict 中查值。
    左起第一个必须是白名单根；禁止下划线/双下划线键。"""
    path: list[str] = []
    cur = node
    while isinstance(cur, ast.Attribute):
        path.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        path.append(cur.id)
    else:
        raise PolicyEvalError("点路径必须以标识符开头")
    path.reverse()  # 左起

    if path[0] not in ROOTS:
        raise PolicyEvalError(f"根标识符 {path[0]!r} 不在白名单")
    obj: Any = ctx
    for i, part in enumerate(path):
        if part.startswith("_"):
            raise PolicyEvalError(f"禁止访问 {part!r}")
        if not isinstance(obj, Mapping):
            raise PolicyEvalError(
                f"{'.'.join(path[:i])!r} 不是可索引的对象")
        if part not in obj:
            raise PolicyEvalError(f"未知键: {'.'.join(path[:i + 1])!r}")
        obj = obj[part]
    return obj


def _compare(a: Any, b: Any, op: ast.cmpop) -> bool:
    if isinstance(op, ast.In):
        if not isinstance(b, (list, tuple)):
            raise PolicyEvalError("in 右端必须是列表")
        return a in b
    if isinstance(op, ast.NotIn):
        if not isinstance(b, (list, tuple)):
            raise PolicyEvalError("not in 右端必须是列表")
        return a not in b

    # 相等/不等允许任意可比较类型
    if isinstance(op, ast.Eq):
        return a == b
    if isinstance(op, ast.NotEq):
        return a != b

    # 大小比较：要求同类型（数值 int/float 可互通），布尔除外
    is_num = lambda v: isinstance(v, (int, float)) and not isinstance(v, bool)  # noqa: E731
    both_num = is_num(a) and is_num(b)
    both_str = isinstance(a, str) and isinstance(b, str)
    if not (both_num or both_str):
        raise PolicyEvalError(
            f"类型不可比较: {type(a).__name__} vs {type(b).__name__}")
    try:
        if isinstance(op, ast.Lt):
            return a < b
        if isinstance(op, ast.LtE):
            return a <= b
        if isinstance(op, ast.Gt):
            return a > b
        return a >= b
    except TypeError as e:
        raise PolicyEvalError(f"比较失败: {e}") from e


def _eval(node: ast.AST, ctx: Mapping[str, Any]) -> Any:
    if isinstance(node, ast.Expression):
        return _eval(node.body, ctx)
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, (ast.Name, ast.Attribute)):
        return _resolve(node, ctx)
    if isinstance(node, (ast.List, ast.Tuple)):
        return [_eval(e, ctx) for e in node.elts]
    if isinstance(node, ast.UnaryOp):
        if isinstance(node.op, ast.Not):
            return not _truthy(_eval(node.operand, ctx))
        if isinstance(node.op, ast.USub):
            return -_eval(node.operand, ctx)
    if isinstance(node, ast.BoolOp):
        vals = [_eval(v, ctx) for v in node.values]
        if isinstance(node.op, ast.And):
            return all(_truthy(v) for v in vals)
        return any(_truthy(v) for v in vals)
    if isinstance(node, ast.Compare):
        left = _eval(node.left, ctx)
        for op, comp in zip(node.ops, node.comparators):
            right = _eval(comp, ctx)
            if not _compare(left, right, op):
                return False
            left = right
        return True
    raise PolicyEvalError(f"未实现的节点: {type(node).__name__}")


def evaluate(expr: str, ctx: Mapping[str, Any]) -> bool:
    """编译 + 求值（便捷入口）。生产环境应先用 compile_expression 缓存 AST。"""
    return bool(_eval(compile_expression(expr), ctx))


# ---------------------------------------------------------------------------
# 规则 / 引擎
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PolicyRule:
    name: str
    when: str
    compiled: ast.Expression
    action: str
    reason: str = ""


@dataclass
class Decision:
    action: str                 # AUTO / REVIEW
    rule: str                   # 命中的规则名，或 "default"
    reason: str
    context: Mapping[str, Any] = field(repr=False, default_factory=dict)

    def to_log(self) -> dict:
        """decision_logs 落库字段（定稿 §2.2.2 可追溯）。"""
        return {"action": self.action, "policy_rule": self.rule,
                "policy_reason": self.reason}


VALID_ACTIONS = {"AUTO", "REVIEW"}


class PolicyEngine:
    """first-match-wins 规则链。硬规则在前，模型分数规则在后，默认落 REVIEW。"""

    def __init__(self, policy_id: str, version: str,
                 rules: Sequence[PolicyRule], default_action: str = "REVIEW"):
        if default_action not in VALID_ACTIONS:
            raise PolicyCompileError(f"default_action 非法: {default_action}")
        for r in rules:
            if r.action not in VALID_ACTIONS:
                raise PolicyCompileError(f"规则 {r.name} action 非法: {r.action}")
        self.policy_id = policy_id
        self.version = version
        self.rules = list(rules)
        self.default_action = default_action

    # ---- 加载 / 编译 ----
    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "PolicyEngine":
        """接受两种形态：
        1. 原始策略文档：{"policy": {id, version, default_action}, "rules": [...]}
        2. snapshot() 产物：{"policy_id", "version", "default_action", "rules": [...]}
           （决策服务回滚时从存储快照重建引擎，必须无损往返）
        """
        pol = d.get("policy", {}) if isinstance(d.get("policy"), Mapping) else {}
        rules_raw = d.get("rules", [])
        rules = []
        seen = set()
        for r in rules_raw:
            name = r.get("name")
            if not name:
                raise PolicyCompileError("规则缺少 name")
            if name in seen:
                raise PolicyCompileError(f"规则重名: {name}")
            seen.add(name)
            compiled = compile_expression(r.get("when", ""))
            rules.append(PolicyRule(
                name=name, when=r.get("when", ""), compiled=compiled,
                action=str(r.get("action", "")).upper(),
                reason=r.get("reason", "")))
        return cls(
            pol.get("id") or d.get("policy_id", "policy"),
            str(pol.get("version") or d.get("version", "0.0")),
            rules,
            pol.get("default_action") or d.get("default_action", "REVIEW"),
        )

    @classmethod
    def from_yaml(cls, path: str | Path,
                  golden: Sequence[Mapping[str, Any]] | None = None) -> "PolicyEngine":
        """编译 + （可选）干跑黄金样例。任一步失败抛错 → 拒绝启动（§7-A）。"""
        text = Path(path).read_text(encoding="utf-8")
        data = yaml.safe_load(text)
        if not isinstance(data, dict):
            raise PolicyCompileError("YAML 顶层必须是映射")
        eng = cls.from_dict(data)
        if golden is not None:
            eng.validate(golden)
        return eng

    # ---- 黄金样例校验（启动期） ----
    def validate(self, samples: Sequence[Mapping[str, Any]]) -> None:
        for i, s in enumerate(samples):
            ctx = s["context"]
            expect = s["action"].upper()
            got = self.decide(ctx)
            if got.action != expect:
                raise PolicyCompileError(
                    f"黄金样例 #{i} 不符: 期望 {expect} 实际 {got.action} "
                    f"(rule={got.rule})")

    # ---- 决策（first-match-wins） ----
    def decide(self, ctx: Mapping[str, Any]) -> Decision:
        for r in self.rules:
            try:
                hit = _truthy(_eval(r.compiled, ctx))
            except PolicyEvalError:
                # 求值失败（如缺键）→ 保守落 REVIEW，绝不静默 AUTO
                return Decision("REVIEW", f"error:{r.name}",
                                f"规则 {r.name} 求值失败，保守 REVIEW", ctx)
            if hit:
                return Decision(r.action, r.name, r.reason, ctx)
        return Decision(self.default_action, "default",
                        f"无规则命中，默认 {self.default_action}", ctx)

    def snapshot(self) -> dict:
        """decision_policies 版本化字段（定稿 §2.2.1 可回滚）。"""
        return {"policy_id": self.policy_id, "version": self.version,
                "default_action": self.default_action,
                "rules": [{"name": r.name, "when": r.when,
                           "action": r.action, "reason": r.reason}
                          for r in self.rules]}


# ---------------------------------------------------------------------------
# 上下文构建（松耦合：只依赖 f/calib/payment 三个根，不依赖 features 内部）
# ---------------------------------------------------------------------------

def build_context(features: Mapping[str, Any],
                  calib_probs: Sequence[float],
                  label: str,
                  payment: Mapping[str, Any]) -> dict:
    """把 模型输出 + 特征 + 付款事实 组装成策略变量域。

    features  : build_features(inp) 的输出 dict
    calib_probs: 校准后三分类概率 [p(MATCHED), p(PARTIAL), p(UNMATCHED)]
    label     : 预测三态（top-1，= LABELS[argmax(calib_probs)]）
    payment   : {amount, currency, channel, payer_id}
    """
    cp = [float(x) for x in calib_probs]
    if len(cp) != 3:
        raise ValueError("calib_probs 必须 3 维")
    return {
        "f": dict(features),
        "calib": {
            "matched": cp[0], "partial": cp[1], "unmatched": cp[2],
            "label": label, "conf": max(cp),
        },
        "payment": dict(payment),
    }
