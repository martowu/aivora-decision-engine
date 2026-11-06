"""决策存储层：全量落库可追溯（§2.2.2）+ 策略版本注册可回滚（§2.2.1）。

MVP 用**内存 + JSONL 审计日志**实现 `Store` 抽象：
- decision_logs 逐条 append 到 JSONL（durable、可 grep、可重放）；
- policy registry 在内存持有当前策略快照 + 历史版本（回滚 = 切指针）；
- model registry 持有当前 model_version。

生产后端是 Postgres 15（定稿）。这里刻意**不引入完整 ORM / migration**
（架构克制）：接口按"一张表一个 append/query"设计，Postgres 实现可直接替换
`JsonlStore` → `PostgresStore`（同接口），决策服务逻辑不动。
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional


@dataclass
class DecisionLog:
    decision_id: str
    payment_id: str
    invoice_id: str
    action: str                    # AUTO / REVIEW
    policy_rule: str
    policy_reason: str
    policy_version: str
    model_version: str
    feature_version: str
    label: str                     # 模型 top-1
    conf: float
    probs: dict                    # {MATCHED, PARTIAL, UNMATCHED}
    is_shadow: bool
    ts: float
    extra: dict = field(default_factory=dict)

    def to_row(self) -> dict:
        return {
            "decision_id": self.decision_id, "payment_id": self.payment_id,
            "invoice_id": self.invoice_id, "action": self.action,
            "policy_rule": self.policy_rule, "policy_reason": self.policy_reason,
            "policy_version": self.policy_version, "model_version": self.model_version,
            "feature_version": self.feature_version, "label": self.label,
            "conf": self.conf, "probs": self.probs, "is_shadow": self.is_shadow,
            "ts": self.ts, **self.extra,
        }


class Store:
    """决策存储接口（Postgres 实现可替换）。"""

    def append_decision(self, log: DecisionLog) -> None: ...
    def decisions(self, since_ts: float = 0.0, limit: int = 1000) -> Iterator[DecisionLog]: ...
    def current_policy_version(self) -> str: ...
    def set_current_policy_version(self, v: str) -> None: ...
    def policy_versions(self) -> list[dict]: ...
    def rollback_policy(self, target_version: str) -> str: ...
    def current_model_version(self) -> str: ...
    def set_current_model_version(self, v: str) -> None: ...
    def model_versions(self) -> list[str]: ...


class JsonlStore(Store):
    """内存索引 + JSONL 持久化。线程安全（单进程服务足够）。"""

    def __init__(self, decisions_path: str | Path | None = None):
        # RLock：register_policy / rollback_policy 持锁后会再调
        # set_current_policy_version（内部再取锁），普通 Lock 会死锁。
        self._lock = threading.RLock()
        self._mem: list[DecisionLog] = []
        self._decisions_path = Path(decisions_path) if decisions_path else None
        self._policy_versions: list[dict] = []   # [{version, snapshot, ts, active}]
        self._current_policy: Optional[str] = None
        self._model_versions: list[str] = []
        self._current_model: Optional[str] = None

    # ---- decision_logs ----
    def append_decision(self, log: DecisionLog) -> None:
        with self._lock:
            self._mem.append(log)
            if self._decisions_path:
                self._decisions_path.parent.mkdir(parents=True, exist_ok=True)
                with self._decisions_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(log.to_row(), ensure_ascii=False) + "\n")

    def decisions(self, since_ts: float = 0.0, limit: int = 1000) -> Iterator[DecisionLog]:
        with self._lock:
            rows = [d for d in self._mem if d.ts >= since_ts]
        yield from rows[-limit:]

    # ---- policy registry ----
    def current_policy_version(self) -> str:
        return self._current_policy or "none"

    def set_current_policy_version(self, v: str) -> None:
        with self._lock:
            if self._policy_versions:
                for p in self._policy_versions:
                    p["active"] = (p["version"] == v)
            self._current_policy = v

    def policy_versions(self) -> list[dict]:
        with self._lock:
            return [dict(p) for p in self._policy_versions]

    def register_policy(self, version: str, snapshot: dict) -> None:
        """登记一个策略版本快照（启动时 + 热切换时调用）。"""
        with self._lock:
            if any(p["version"] == version for p in self._policy_versions):
                return
            self._policy_versions.append({"version": version, "snapshot": snapshot,
                                          "ts": time.time(), "active": False})
            self.set_current_policy_version(version)

    def rollback_policy(self, target_version: str) -> str:
        with self._lock:
            if not any(p["version"] == target_version for p in self._policy_versions):
                raise ValueError(f"策略版本 {target_version} 未登记，无法回滚")
            self.set_current_policy_version(target_version)
            return target_version

    # ---- model registry ----
    def current_model_version(self) -> str:
        return self._current_model or "none"

    def set_current_model_version(self, v: str) -> None:
        with self._lock:
            if v not in self._model_versions:
                self._model_versions.append(v)
            self._current_model = v

    def model_versions(self) -> list[str]:
        with self._lock:
            return list(self._model_versions)


def make_decision_id() -> str:
    return f"DEC-{time.time_ns():x}"
