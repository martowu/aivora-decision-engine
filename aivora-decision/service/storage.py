"""决策存储层：全量落库可追溯（§2.2.2）+ 策略版本注册可回滚（§2.2.1）。

MVP 用**内存 + JSONL 审计日志**实现 `Store` 抽象：
- decision_logs 逐条 append 到 JSONL（durable、可 grep、可重放）；
- policy registry 在内存持有当前策略快照 + 历史版本（回滚 = 切指针）；
- model registry 持有当前 model_version；
- feedback（人工复核回写）持久化到独立 JSONL，按 (request_id, reviewer) upsert；
- **重启可恢复**：启动时从 JSONL 重建 decision 索引与 feedback，幂等重放跨重启有效。

生产后端是 Postgres 15（定稿）。这里刻意**不引入完整 ORM / migration**
（架构克制）：接口按"一张表一个 append/query"设计，Postgres 实现可直接替换
`JsonlStore` → `PostgresStore`（同接口），决策服务逻辑不动。
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import dataclass, field, fields
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
    request_id: str = ""           # 幂等键（调用方提供；缺省服务端生成，不参与去重）
    tenant_id: str = "default"     # 一期单租户，字段先贯穿（行级隔离预留）
    input_digest: str = ""         # 输入摘要 sha256（定稿 §2.2.8：明细按脱敏策略存 input_uri）
    latency_ms: int = 0            # 推理+策略求值耗时（G-Shadow P95≤100ms 监控源）
    degraded: bool = False         # 降级标志（503 全停不产生日志，恒 False）
    extra: dict = field(default_factory=dict)

    def to_row(self) -> dict:
        return {
            "decision_id": self.decision_id, "request_id": self.request_id,
            "tenant_id": self.tenant_id,
            "payment_id": self.payment_id, "invoice_id": self.invoice_id,
            "action": self.action,
            "policy_rule": self.policy_rule, "policy_reason": self.policy_reason,
            "policy_version": self.policy_version, "model_version": self.model_version,
            "feature_version": self.feature_version, "label": self.label,
            "conf": self.conf, "probs": self.probs, "is_shadow": self.is_shadow,
            "ts": self.ts, "input_digest": self.input_digest,
            "latency_ms": self.latency_ms, "degraded": self.degraded, **self.extra,
        }

    @classmethod
    def from_row(cls, row: dict) -> "DecisionLog":
        """从 JSONL 行重建（兼容旧格式行：缺 request_id/tenant_id 用默认值）。"""
        known = {f.name for f in fields(cls) if f.name != "extra"}
        extra = {k: v for k, v in row.items() if k not in known}
        kwargs = {k: v for k, v in row.items() if k in known}
        return cls(extra=extra, **kwargs)


class Store:
    """决策存储接口（Postgres 实现可替换）。"""

    def append_decision(self, log: DecisionLog) -> None: ...
    def decisions(self, since_ts: float = 0.0, limit: int = 1000) -> Iterator[DecisionLog]: ...
    def decision_by_request_id(self, request_id: str) -> Optional[DecisionLog]: ...
    def current_policy_version(self) -> str: ...
    def set_current_policy_version(self, v: str) -> None: ...
    def policy_versions(self) -> list[dict]: ...
    def rollback_policy(self, target_version: str) -> str: ...
    def current_model_version(self) -> str: ...
    def set_current_model_version(self, v: str) -> None: ...
    def model_versions(self) -> list[str]: ...
    def append_feedback(self, rec: dict) -> None: ...
    def feedback_records(self, status: Optional[str] = None,
                         limit: int = 1000) -> list[dict]: ...


class JsonlStore(Store):
    """内存索引 + JSONL 持久化。线程安全（单进程服务足够）。

    幂等索引语义：request_id → **窗口内第一条**决策。
    窗口内（< IDEMPOTENCY_WINDOW_S）重复 → 服务层返回首次结果（409 重放）；
    超窗口的旧条目 → 视为新请求，索引刷新为新条目。
    """

    def __init__(self, decisions_path: str | Path | None = None,
                 feedback_path: str | Path | None = None,
                 idempotency_window_s: float = 7 * 86400):
        self._lock = threading.RLock()
        # RLock：register_policy / rollback_policy 持锁后会再调
        # set_current_policy_version（内部再取锁），普通 Lock 会死锁。
        self._mem: list[DecisionLog] = []
        self._decisions_path = Path(decisions_path) if decisions_path else None
        self._feedback_path = (Path(feedback_path) if feedback_path
                               else (self._decisions_path.parent / "feedback.jsonl"
                                     if self._decisions_path else None))
        self._by_request: dict[str, DecisionLog] = {}
        self._idempotency_window_s = idempotency_window_s
        self._policy_versions: list[dict] = []   # [{version, snapshot, ts, active}]
        self._current_policy: Optional[str] = None
        self._model_versions: list[str] = []
        self._current_model: Optional[str] = None
        self._feedback: dict[tuple[str, str], dict] = {}
        self._load_from_disk()

    # ---- 启动重载（重启不丢幂等键 / feedback）----
    def _load_from_disk(self) -> None:
        if self._decisions_path and self._decisions_path.exists():
            for line in self._decisions_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    log = DecisionLog.from_row(row)
                except (json.JSONDecodeError, TypeError, ValueError):
                    continue  # 坏行跳过（审计文件容错），不阻塞启动
                self._mem.append(log)
                if log.request_id and log.request_id not in self._by_request:
                    self._by_request[log.request_id] = log
            self._mem.sort(key=lambda d: d.ts)
        if self._feedback_path and self._feedback_path.exists():
            for line in self._feedback_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                    self._feedback[(rec["request_id"], rec["reviewer"])] = rec
                except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                    continue

    # ---- decision_logs ----
    def append_decision(self, log: DecisionLog) -> None:
        with self._lock:
            self._mem.append(log)
            if log.request_id:
                prev = self._by_request.get(log.request_id)
                # 仅当无旧条目或旧条目已出窗口时才刷新（窗口内首次结果保持权威）
                if prev is None or (log.ts - prev.ts) >= self._idempotency_window_s:
                    self._by_request[log.request_id] = log
            if self._decisions_path:
                self._decisions_path.parent.mkdir(parents=True, exist_ok=True)
                with self._decisions_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(log.to_row(), ensure_ascii=False) + "\n")

    def decisions(self, since_ts: float = 0.0, limit: int = 1000) -> Iterator[DecisionLog]:
        with self._lock:
            rows = [d for d in self._mem if d.ts >= since_ts]
        yield from rows[-limit:]

    def decision_by_request_id(self, request_id: str) -> Optional[DecisionLog]:
        with self._lock:
            return self._by_request.get(request_id)

    def in_idempotency_window(self, request_id: str) -> bool:
        """request_id 是否存在窗口内的首次决策（服务层幂等判定用）。"""
        with self._lock:
            prev = self._by_request.get(request_id)
            return prev is not None and (time.time() - prev.ts) < self._idempotency_window_s

    # ---- feedback（人工复核回写；(request_id, reviewer) 唯一，重提 = 更新）----
    def append_feedback(self, rec: dict) -> None:
        key = (rec["request_id"], rec["reviewer"])
        with self._lock:
            self._feedback[key] = rec
            if self._feedback_path:
                # MVP 量级小：整文件重写保证 upsert 一致性
                self._feedback_path.parent.mkdir(parents=True, exist_ok=True)
                rows = sorted(self._feedback.values(), key=lambda r: r.get("reviewed_at", 0))
                tmp = self._feedback_path.with_suffix(".tmp")
                tmp.write_text(
                    "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                    encoding="utf-8")
                tmp.replace(self._feedback_path)

    def feedback_records(self, status: Optional[str] = None,
                         limit: int = 1000) -> list[dict]:
        with self._lock:
            rows = sorted(self._feedback.values(), key=lambda r: r.get("reviewed_at", 0))
        if status:
            rows = [r for r in rows if r.get("status") == status]
        return rows[:limit]

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


def make_request_id() -> str:
    """调用方未提供 request_id 时服务端生成（不参与幂等去重）。"""
    return f"REQ-{uuid.uuid4().hex[:16]}"
