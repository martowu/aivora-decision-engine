"""PostgresStore：Store 接口的生产后端（定稿 §2.1 / §2.2.8，Postgres 15）。

与 `JsonlStore` 同接口、可直接替换（storage.py 设计）：decision-service 逻辑不变。
DDL = `infra/migrations/0001_init.sql`。

与 JsonlStore 的语义对齐（全部保守）：
- request_id 永久幂等键（DB 主键）：重复 request_id → ON CONFLICT DO NOTHING，
  服务层 409 重放首次结果。定稿"7 天窗口"= 最短保留期；审计保留期更长，
  幂等键对逻辑请求唯一，重放语义不变。
- feedback 按 (decision_log_id, reviewer) upsert，同 JsonlStore。
- 策略/模型注册表落 decision_policies / decision_models 的 status 列：
  策略回滚 = active 指针切换（定稿 §2.2.1，比模型回滚更常用）；
  模型热加载 = production 指针切换（工件已版本化，分钟级）。
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Iterator, Optional

from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from storage import DecisionLog, Store


def _dt_to_epoch(dt) -> float:
    return dt.timestamp() if dt is not None else 0.0


class PostgresStore(Store):
    def __init__(self, db_url: str,
                 task_id: str = "recon.transaction_match",
                 task_version: str = "1.0"):
        import psycopg_pool
        self._task_id = task_id
        self._task_version = task_version
        self._pool = psycopg_pool.ConnectionPool(
            db_url, min_size=1, max_size=4,
            kwargs={"autocommit": True}, open=False)
        self._lock = threading.RLock()
        # RLock：register_policy 持锁后会内调 set_current_policy_version（再取锁），
        # 普通 Lock 会死锁（与 JsonlStore 同源的教训）。
        self._policy_cache: dict[str, dict] = {}   # version → snapshot（回滚重建兜底）

    # ---- 连接 ----
    def open(self, timeout_s: float = 30.0) -> None:
        """启动期打开连接池并探活；失败抛错 → app 层保持 not-ready（503 全停）。"""
        self._pool.open(wait=True, timeout=timeout_s)
        with self._pool.connection() as conn:
            conn.execute("SELECT 1")

    def close(self) -> None:
        self._pool.close()

    @contextmanager
    def _conn(self):
        conn = self._pool.getconn()
        try:
            conn.row_factory = dict_row
            yield conn
        finally:
            self._pool.putconn(conn)

    # ---- decision_logs ----
    def append_decision(self, log: DecisionLog) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO decision_logs (
                  request_id, decision_id, tenant_id, task_id, task_version,
                  model_version, policy_version, feature_version, mode,
                  input_digest, prediction, calibrated_prob, action,
                  reason_codes, rules_triggered, latency_ms, degraded, created_at
                ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,
                          %s,%s,%s,%s,%s,%s)
                ON CONFLICT (request_id) DO NOTHING
                """,
                (
                    log.request_id, log.decision_id, log.tenant_id,
                    self._task_id, self._task_version,
                    log.model_version, log.policy_version, log.feature_version,
                    "shadow" if log.is_shadow else "online",
                    log.input_digest or "",
                    Jsonb({"label": log.label, "conf": log.conf,
                           "payment_id": log.payment_id, "invoice_id": log.invoice_id,
                           "decision_id": log.decision_id}),
                    Jsonb(log.probs),
                    log.action,
                    Jsonb([log.policy_reason]), Jsonb([log.policy_rule]),
                    log.latency_ms, log.degraded,
                    datetime.fromtimestamp(log.ts, tz=timezone.utc),
                ),
            )

    _SELECT_LOG = f"""
      SELECT request_id, decision_id, tenant_id, model_version, policy_version,
             feature_version, mode, input_digest, prediction, calibrated_prob,
             action, reason_codes, rules_triggered, latency_ms, degraded, created_at
      FROM decision_logs
    """

    @classmethod
    def _row_to_log(cls, d: dict) -> DecisionLog:
        pred = d.get("prediction") or {}
        cal = d.get("calibrated_prob") or {}
        reasons = d.get("reason_codes") or []
        rules = d.get("rules_triggered") or []
        return DecisionLog(
            decision_id=pred.get("decision_id") or d.get("decision_id") or "",
            payment_id=pred.get("payment_id") or "",
            invoice_id=pred.get("invoice_id") or "",
            action=d.get("action") or "",
            policy_rule=rules[0] if rules else "",
            policy_reason=reasons[0] if reasons else "",
            policy_version=d.get("policy_version") or "",
            model_version=d.get("model_version") or "",
            feature_version=d.get("feature_version") or "",
            label=pred.get("label") or "",
            conf=float(pred.get("conf") or 0.0),
            probs=dict(cal),
            is_shadow=(d.get("mode") == "shadow"),
            ts=_dt_to_epoch(d.get("created_at")),
            request_id=d.get("request_id") or "",
            tenant_id=d.get("tenant_id") or "default",
            input_digest=d.get("input_digest") or "",
            latency_ms=d.get("latency_ms") or 0,
            degraded=bool(d.get("degraded")),
        )

    def decisions(self, since_ts: float = 0.0, limit: int = 1000) -> Iterator[DecisionLog]:
        with self._conn() as conn:
            rows = conn.execute(
                self._SELECT_LOG + """
                WHERE created_at >= %s
                ORDER BY created_at DESC
                LIMIT %s""",
                (datetime.fromtimestamp(since_ts, tz=timezone.utc) if since_ts
                 else datetime.fromtimestamp(0, tz=timezone.utc), limit),
            ).fetchall()
        for d in reversed(rows):   # 与 JsonlStore 一致：按时间升序产出
            yield self._row_to_log(d)

    def decision_by_request_id(self, request_id: str) -> Optional[DecisionLog]:
        with self._conn() as conn:
            d = conn.execute(self._SELECT_LOG + " WHERE request_id = %s",
                             (request_id,)).fetchone()
        return self._row_to_log(d) if d else None

    def in_idempotency_window(self, request_id: str,
                              window_s: float = 7 * 86400) -> bool:
        with self._conn() as conn:
            d = conn.execute("SELECT created_at FROM decision_logs WHERE request_id = %s",
                             (request_id,)).fetchone()
        if not d:
            return False
        return (datetime.now(timezone.utc).timestamp() - _dt_to_epoch(d["created_at"])) < window_s

    # ---- decision_feedback（(decision_log_id, reviewer) 唯一，重提 = 更新）----
    def append_feedback(self, rec: dict) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO decision_feedback (
                  decision_log_id, reviewer, reviewed_at, review_action,
                  correction_reason, status, corrected_label, model_label, tenant_id
                ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (decision_log_id, reviewer) DO UPDATE SET
                  reviewed_at = excluded.reviewed_at,
                  review_action = excluded.review_action,
                  correction_reason = excluded.correction_reason,
                  status = excluded.status,
                  corrected_label = excluded.corrected_label,
                  model_label = excluded.model_label,
                  tenant_id = excluded.tenant_id
                """,
                (
                    rec["request_id"], rec["reviewer"],
                    datetime.fromtimestamp(rec["reviewed_at"], tz=timezone.utc),
                    rec.get("review_action"), rec.get("correction_reason"),
                    rec.get("status", "pending"),
                    rec.get("corrected_label"), rec.get("model_label"),
                    rec.get("tenant_id"),
                ),
            )

    def feedback_records(self, status: Optional[str] = None,
                         limit: int = 1000) -> list[dict]:
        with self._conn() as conn:
            if status:
                rows = conn.execute(
                    """SELECT decision_log_id AS request_id, reviewer, reviewed_at,
                              review_action, correction_reason, status,
                              corrected_label, model_label, tenant_id
                       FROM decision_feedback WHERE status = %s
                       ORDER BY reviewed_at ASC LIMIT %s""",
                    (status, limit)).fetchall()
            else:
                rows = conn.execute(
                    """SELECT decision_log_id AS request_id, reviewer, reviewed_at,
                              review_action, correction_reason, status,
                              corrected_label, model_label, tenant_id
                       FROM decision_feedback
                       ORDER BY reviewed_at ASC LIMIT %s""",
                    (limit,)).fetchall()
        out = []
        for d in rows:
            out.append({
                "request_id": d["request_id"], "reviewer": d["reviewer"],
                "reviewed_at": _dt_to_epoch(d["reviewed_at"]),
                "review_action": d["review_action"],
                "correction_reason": d["correction_reason"],
                "status": d["status"],
                "corrected_label": d["corrected_label"],
                "model_label": d["model_label"],
                "tenant_id": d["tenant_id"],
            })
        return out

    # ---- 策略注册表（decision_policies：draft/active/retired 指针）----
    def register_policy(self, version: str, snapshot: dict) -> None:
        with self._lock:
            with self._conn() as conn:
                conn.execute(
                    """INSERT INTO decision_policies
                       (policy_id, task_id, version, spec_jsonb, status)
                       VALUES (%s, %s, %s, %s, 'active')
                       ON CONFLICT (policy_id, version) DO NOTHING""",
                    (self._task_id, self._task_id, version, Jsonb(snapshot)))
            self._policy_cache[version] = snapshot
            self.set_current_policy_version(version)

    def set_current_policy_version(self, v: str) -> None:
        with self._lock:
            with self._conn() as conn:
                conn.execute(
                    "UPDATE decision_policies SET status='retired', effective_at=NULL "
                    "WHERE policy_id = %s AND status = 'active'",
                    (self._task_id,))
                cur = conn.execute(
                    "UPDATE decision_policies SET status='active', effective_at=now() "
                    "WHERE policy_id = %s AND version = %s",
                    (self._task_id, v))
                if cur.rowcount == 0:
                    raise ValueError(f"策略版本 {v} 未登记，无法切换")

    def policy_versions(self) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT version, spec_jsonb AS snapshot, created_at,
                          (status = 'active') AS active
                   FROM decision_policies WHERE policy_id = %s
                   ORDER BY created_at ASC""",
                (self._task_id,)).fetchall()
        return [{"version": d["version"], "snapshot": d["snapshot"],
                 "ts": _dt_to_epoch(d["created_at"]), "active": bool(d["active"])}
                for d in rows]

    def rollback_policy(self, target_version: str) -> str:
        with self._lock:
            with self._conn() as conn:
                d = conn.execute(
                    "SELECT version FROM decision_policies "
                    "WHERE policy_id = %s AND version = %s",
                    (self._task_id, target_version)).fetchone()
                if not d:
                    raise ValueError(f"策略版本 {target_version} 未登记，无法回滚")
            self.set_current_policy_version(target_version)
            return target_version

    def current_policy_version(self) -> str:
        with self._conn() as conn:
            d = conn.execute(
                "SELECT version FROM decision_policies "
                "WHERE policy_id = %s AND status = 'active' "
                "ORDER BY effective_at DESC NULLS LAST LIMIT 1",
                (self._task_id,)).fetchone()
        return d["version"] if d else "none"

    # ---- 模型注册表（decision_models：production 指针）----
    def set_current_model_version(self, v: str) -> None:
        """v 形如 "<model_id>:<data_snapshot_hash[:16]>"（inference.load_bundle 生成）。"""
        name, _, snap = v.partition(":")
        model_id = name or v
        with self._lock:
            with self._conn() as conn:
                conn.execute(
                    "UPDATE decision_models SET status='retired' "
                    "WHERE model_id = %s AND status = 'production' AND version <> %s",
                    (model_id, v))
                conn.execute(
                    """INSERT INTO decision_models
                       (model_id, task_id, version, data_snapshot_hash, status, notes)
                       VALUES (%s, %s, %s, %s, 'production', 'decision-service 热加载')
                       ON CONFLICT (model_id, version)
                       DO UPDATE SET status = 'production'""",
                    (model_id, self._task_id, v, snap or None))

    def current_model_version(self) -> str:
        with self._conn() as conn:
            d = conn.execute(
                "SELECT version FROM decision_models WHERE status = 'production' "
                "ORDER BY created_at DESC LIMIT 1").fetchone()
        return d["version"] if d else "none"

    def model_versions(self) -> list[str]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT version FROM decision_models ORDER BY created_at ASC").fetchall()
        return [d["version"] for d in rows]
