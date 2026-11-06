"""Aivora Decision Engine — 数据抽取/脱敏/质检管线（定稿 §7-C）。

设计约束：
- 参数化：时间窗 / 租户 / 输出路径，全部来自 RunConfig，不允许硬编码；
- 可重跑：同参数重跑必须产出相同的 data_snapshot_hash（行排序 + 稳定序列化 + 确定性假名化）；
- 脱敏固化在脚本内：主体 ID 假名化（加盐 SHA-256 截断，映射表单独加密存放，默认不落输出）；
  金额保留；文本截断；敏感字段按 SENSITIVE_FIELDS 处理；
- 数据本身不进 git（输出目录默认 git-ignored）。
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

PIPELINE_VERSION = "1.0"  # 管线代码版本；改动脱敏/排序/序列化逻辑时必须 +0.1


# ---------------------------------------------------------------------------
# 运行配置（参数化的唯一入口）
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RunConfig:
    """一次抽取运行的全部参数。重跑 = 构造出完全相同的 RunConfig。"""

    # 范围
    time_start: date                      # 抽取时间窗（含）
    time_end: date                        # （含）
    tenant_ids: tuple[str, ...]           # 一期默认单租户
    doc_types: tuple[str, ...]            # AP=应付 AR=应收，一期对准 PoC 场景
    masking_salt: str                     # 假名化盐值；从密钥管理读取，不进 git、不进输出

    # 输出
    output_dir: Path                      # 输出目录（git-ignored）
    format: str = "csv"                   # csv | parquet

    # 脱敏选项
    text_max_len: int = 200               # 文本字段截断长度（定稿：文本截断）
    keep_pseudonym_map: bool = False      # 是否输出 id 映射表（默认否；需要时单独加密交付）

    def fingerprint(self) -> str:
        """运行参数指纹（不含盐值本身，含盐值 hash，防止盐值误改导致假名漂移）。"""
        names = [f.name for f in dataclasses.fields(self)]
        payload = {
            f: getattr(self, f)
            for f in names
            if f not in ("masking_salt", "output_dir", "format")
        }
        payload["masking_salt_sha256_8"] = sha256_hex(self.masking_salt)[:8]
        payload["pipeline_version"] = PIPELINE_VERSION
        return sha256_hex(json.dumps(payload, sort_keys=True, default=str))


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------

def sha256_hex(s: str | bytes) -> str:
    if isinstance(s, str):
        s = s.encode("utf-8")
    return hashlib.sha256(s).hexdigest()


def pseudonym(value: str, salt: str, length: int = 16) -> str:
    """确定性假名化：同一 (value, salt) 永远得到同一假名。

    注意：这是**单向**假名化（不可逆），满足一期"主体 ID 假名化"要求。
    若业务需要可逆映射（如审计回溯），用 keep_pseudonym_map + 加密映射表，
    映射表与脱敏数据分开放置、不同权限。
    """
    if not value:
        return ""
    return "p_" + sha256_hex(f"{salt}::{value}")[:length]


_SENSITIVE_TEXT_RE = re.compile(r"[\u4e00-\u9fffA-Za-z0-9]+")


def truncate_text(value: str | None, max_len: int) -> str | None:
    """文本截断（定稿脱敏策略）。截断位置不切半个词（中文按字符、英文按词边界回退）。"""
    if value is None:
        return None
    v = value.strip()
    if len(v) <= max_len:
        return v
    cut = v[:max_len]
    # 英文单词边界回退
    m = _SENSITIVE_TEXT_RE.findall(cut)
    if m and len(m[-1]) < 40 and max_len - len(cut) >= 0:
        last = m[-1]
        if cut.endswith(last) and len(last) > 3:
            # 若截断点落在一个长 token 中间，回退到该 token 起点
            start = cut.rfind(last)
            if start > max_len - 30:
                cut = cut[:start]
    return cut.rstrip() + "…"


# 敏感字段处理策略：mask=假名化（id 类）/ truncate=截断（文本类）/ drop=不落库
# 金额、日期、币种等决策必需字段保留（定稿 §4 脱敏方案）
SENSITIVE_FIELDS: Mapping[str, str] = {
    "payer_id": "mask",
    "supplier_id": "mask",
    "customer_id": "mask",
    "reference": "truncate",     # 付款附言
    "note": "truncate",          # 发票备注
    "supplier_name": "truncate", # 名称按文本截断（假名化会破坏模糊匹配特征的可读性，
    # 但一期 supplier_fuzzy_score 特征基于假名后的名称计算，保持口径一致即可）
    "bank_account": "drop",
    "contact_email": "drop",
    "contact_phone": "drop",
}


def mask_record(record: dict[str, Any], cfg: RunConfig) -> dict[str, Any]:
    """对一条原始记录执行脱敏。返回新 dict，不修改输入。"""
    out: dict[str, Any] = {}
    for k, v in record.items():
        policy = SENSITIVE_FIELDS.get(k, "keep")
        if policy == "drop":
            continue
        if policy == "mask":
            out[k] = pseudonym(str(v), cfg.masking_salt) if v is not None else None
        elif policy == "truncate":
            out[k] = truncate_text(v, cfg.text_max_len) if isinstance(v, str) else v
        else:
            out[k] = v
    return out


# ---------------------------------------------------------------------------
# 数据源适配（Recon 真实表结构接入点）
# ---------------------------------------------------------------------------

class DataSource:
    """数据源适配器接口。生产实现连 Recon Postgres；测试/联调用 SyntheticSource。"""

    name: str = "abstract"

    def fetch_payments(self, cfg: RunConfig) -> Iterable[Mapping[str, Any]]:
        raise NotImplementedError

    def fetch_candidate_invoices(
        self, cfg: RunConfig, payment_ids: Sequence[str]
    ) -> Iterable[Mapping[str, Any]]:
        raise NotImplementedError

    def fetch_label_source(
        self, cfg: RunConfig, payment_id: str
    ) -> str | None:
        """返回该付款的标签来源：human_review / historical_confirmed / rule_inferred / None。

        一期规则（定稿 §3-8）：只有 human_review 直接作 gold；
        historical_confirmed 需抽检 5% 后使用；rule_inferred 仅 baseline。
        真实实现：join 核销结果表，人工确认记录 → human_review，
        系统自动核销 → rule_inferred，历史确认匹配关系 → historical_confirmed。
        """
        raise NotImplementedError


class SyntheticSource(DataSource):
    """合成数据源：**仅用于验证管线（跑通、hash 稳定、脱敏正确），
    不得用于模型验收（定稿 §1-D2 规则）。**"""

    name = "synthetic"

    def __init__(self, n: int = 50, seed: int = 42):
        self._n = n
        self._seed = seed

    def _rng(self):
        import random
        return random.Random(self._seed)

    def fetch_payments(self, cfg: RunConfig):
        rng = self._rng()
        for i in range(self._n):
            yield {
                "payment_id": f"PAY-{i:06d}",
                "tenant_id": cfg.tenant_ids[0],
                "doc_type": cfg.doc_types[0],
                "amount": round(rng.uniform(1_000, 500_000), 2),
                "currency": "CNY",
                "date": cfg.time_start.isoformat(),
                "payer_id": f"CP-{rng.randint(1, 20):03d}",
                "channel": rng.choice(["BANK_TRANSFER", "CHECK", "ONLINE"]),
                "reference": f"付款 发票 INV-{rng.randint(1, 30):06d} 第{'一' if i % 2 else '二'}期" + "说明" * 80,
            }

    def fetch_candidate_invoices(self, cfg: RunConfig, payment_ids):
        rng = self._rng()
        for pid in payment_ids:
            n_inv = rng.randint(1, 3)
            for j in range(n_inv):
                yield {
                    "invoice_id": f"INV-{pid}-{j}",
                    "payment_id": pid,
                    "tenant_id": cfg.tenant_ids[0],
                    "amount": round(rng.uniform(1_000, 500_000), 2),
                    "currency": rng.choice(["CNY", "CNY", "USD"]),
                    "date": cfg.time_start.isoformat(),
                    "supplier_id": f"SP-{rng.randint(1, 20):03d}",
                    "supplier_name": "某某供应链有限公司" * 3,
                    "note": "备注" * 150,
                }

    def fetch_label_source(self, cfg: RunConfig, payment_id: str) -> str | None:
        import random
        r = random.Random(payment_id).random()
        if r < 0.5:
            return "human_review"
        if r < 0.8:
            return "historical_confirmed"
        if r < 0.95:
            return "rule_inferred"
        return None


# ---------------------------------------------------------------------------
# 质检（对应 W1 数据盘点报告 §3 的数据质量检查项）
# ---------------------------------------------------------------------------

@dataclass
class QualityReport:
    total: int = 0
    issues: list[dict[str, Any]] = field(default_factory=list)

    def add(self, code: str, count: int, detail: str = ""):
        if count:
            self.issues.append({"code": code, "count": count, "detail": detail})

    @property
    def passed(self) -> bool:
        """一期硬门槛：主键唯一 + 关键字段非空。其余问题记录不阻塞。"""
        return not any(i["code"] in ("DUPLICATE_PK", "NULL_CRITICAL") for i in self.issues)

    def to_dict(self) -> dict:
        return {"passed": self.passed, "total": self.total, "issues": self.issues}


def quality_check(rows: Sequence[Mapping[str, Any]], pk: str,
                  critical: Sequence[str]) -> QualityReport:
    rep = QualityReport(total=len(rows))
    seen: set[str] = set()
    dup = 0
    null_crit = 0
    for r in rows:
        key = str(r.get(pk))
        if key in seen:
            dup += 1
        seen.add(key)
        if any(r.get(c) in (None, "") for c in critical):
            null_crit += 1
    rep.add("DUPLICATE_PK", dup, f"{pk} 重复")
    rep.add("NULL_CRITICAL", null_crit, f"关键字段空值: {list(critical)}")
    return rep


# ---------------------------------------------------------------------------
# 快照 hash（可重跑的核心保证）
# ---------------------------------------------------------------------------

def canonical_rows(rows: Sequence[Mapping[str, Any]], pk: str) -> list[dict[str, Any]]:
    """按主键排序 + 键排序的稳定化，保证序列化字节级确定。"""
    return [dict(sorted(r.items())) for r in sorted(rows, key=lambda r: str(r.get(pk)))]


def compute_snapshot_hash(rows: Sequence[Mapping[str, Any]], pk: str,
                          cfg: RunConfig) -> str:
    """data_snapshot_hash = sha256( 管线版本 ‖ 运行指纹 ‖ 稳定化后的全部脱敏行 )。

    锚定 decision_models.data_snapshot_hash（定稿 §2.3）：
    训练时记录该 hash，P2 月度重训用同参数重跑管线，hash 一致即数据未漂移。
    """
    h = hashlib.sha256()
    h.update(f"pipe:{PIPELINE_VERSION}\n".encode())
    h.update(f"run:{cfg.fingerprint()}\n".encode())
    for r in canonical_rows(rows, pk):
        h.update((json.dumps(r, sort_keys=True, ensure_ascii=False, default=str) + "\n").encode())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# 输出
# ---------------------------------------------------------------------------

def write_output(rows: Sequence[Mapping[str, Any]], out_dir: Path,
                 name: str, cfg: RunConfig, pk: str = "payment_id") -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    if cfg.format == "csv":
        path = out_dir / f"{name}.csv"
        if not rows:
            path.write_text("")
            return path
        import csv
        cols: list[str] = []
        for r in rows:
            for k in r:
                if k not in cols:
                    cols.append(k)
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=sorted(cols))
            w.writeheader()
            for r in canonical_rows(rows, pk):
                w.writerow(r)
        return path
    elif cfg.format == "parquet":
        import pandas as pd
        path = out_dir / f"{name}.parquet"
        pd.DataFrame(canonical_rows(rows, pk)).to_parquet(path, index=False)
        return path
    else:
        raise ValueError(f"unsupported format: {cfg.format}")
