# training/data — 数据抽取 / 脱敏 / 质检管线

对应《最佳方案（定稿）》§7-C：**数据管线必须可重跑**。

## 文件

| 文件 | 作用 |
|---|---|
| `pipe.py` | 管线核心：`RunConfig`（参数）、脱敏（`mask_record`）、质检（`quality_check`）、快照 hash（`compute_snapshot_hash`）、输出（`write_output`） |
| `extract.py` | CLI 入口：抽取 → 脱敏 → 质检 → 输出 CSV/parquet + `snapshot.json`（含 `data_snapshot_hash`） |
| `test_repro.py` | **可重跑性验收测试**：同参重跑 hash/字节一致、参数变化 hash 变化、脱敏抽查 |

## 用法

```bash
# 1. 盐值从密钥管理读取，不进 git、不进输出
printf '%s' "$SALT" > /secure/salt.txt

# 2. 抽取（synthetic 仅用于验证管线；生产接 recon source）
python3 extract.py \
  --start 2026-08-01 --end 2026-10-31 \
  --tenant t-demo \
  --source synthetic --n 50 \
  --salt-file /secure/salt.txt \
  --output ../out

# 3. 验证可重跑性
python3 test_repro.py
```

## 产出

`<output>/` 下：`payments.csv`、`invoices.csv`、`labels.csv`、**`snapshot.json`**（含 `data_snapshot_hash`、运行指纹、质检结果、文件清单）。

`data_snapshot_hash` 锚定 `decision_models.data_snapshot_hash` 字段：训练时记录它，P2 月度重训用同参数重跑，hash 一致 = 数据未漂移。

## 红线（不可违反）

1. **数据不进 git** —— 本目录只提交脚本与配置；`../out/`、`*.csv`、盐值文件全部 git-ignore（见 `.gitignore`）。
2. **合成数据只验证 pipeline，不得用于模型验收**（定稿 §1-D2）。`--source synthetic` 的产出不许进 G-数据/G-离线门禁的样本计数。
3. **脱敏固化在脚本内**：主体 ID 单向假名化（`p_<sha256[:16]>`）、金额保留、文本截断、`SENSITIVE_FIELDS` 中 `drop` 字段（银行账号/邮箱/电话）不落库。改脱敏逻辑必须 `PIPELINE_VERSION +0.1` 并过评审。
4. **可逆映射默认关闭**（`keep_pseudonym_map=False`）。若审计需要可逆，映射表与脱敏数据**分开放置、不同权限**，不得混在同一输出目录。
5. **盐值不进输出、不进 git、不进日志**。`run_fingerprint` 只含盐值的 8 位 hash，用于检测盐值误改导致的假名漂移。

## 生产接入点

`DataSource` 是适配器接口。当前只有 `SyntheticSource`。接真实 Recon 时实现 `ReconSource(DataSource)`：
- `fetch_payments` / `fetch_candidate_invoices`：join Recon Postgres 的付款与发票表（按 `tenant_id`、`doc_type`、时间窗过滤）；
- `fetch_label_source`：join 核销结果表，映射 `human_review / historical_confirmed / rule_inferred / None`（定稿 §3-8 标签源规则）。

实现后在 `extract.py::build_source` 注册 `--source recon` 即可，其余管线（脱敏/质检/hash）零改动。

## 质检口径

`quality_check` 一期硬门槛：`DUPLICATE_PK`（主键重复）与 `NULL_CRITICAL`（关键字段 amount/date/主体 id 空值）任一非零 → 退出码 2，不产出。其余问题记录在 `snapshot.json.quality` 供人工判断。
