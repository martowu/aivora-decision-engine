-- ============================================================================
-- Aivora Decision Engine · 一期五表 DDL（定稿 §2.3 / v1.0 §2.2.8）
-- 0001_init.sql —— Postgres 15
--
-- 两条硬约束（定稿 §2.3）：
--   1. decision_logs.request_id 为幂等唯一键（服务层 409 重放首次结果）
--   2. decision_models 带 data_snapshot_hash（锚定 §7-C 可重跑数据管线）
--
-- MVP 调整（均为超集，不破坏定稿；逐处注释标出）：
--   - decision_logs 增加 decision_id / input_digest（服务实现字段，审计追溯）
--   - decision_feedback 增加 corrected_label / model_label / tenant_id
--     （定稿 DDL 摘要省略；§3-8 闭环"仅 approved 修正进训练集"需要纠正标签）
--   - decision_models.artifact_uri 可空（当前服务从挂载卷加载工件；
--     MinIO 接线后改 NOT NULL）
-- ============================================================================

BEGIN;

-- ---- 任务注册表（任务定义 = Schema + 标签规范引用）----
CREATE TABLE IF NOT EXISTS decision_tasks (
  task_id        text NOT NULL,
  version        text NOT NULL,
  spec_jsonb     jsonb NOT NULL,        -- 输入/输出 Schema、标签规范引用
  label_spec_uri text,                  -- MinIO: 标签规范文档
  created_at     timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (task_id, version)
);

-- ---- 模型注册表（模型 = 特征函数 + 训练配置 + 工件 + 校准器，定稿 §1.2）----
CREATE TABLE IF NOT EXISTS decision_models (
  model_id   text NOT NULL,
  task_id    text NOT NULL,
  version    text NOT NULL,
  artifact_uri text,                    -- MVP 调整：可空（MinIO 接线后 NOT NULL）
  data_snapshot_hash text,              -- 硬约束 2：训练数据快照指纹（§7-C）
  metrics_jsonb jsonb,                  -- F1/AUC/ECE/Brier + 切片
  status     text NOT NULL DEFAULT 'draft'
             CHECK (status IN ('draft','canary','production','retired')),
  created_by text,
  notes      text,
  created_at timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (model_id, version)
);

-- ---- 策略注册表（策略独立版本、独立回滚，定稿 §1.3.1）----
CREATE TABLE IF NOT EXISTS decision_policies (
  policy_id   text NOT NULL,
  task_id     text NOT NULL,
  version     text NOT NULL,
  spec_jsonb  jsonb NOT NULL,           -- 策略快照（YAML 序列化：rules + default_action）
  status      text NOT NULL DEFAULT 'draft'
             CHECK (status IN ('draft','active','retired')),
  effective_at timestamptz,
  created_at  timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (policy_id, version)
);

-- ---- 决策日志（全量落库可追溯，定稿 §2.2.2）----
CREATE TABLE IF NOT EXISTS decision_logs (
  request_id    text PRIMARY KEY,       -- 硬约束 1：幂等唯一键
  decision_id   text,                   -- MVP 调整：服务内部决策号（DEC-…）
  tenant_id     text NOT NULL,          -- 行级租户隔离（一期单租户，字段贯穿）
  task_id       text NOT NULL DEFAULT 'recon.transaction_match',
  task_version  text NOT NULL DEFAULT '1.0',
  model_version text,
  policy_version text,
  feature_version text,
  mode          text NOT NULL CHECK (mode IN ('online','shadow','replay')),
  input_digest  text NOT NULL,          -- 输入摘要 sha256（明细按脱敏策略存 input_uri/MinIO）
  prediction    jsonb,                  -- {label, conf, payment_id, invoice_id, decision_id}
  calibrated_prob jsonb,                -- {MATCHED, PARTIAL, UNMATCHED} 校准概率
  action        text,                   -- 真实决策 AUTO/REVIEW（shadow 期也记 AUTO，对外掩码在服务层）
  reason_codes  jsonb,                  -- [policy_reason]
  rules_triggered jsonb,                -- [policy_rule]
  latency_ms    int,                    -- 推理+策略求值耗时（G-Shadow P95≤100ms 监控源）
  degraded      bool NOT NULL DEFAULT false,
  created_at    timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_decision_logs_tenant_task_time
  ON decision_logs (tenant_id, task_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_decision_logs_mode_created
  ON decision_logs (mode, created_at DESC);

-- ---- 人工复核回写（闭环：pending → approved/rejected，§3-8 仅 approved 进训练集）----
CREATE TABLE IF NOT EXISTS decision_feedback (
  decision_log_id text NOT NULL REFERENCES decision_logs(request_id),
  reviewer        text NOT NULL,
  reviewed_at     timestamptz NOT NULL DEFAULT now(),
  review_action   text,                 -- 维持 / 纠正说明
  correction_reason text,
  status          text NOT NULL DEFAULT 'pending'
                 CHECK (status IN ('pending','approved','rejected')),
  corrected_label text,                 -- MVP 调整：纠正后的三态（MATCHED/PARTIAL/UNMATCHED）
  model_label     text,                 -- MVP 调整：模型当时 top-1（复核对照基线）
  tenant_id       text,                 -- MVP 调整：租户贯穿
  PRIMARY KEY (decision_log_id, reviewer)
);
CREATE INDEX IF NOT EXISTS idx_decision_feedback_status
  ON decision_feedback (status, reviewed_at DESC);

-- ---- 首任务注册（recon.transaction_match@1.0，定稿 §3）----
INSERT INTO decision_tasks (task_id, version, spec_jsonb, label_spec_uri)
VALUES (
  'recon.transaction_match', '1.0',
  jsonb_build_object(
    'type', 'choice+rank',
    'labels', jsonb_build_array('MATCHED','PARTIAL','UNMATCHED'),
    'actions', jsonb_build_array('AUTO_RECONCILE','REVIEW','REJECT'),
    'review_semantics', 'REVIEW 是策略动作不是训练标签（定稿 §1.3.4）',
    'feature_version', '1.0',
    'candidate_limit', 50
  ),
  NULL
)
ON CONFLICT (task_id, version) DO NOTHING;

COMMIT;
