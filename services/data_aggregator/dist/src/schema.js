import { pgTable, uuid, text, integer, boolean, doublePrecision, timestamp, index, uniqueIndex, jsonb, } from "drizzle-orm/pg-core";
import { sql } from "drizzle-orm";
export const bertMatmulResults = pgTable("bert_matmul_results", {
    // identity
    id: uuid("id").primaryKey().default(sql `gen_random_uuid()`),
    ingestedAt: timestamp("ingested_at", { withTimezone: true }).defaultNow().notNull(),
    // time dimension — TimescaleDB partitions on this
    ts: timestamp("ts", { withTimezone: true }).notNull(),
    // benchmark dimensions
    kernel: text("kernel").notNull(),
    variant: text("variant").notNull(),
    target: text("target").notNull(),
    source: text("source").default("").notNull(), // "MetaSchedule-db" or ""
    // matrix shape
    m: integer("m").notNull(),
    k: integer("k").notNull(),
    n: integer("n").notNull(),
    // timing
    latencyUs: doublePrecision("latency_us").notNull(),
    stdUs: doublePrecision("std_us").notNull(),
    // run metadata — nullable because MetaSchedule records omit these
    number: integer("number"),
    repeat: integer("repeat"),
    minRepeatMs: integer("min_repeat_ms"),
    iteration: integer("iteration"),
    totalIterations: integer("total_iterations"),
}, (table) => [
    // your most common query pattern: filter by kernel+variant, order by time
    index("idx_kernel_variant_ts").on(table.kernel, table.variant, table.ts),
    // shape-based lookups
    index("idx_shape").on(table.m, table.k, table.n),
]);
export const bestSchedules = pgTable("best_schedules", {
    id: uuid("id").primaryKey().default(sql `gen_random_uuid()`),
    ingestedAt: timestamp("ingested_at", { withTimezone: true }).defaultNow().notNull(),
    kernel: text("kernel").notNull(),
    m: integer("m").notNull(),
    k: integer("k").notNull(),
    n: integer("n").notNull(),
    latencyUs: doublePrecision("latency_us").notNull(),
    stdUs: doublePrecision("std_us").notNull(),
    trace: text("trace").notNull(),
    decisions: jsonb("decisions").$type().notNull().default(sql `'[]'::jsonb`),
}, (table) => [
    index("idx_best_schedules_kernel_shape").on(table.kernel, table.m, table.k, table.n),
    index("idx_best_schedules_latency").on(table.latencyUs),
]);
export const bestPrunedConfig = pgTable("best_pruned_config", {
    id: uuid("id").primaryKey().default(sql `gen_random_uuid()`),
    ingestedAt: timestamp("ingested_at", { withTimezone: true }).defaultNow().notNull(),
    ts: timestamp("ts", { withTimezone: true }).notNull(),
    target: text("target").notNull().default(""),
    selectedConfigName: text("selected_config_name").notNull(),
    selectedStateToken: text("selected_state_token").notNull().default(""),
    selectionReason: text("selection_reason").notNull().default(""),
    latencyRetention: doublePrecision("latency_retention"),
    timeReduction: doublePrecision("time_reduction"),
    trialReduction: doublePrecision("trial_reduction"),
    score: doublePrecision("score"),
    payloadHash: text("payload_hash").notNull(),
    payload: jsonb("payload").$type().notNull(),
}, (table) => [
    index("idx_best_pruned_config_ts").on(table.ts),
    index("idx_best_pruned_config_name").on(table.selectedConfigName),
    index("idx_best_pruned_config_score").on(table.score),
    uniqueIndex("uniq_best_pruned_config_payload_hash").on(table.payloadHash),
]);
export const pruningExperiments = pgTable("pruning_experiments", {
    id: uuid("id").primaryKey().default(sql `gen_random_uuid()`),
    ingestedAt: timestamp("ingested_at", { withTimezone: true }).defaultNow().notNull(),
    runId: text("run_id").notNull(),
    ts: timestamp("ts", { withTimezone: true }).notNull(),
    mode: text("mode").notNull(),
    iteration: integer("iteration").notNull(),
    configName: text("config_name").notNull(),
    configHash: text("config_hash").notNull(),
    tasksSignature: text("tasks_signature").notNull().default(""),
    isBaseline: boolean("is_baseline").notNull().default(false),
    benchmarkOnly: boolean("benchmark_only").notNull().default(false),
    numTasks: integer("num_tasks"),
    numSuccessfulTasks: integer("num_successful_tasks"),
    allTasksSucceeded: boolean("all_tasks_succeeded"),
    latencyGeomeanUs: doublePrecision("latency_geomean_us"),
    totalTuningTimeSec: doublePrecision("total_tuning_time_sec"),
    totalTrials: integer("total_trials"),
    latencyRetention: doublePrecision("latency_retention"),
    timeReduction: doublePrecision("time_reduction"),
    trialReduction: doublePrecision("trial_reduction"),
    score: doublePrecision("score"),
    metadata: jsonb("metadata").$type().notNull().default(sql `'{}'::jsonb`),
    latestPruningRun: jsonb("latest_pruning_run").$type().notNull().default(sql `'{}'::jsonb`),
    experiment: jsonb("experiment").$type().notNull(),
}, (table) => [
    uniqueIndex("uniq_pruning_experiments_run_id").on(table.runId),
    index("idx_pruning_experiments_ts").on(table.ts),
    index("idx_pruning_experiments_cfg_iter").on(table.configName, table.iteration),
    index("idx_pruning_experiments_score").on(table.score),
]);
export const comparisonResults = pgTable("comp_summary", {
    id: uuid("id").primaryKey().default(sql `gen_random_uuid()`),
    ingestedAt: timestamp("ingested_at", { withTimezone: true }).defaultNow().notNull(),
    updatedAt: timestamp("updated_at", { withTimezone: true }).defaultNow().notNull(),
    rowLabel: text("row_label").notNull(),
    rowOrder: integer("row_order").notNull(),
    rowKind: text("row_kind").notNull(),
    compareId: text("compare_id").notNull(),
    compareTs: timestamp("compare_ts", { withTimezone: true }).notNull(),
    mode: text("mode").notNull(),
    shape: text("shape").notNull(),
    numShapes: integer("num_shapes"),
    baselineLatencyUs: doublePrecision("baseline_latency_us"),
    candidateLatencyUs: doublePrecision("candidate_latency_us"),
    baselineTaskTuneTime: text("baseline_task_tune_time").notNull().default(""),
    candidateTaskTuneTime: text("candidate_task_tune_time").notNull().default(""),
    baselineTaskTuneTimeSec: doublePrecision("baseline_task_tune_time_sec"),
    candidateTaskTuneTimeSec: doublePrecision("candidate_task_tune_time_sec"),
    latencyRetention: doublePrecision("latency_retention"),
    execTimeReduction: doublePrecision("exec_time_reduction"),
    rowPayload: jsonb("row_payload").$type().notNull().default(sql `'{}'::jsonb`),
}, (table) => [
    uniqueIndex("uniq_comp_summary_row_label").on(table.rowLabel),
    index("idx_comp_summary_row_order").on(table.rowOrder),
    index("idx_comp_summary_compare_ts").on(table.compareTs),
]);
