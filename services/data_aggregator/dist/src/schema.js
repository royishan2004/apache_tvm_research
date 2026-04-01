import { pgTable, uuid, text, integer, doublePrecision, timestamp, index, } from "drizzle-orm/pg-core";
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
