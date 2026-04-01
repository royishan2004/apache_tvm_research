CREATE TABLE "bert_matmul_results" (
	"id" uuid PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"ingested_at" timestamp with time zone DEFAULT now() NOT NULL,
	"ts" timestamp with time zone NOT NULL,
	"kernel" text NOT NULL,
	"variant" text NOT NULL,
	"target" text NOT NULL,
	"source" text DEFAULT '' NOT NULL,
	"m" integer NOT NULL,
	"k" integer NOT NULL,
	"n" integer NOT NULL,
	"latency_us" double precision NOT NULL,
	"std_us" double precision NOT NULL,
	"number" integer,
	"repeat" integer,
	"min_repeat_ms" integer,
	"iteration" integer,
	"total_iterations" integer
);
--> statement-breakpoint
CREATE INDEX "idx_kernel_variant_ts" ON "bert_matmul_results" USING btree ("kernel","variant","ts");--> statement-breakpoint
CREATE INDEX "idx_shape" ON "bert_matmul_results" USING btree ("m","k","n");
--> statement-breakpoint
-- Enable TimescaleDB extension (idempotent)
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;
--> statement-breakpoint
-- Convert bert_matmul_results into a hypertable partitioned by ts
-- chunk_time_interval: 7 days is reasonable for benchmark runs
SELECT create_hypertable(
  'bert_matmul_results',
  'ts',
  chunk_time_interval => INTERVAL '7 days'
);
--> statement-breakpoint
-- Optional: compress chunks older than 30 days
ALTER TABLE bert_matmul_results SET (
  timescaledb.compress,
  timescaledb.compress_segmentby = 'kernel, variant, target'
);
--> statement-breakpoint
SELECT add_compression_policy('bert_matmul_results', INTERVAL '30 days');