# Scripts

## Import BERT matmul results into Neon

This script imports an existing results JSON file into the profile-specific table (for example, `i5_1235u_bert_matmul_results`) and skips rows that already exist.

### Requirements

- For `api` mode (default): run the data_aggregator server and set `DATA_AGGREGATOR_URL` if needed
- For `direct` mode: `DATABASE_URL` set in the environment, or present in `services/data_aggregator/.env`
- Python dependency: `psycopg[binary]` (direct mode only)

### Usage

```sh
python3 scripts/import_bert_matmul_results.py --dry-run
python3 scripts/import_bert_matmul_results.py
python3 scripts/import_bert_matmul_results.py --mode=direct
```

Options:
- `--mode` to choose `api` (default) or `direct` DB connection
- `--api-url` to override `DATA_AGGREGATOR_URL`
- `--profile` to override `DATA_AGGREGATOR_PROFILE` (also selects the target table in `direct` mode)
- `--no-dedupe` to disable dedupe when using `api` mode
- `--timeout` to set API upload timeout (seconds)
- `--file` to point at a different results file
- `--db-url` to override `DATABASE_URL`
- `--chunk-size` to control batch insert size
- `--sslmode` to override libpq SSL mode (example: `require`)
- `--sslrootcert` to specify root cert path or `system` for OS trust store

### Notes

- Naive timestamps (no timezone offset) are interpreted as IST (Asia/Kolkata).
- Duplicates are detected by comparing all columns except `id` and `ingested_at`.
- For Neon, SSL is required in `direct` mode; if your URL has no `sslmode`, the script defaults to `sslmode=require`.
- API uploads default to 300 seconds for this script (override with `--timeout`).
- Regular uploads (from `qkv_mlp_run` or `metaschedule_best_schedules`) default to 10 seconds via `DATA_AGGREGATOR_TIMEOUT`.
 - Direct mode uses the sanitized profile table name: lowercase, non-alphanumerics replaced by `_`, and a leading digit is prefixed with `p`.
