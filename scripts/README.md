# Scripts

## Import benchmark results into Neon

This script imports benchmark JSON files into profile-specific tables and skips rows that already exist.

Supported commands:
- `bert-matmul` -> imports `research/results/bert_matmul_results.json` into `*_bert_matmul_results`
- `best-schedules` -> imports `research/results/metaschedule/best_schedules.json` into `*_best_schedules`

### Requirements

- For `api` mode (default): run the data_aggregator server and set `DATA_AGGREGATOR_URL` if needed
- For `direct` mode: `DATABASE_URL` set in the environment, or present in `services/data_aggregator/.env`
- Python dependency: `psycopg[binary]` (direct mode only)

### Usage

```sh
python3 scripts/import_bert_matmul_results.py bert-matmul --dry-run
python3 scripts/import_bert_matmul_results.py bert-matmul
python3 scripts/import_bert_matmul_results.py best-schedules
python3 scripts/import_bert_matmul_results.py best-schedules --mode=direct
```

Backward compatible shortcut:

```sh
python3 scripts/import_bert_matmul_results.py --dry-run
```

This defaults to the `bert-matmul` command.

Options:
- `bert-matmul` or `best-schedules` command to select which JSON artifact to import
- `--mode` to choose `api` (default) or `direct` DB connection
- `--api-url` to override `DATA_AGGREGATOR_URL`
- `best-schedules` command uses `DATA_AGGREGATOR_BEST_SCHEDULES_URL` by default
- `--profile` to override `DATA_AGGREGATOR_PROFILE` (also selects the target table in `direct` mode)
- `--no-dedupe` to disable dedupe checks
- `--timeout` to set API upload timeout (seconds)
- `--file` to point at a different results file
- `--db-url` to override `DATABASE_URL`
- `--chunk-size` to control batch insert size
- `--sslmode` to override libpq SSL mode (example: `require`)
- `--sslrootcert` to specify root cert path or `system` for OS trust store

### Notes

- Naive timestamps (no timezone offset) are interpreted as IST (Asia/Kolkata).
- Duplicates are detected by comparing all content columns (excluding generated `id` and `ingested_at`).
- For Neon, SSL is required in `direct` mode; if your URL has no `sslmode`, the script defaults to `sslmode=require`.
- API uploads default to 300 seconds for this script (override with `--timeout`).
- Regular uploads (from `qkv_mlp_run` or `metaschedule_best_schedules`) default to 10 seconds via `DATA_AGGREGATOR_TIMEOUT`.
 - Direct mode uses the sanitized profile table name: lowercase, non-alphanumerics replaced by `_`, and a leading digit is prefixed with `p`.
