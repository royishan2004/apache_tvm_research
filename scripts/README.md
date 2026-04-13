# Scripts

## Import benchmark results into Neon

This script imports benchmark JSON files into profile-specific tables and skips rows that already exist.

Supported commands:
- `bert-matmul` -> imports `research/results/bert_matmul_results.json` into `*_bert_matmul_results`
- `best-schedules` -> imports `research/results/metaschedule/best_schedules.json` into `*_best_schedules`
- `best-pruned-config` -> imports archived `research/archive/metaschedule_8020/results/metaschedule/best_pruned_config.json` into `*_best_pruned_config`
- `pruning-experiments` -> imports archived `research/archive/metaschedule_8020/results/metaschedule/pruning_experiments.json` into `*_pruning_experiments`
- `comparison-results` -> imports archived `research/archive/metaschedule_8020/results/metaschedule/comparison_results.json` into `*_comp_summary`

### Requirements

- For `api` mode (default): run the data_aggregator server and set `DATA_AGGREGATOR_URL` if needed
- For `direct` mode: `DATABASE_URL` set in the environment, or present in `services/data_aggregator/.env`
- Python dependency: `psycopg[binary]` (direct mode only)

### Usage

```sh
python3 scripts/import_bert_matmul_results.py bert-matmul --dry-run
python3 scripts/import_bert_matmul_results.py bert-matmul
python3 scripts/import_bert_matmul_results.py best-schedules
python3 scripts/import_bert_matmul_results.py best-pruned-config
python3 scripts/import_bert_matmul_results.py pruning-experiments
python3 scripts/import_bert_matmul_results.py comparison-results
python3 scripts/import_bert_matmul_results.py best-schedules --mode=direct
python3 scripts/import_bert_matmul_results.py pruning-experiments --mode=direct
python3 scripts/import_bert_matmul_results.py comparison-results --mode=direct
```

### Importing 80/20 best-schedules

If you need to import archived 80/20 best-config schedules, use:

```sh
python3 scripts/import_bert_matmul_results.py best-schedules --file=research/archive/metaschedule_8020/results/metaschedule/best_schedules_metaschedule_best_config.json
```

For a direct DB upload use:

```sh
python3 scripts/import_bert_matmul_results.py best-schedules --file=research/archive/metaschedule_8020/results/metaschedule/best_schedules_metaschedule_best_config.json --mode=direct
```

You can also add `--profile`, `--api-url` or other flags described below as needed.

Backward compatible shortcut:

```sh
python3 scripts/import_bert_matmul_results.py --dry-run
```

This defaults to the `bert-matmul` command.

Options:
- `bert-matmul`, `best-schedules`, `best-pruned-config`, `pruning-experiments`, or `comparison-results` to select which JSON artifact to import
- `--mode` to choose `api` (default) or `direct` DB connection
- `--api-url` to override `DATA_AGGREGATOR_URL`
- `best-schedules` command uses `DATA_AGGREGATOR_BEST_SCHEDULES_URL` by default
- `best-pruned-config` command uses `DATA_AGGREGATOR_BEST_PRUNED_CONFIG_URL` by default
- `pruning-experiments` command uses `DATA_AGGREGATOR_PRUNING_EXPERIMENTS_URL` by default
- `comparison-results` command uses `DATA_AGGREGATOR_COMPARISON_RESULTS_URL` by default
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
- Regular uploads (from `qkv_mlp_run` and `metaschedule_best_schedules`) default to 10 seconds via `DATA_AGGREGATOR_TIMEOUT`.
- Direct mode uses the sanitized profile table name: lowercase, non-alphanumerics replaced by `_`, and a leading digit is prefixed with `p`.
- `comparison-results` keeps a fixed snapshot table keyed by row label (`shape:*` + `overall`), so each import overwrites existing rows instead of appending history.
