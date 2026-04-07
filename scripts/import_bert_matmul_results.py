#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BERT_RESULTS = ROOT / "research" / "results" / "bert_matmul_results.json"
DEFAULT_BEST_SCHEDULES = ROOT / "research" / "results" / "metaschedule" / "best_schedules.json"
DEFAULT_BEST_PRUNED_CONFIG = ROOT / "research" / "results" / "metaschedule" / "best_pruned_config.json"
DEFAULT_PRUNING_EXPERIMENTS = ROOT / "research" / "results" / "metaschedule" / "pruning_experiments.json"
DEFAULT_ENV = ROOT / "services" / "data_aggregator" / ".env"
IST = ZoneInfo("Asia/Kolkata")
DEFAULT_API_TIMEOUT = 300

DATASET_BERT = "bert-matmul"
DATASET_BEST = "best-schedules"
DATASET_BEST_PRUNED = "best-pruned-config"
DATASET_PRUNING = "pruning-experiments"
DATASET_CHOICES = (DATASET_BERT, DATASET_BEST, DATASET_BEST_PRUNED, DATASET_PRUNING)

BERT_TABLE_SUFFIX = "bert_matmul_results"
BEST_SCHEDULES_TABLE_SUFFIX = "best_schedules"
BEST_PRUNED_CONFIG_TABLE_SUFFIX = "best_pruned_config"
PRUNING_EXPERIMENTS_TABLE_SUFFIX = "pruning_experiments"
DEFAULT_PROFILE = "i5-1235U"
PROFILE_PATTERN = re.compile(r"^[A-Za-z0-9 _-]+$")
TABLE_NAME_MAX = 63
PROFILE_MAX_LEN = max(
    1,
    TABLE_NAME_MAX
    - (
        max(
            len(BERT_TABLE_SUFFIX),
            len(BEST_SCHEDULES_TABLE_SUFFIX),
            len(BEST_PRUNED_CONFIG_TABLE_SUFFIX),
            len(PRUNING_EXPERIMENTS_TABLE_SUFFIX),
        )
        + 1
    ),
)

DEFAULT_BERT_API_URL = "http://localhost:3000/api/upload/bert_matmul_results"
DEFAULT_BEST_SCHEDULES_API_URL = "http://localhost:3000/api/upload/best_schedules"
DEFAULT_BEST_PRUNED_CONFIG_API_URL = "http://localhost:3000/api/upload/best_pruned_config"
DEFAULT_PRUNING_EXPERIMENTS_API_URL = "http://localhost:3000/api/upload/pruning_experiments"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.workloads.common.data_aggregator_client import (  # noqa: E402
    resolve_profile as resolve_runtime_profile,
    upload_best_pruned_config,
    upload_best_schedules,
    upload_pruning_experiments,
    upload_results,
)

RE_TZ = re.compile(r"[Zz]|[+-]\d{2}(:?\d{2})?$")


def load_env_file(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    data: Dict[str, str] = {}
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        key, _, value = line.partition("=")
        if not key:
            continue
        data[key.strip()] = value.strip().strip('"').strip("'")
    return data


def normalize_profile(raw: str) -> Optional[str]:
    trimmed = raw.strip()
    if not trimmed:
        return None
    if PROFILE_PATTERN.fullmatch(trimmed) is None:
        return None
    if len(trimmed) > PROFILE_MAX_LEN:
        return None
    return trimmed.lower()


def clamp_identifier(name: str) -> str:
    return name[:TABLE_NAME_MAX] if len(name) > TABLE_NAME_MAX else name


def table_key(profile: str) -> str:
    base = re.sub(r"[^a-z0-9]+", "_", profile.lower()).strip("_")
    if not base:
        base = "profile"
    if base[0].isdigit():
        base = f"p{base}"
    if len(base) > PROFILE_MAX_LEN:
        base = base[:PROFILE_MAX_LEN]
    return base


def profile_table_name(profile: str, suffix: str) -> str:
    return clamp_identifier(f"{table_key(profile)}_{suffix}")


def legacy_profile_table_name(profile: str, suffix: str) -> str:
    return clamp_identifier(f"{profile} - {suffix}")


def parse_timestamp(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    if "T" not in raw and " " in raw:
        raw = raw.replace(" ", "T", 1)

    has_tz = RE_TZ.search(raw) is not None
    if has_tz:
        if raw.endswith("Z") or raw.endswith("z"):
            raw = raw[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(raw)
        except ValueError:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=ZoneInfo("UTC"))
        return parsed

    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        return None
    return parsed.replace(tzinfo=IST)


def to_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def to_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if value == 1:
            return True
        if value == 0:
            return False
        return None
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return None


def normalize_bert_entry(entry: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    kernel = entry.get("kernel")
    variant = entry.get("variant")
    target = entry.get("target") or "llvm"
    source = entry.get("source") or ""

    m = to_int(entry.get("M", entry.get("m")))
    k = to_int(entry.get("K", entry.get("k")))
    n = to_int(entry.get("N", entry.get("n")))

    latency_us = to_float(entry.get("latency_us", entry.get("latencyUs")))
    std_us = to_float(entry.get("std_us", entry.get("stdUs", 0.0)))

    ts_value = entry.get("timestamp", entry.get("ts"))
    ts = parse_timestamp(ts_value)

    if not isinstance(kernel, str) or not kernel:
        return None, "Missing kernel"
    if not isinstance(variant, str) or not variant:
        return None, "Missing variant"
    if not isinstance(target, str) or not target:
        return None, "Missing target"
    if m is None or k is None or n is None:
        return None, "Missing shape"
    if latency_us is None or std_us is None:
        return None, "Missing latency"
    if ts is None:
        return None, "Invalid timestamp"

    return {
        "ts": ts,
        "kernel": kernel,
        "variant": variant,
        "target": target,
        "source": source,
        "m": m,
        "k": k,
        "n": n,
        "latency_us": latency_us,
        "std_us": std_us,
        "number": to_int(entry.get("number")),
        "repeat": to_int(entry.get("repeat")),
        "min_repeat_ms": to_int(entry.get("min_repeat_ms", entry.get("minRepeatMs"))),
        "iteration": to_int(entry.get("iteration")),
        "total_iterations": to_int(entry.get("total_iterations", entry.get("totalIterations"))),
    }, None


def canonical_json(value: Any) -> Optional[str]:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError):
        return None


def normalize_best_schedule_entry(entry: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    kernel = entry.get("kernel")
    trace = entry.get("trace")

    m = to_int(entry.get("M", entry.get("m")))
    k = to_int(entry.get("K", entry.get("k")))
    n = to_int(entry.get("N", entry.get("n")))

    latency_us = to_float(entry.get("latency_us", entry.get("latencyUs")))
    std_us = to_float(entry.get("std_us", entry.get("stdUs", 0.0)))

    if not isinstance(kernel, str) or not kernel:
        return None, "Missing kernel"
    if m is None or k is None or n is None:
        return None, "Missing shape"
    if latency_us is None or std_us is None:
        return None, "Missing latency"
    if not isinstance(trace, str) or not trace:
        return None, "Missing trace"

    decisions = entry.get("decisions", [])
    decisions_json = canonical_json(decisions)
    if decisions_json is None:
        return None, "Invalid decisions"

    return {
        "kernel": kernel,
        "m": m,
        "k": k,
        "n": n,
        "latency_us": latency_us,
        "std_us": std_us,
        "trace": trace,
        "decisions": decisions,
        "decisions_json": decisions_json,
    }, None


def normalize_best_pruned_config_payload(
    payload: Dict[str, Any],
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    ts = parse_timestamp(payload.get("timestamp", payload.get("ts")))
    if ts is None:
        return None, "Invalid timestamp"

    selected_config = payload.get("selected_config")
    selected_metrics = payload.get("selected_metrics")

    if not isinstance(selected_config, dict):
        return None, "Missing selected_config object"
    if not isinstance(selected_metrics, dict):
        selected_metrics = {}

    selected_config_name = selected_config.get("name")
    if not isinstance(selected_config_name, str) or not selected_config_name:
        return None, "Missing selected_config.name"

    payload_json = canonical_json(payload)
    if payload_json is None:
        return None, "Invalid payload"

    payload_hash = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()

    return {
        "ts": ts,
        "target": str(payload.get("target") or ""),
        "selected_config_name": selected_config_name,
        "selected_state_token": str(selected_config.get("state_token") or ""),
        "selection_reason": str(payload.get("selection_reason") or ""),
        "latency_retention": to_float(selected_metrics.get("latency_retention")),
        "time_reduction": to_float(selected_metrics.get("time_reduction")),
        "trial_reduction": to_float(selected_metrics.get("trial_reduction")),
        "score": to_float(selected_metrics.get("score")),
        "payload": payload,
        "payload_json": payload_json,
        "payload_hash": payload_hash,
    }, None


def normalize_pruning_experiment_entry(
    entry: Dict[str, Any],
    metadata: Dict[str, Any],
    latest_pruning_run: Dict[str, Any],
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    run_id_raw = entry.get("run_id", entry.get("runId"))
    run_id = run_id_raw if isinstance(run_id_raw, str) else ""
    if not run_id:
        return None, "Missing run_id"

    ts = parse_timestamp(entry.get("timestamp", entry.get("ts")))
    if ts is None:
        return None, "Invalid timestamp"

    config_name_raw = entry.get("config_name", entry.get("configName"))
    config_hash_raw = entry.get("config_hash", entry.get("configHash"))
    config_name = config_name_raw if isinstance(config_name_raw, str) else ""
    config_hash = config_hash_raw if isinstance(config_hash_raw, str) else ""
    if not config_name or not config_hash:
        return None, "Missing config_name or config_hash"

    aggregate = entry.get("aggregate") if isinstance(entry.get("aggregate"), dict) else {}

    experiment_json = canonical_json(entry)
    metadata_json = canonical_json(metadata)
    latest_pruning_run_json = canonical_json(latest_pruning_run)
    if experiment_json is None or metadata_json is None or latest_pruning_run_json is None:
        return None, "Invalid experiment payload"

    return {
        "run_id": run_id,
        "ts": ts,
        "mode": str(entry.get("mode") or "pruning"),
        "iteration": to_int(entry.get("iteration")) or 0,
        "config_name": config_name,
        "config_hash": config_hash,
        "tasks_signature": str(entry.get("tasks_signature", entry.get("tasksSignature")) or ""),
        "is_baseline": to_bool(entry.get("is_baseline", entry.get("isBaseline"))) or False,
        "benchmark_only": to_bool(entry.get("benchmark_only", entry.get("benchmarkOnly"))) or False,
        "num_tasks": to_int(aggregate.get("num_tasks", aggregate.get("numTasks"))),
        "num_successful_tasks": to_int(
            aggregate.get("num_successful_tasks", aggregate.get("numSuccessfulTasks"))
        ),
        "all_tasks_succeeded": to_bool(
            aggregate.get("all_tasks_succeeded", aggregate.get("allTasksSucceeded"))
        ),
        "latency_geomean_us": to_float(
            aggregate.get("latency_geomean_us", aggregate.get("latencyGeomeanUs"))
        ),
        "total_tuning_time_sec": to_float(
            aggregate.get("total_tuning_time_sec", aggregate.get("totalTuningTimeSec"))
        ),
        "total_trials": to_int(aggregate.get("total_trials", aggregate.get("totalTrials"))),
        "latency_retention": to_float(
            aggregate.get("latency_retention", aggregate.get("latencyRetention"))
        ),
        "time_reduction": to_float(aggregate.get("time_reduction", aggregate.get("timeReduction"))),
        "trial_reduction": to_float(
            aggregate.get("trial_reduction", aggregate.get("trialReduction"))
        ),
        "score": to_float(aggregate.get("score")),
        "metadata": metadata,
        "latest_pruning_run": latest_pruning_run,
        "experiment": entry,
        "metadata_json": metadata_json,
        "latest_pruning_run_json": latest_pruning_run_json,
        "experiment_json": experiment_json,
    }, None


def load_entries(path: Path, dataset: str) -> Tuple[List[Dict[str, Any]], List[str]]:
    if not path.exists():
        return [], [f"File not found: {path}"]
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        return [], [f"Invalid JSON: {exc}"]

    rows: List[Dict[str, Any]] = []
    errors: List[str] = []
    seen: set[Tuple[Any, ...]] = set()

    if dataset == DATASET_BEST_PRUNED:
        if not isinstance(payload, dict):
            return [], ["best_pruned_config file must contain a JSON object"]
        normalized, error = normalize_best_pruned_config_payload(payload)
        if error:
            return [], [error]
        rows.append(normalized)
        return rows, errors

    if dataset == DATASET_PRUNING:
        if isinstance(payload, list):
            experiments = payload
            metadata: Dict[str, Any] = {}
            latest_pruning_run: Dict[str, Any] = {}
        elif isinstance(payload, dict):
            experiments_raw = payload.get("experiments")
            if isinstance(experiments_raw, list):
                experiments = experiments_raw
            else:
                experiments = [payload]

            metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
            latest_pruning_run = (
                payload.get("latest_pruning_run")
                if isinstance(payload.get("latest_pruning_run"), dict)
                else {}
            )
        else:
            return [], ["pruning_experiments file must contain a JSON object or array"]

        for idx, entry in enumerate(experiments):
            if not isinstance(entry, dict):
                errors.append(f"Entry {idx}: not an object")
                continue

            normalized, error = normalize_pruning_experiment_entry(
                entry,
                metadata=metadata,
                latest_pruning_run=latest_pruning_run,
            )
            if error:
                errors.append(f"Entry {idx}: {error}")
                continue

            key = (normalized["run_id"],)
            if key in seen:
                continue
            seen.add(key)
            rows.append(normalized)

        return rows, errors

    if not isinstance(payload, list):
        return [], ["Results file must contain a JSON array"]

    for idx, entry in enumerate(payload):
        if not isinstance(entry, dict):
            errors.append(f"Entry {idx}: not an object")
            continue

        if dataset == DATASET_BERT:
            normalized, error = normalize_bert_entry(entry)
            if error:
                errors.append(f"Entry {idx}: {error}")
                continue
            key = (
                normalized["kernel"],
                normalized["variant"],
                normalized["target"],
                normalized["source"],
                normalized["m"],
                normalized["k"],
                normalized["n"],
                normalized["ts"],
                normalized["latency_us"],
                normalized["std_us"],
                normalized["number"],
                normalized["repeat"],
                normalized["min_repeat_ms"],
                normalized["iteration"],
                normalized["total_iterations"],
            )
        elif dataset == DATASET_BEST:
            normalized, error = normalize_best_schedule_entry(entry)
            if error:
                errors.append(f"Entry {idx}: {error}")
                continue
            key = (
                normalized["kernel"],
                normalized["m"],
                normalized["k"],
                normalized["n"],
                normalized["latency_us"],
                normalized["std_us"],
                normalized["trace"],
                normalized["decisions_json"],
            )

        else:
            errors.append(f"Unsupported dataset: {dataset}")
            continue

        if key in seen:
            continue
        seen.add(key)
        rows.append(normalized)

    return rows, errors


def resolve_db_url(cli_url: Optional[str]) -> Optional[str]:
    if cli_url:
        return cli_url
    env_url = os.environ.get("DATABASE_URL")
    if env_url:
        return env_url
    env_data = load_env_file(DEFAULT_ENV)
    return env_data.get("DATABASE_URL")


def apply_ssl_options(
    db_url: str,
    sslmode: Optional[str],
    sslrootcert: Optional[str],
) -> str:
    parsed = urlparse(db_url)
    if not parsed.scheme or not parsed.netloc:
        return db_url

    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    has_sslmode = "sslmode" in query
    if sslmode:
        query["sslmode"] = sslmode
    elif not has_sslmode and not os.environ.get("PGSSLMODE"):
        if "neon.tech" in parsed.netloc:
            query["sslmode"] = "require"

    if sslrootcert:
        query["sslrootcert"] = sslrootcert

    new_query = urlencode(query)
    return urlunparse(parsed._replace(query=new_query))


def chunk_rows(rows: List[Dict[str, Any]], size: int) -> Iterable[List[Dict[str, Any]]]:
    for i in range(0, len(rows), size):
        yield rows[i : i + size]


def run_api_import(
    rows: List[Dict[str, Any]],
    dataset: str,
    api_url: Optional[str],
    profile: str,
    chunk_size: int,
    dedupe: bool,
    timeout: Optional[int],
) -> int:
    if dataset == DATASET_BERT:
        uploader = upload_results
        default_url = DEFAULT_BERT_API_URL
        env_url = "DATA_AGGREGATOR_URL"

        total = 0
        for chunk in chunk_rows(rows, chunk_size):
            ok = uploader(
                chunk,
                url=api_url,
                profile=profile,
                dedupe=dedupe,
                timeout=timeout,
            )
            if not ok:
                effective_url = api_url or os.environ.get(env_url, default_url)
                print(
                    f"Upload failed after {total} rows. "
                    f"Check {env_url} ({effective_url}) and that the server is running."
                )
                return 1
            total += len(chunk)

        print(f"Uploaded {total} rows via data_aggregator API.")
        return 0

    if dataset == DATASET_BEST:
        uploader = upload_best_schedules
        default_url = DEFAULT_BEST_SCHEDULES_API_URL
        env_url = "DATA_AGGREGATOR_BEST_SCHEDULES_URL"

        total = 0
        for chunk in chunk_rows(rows, chunk_size):
            ok = uploader(
                chunk,
                url=api_url,
                profile=profile,
                dedupe=dedupe,
                timeout=timeout,
            )
            if not ok:
                effective_url = api_url or os.environ.get(env_url, default_url)
                print(
                    f"Upload failed after {total} rows. "
                    f"Check {env_url} ({effective_url}) and that the server is running."
                )
                return 1
            total += len(chunk)

        print(f"Uploaded {total} rows via data_aggregator API.")
        return 0

    if dataset == DATASET_BEST_PRUNED:
        default_url = DEFAULT_BEST_PRUNED_CONFIG_API_URL
        env_url = "DATA_AGGREGATOR_BEST_PRUNED_CONFIG_URL"
        payload = rows[0].get("payload") if rows else None
        if not isinstance(payload, dict):
            print("No valid best_pruned_config payload to upload.")
            return 1
        ok = upload_best_pruned_config(
            payload,
            url=api_url,
            profile=profile,
            dedupe=dedupe,
            timeout=timeout,
        )
        if not ok:
            effective_url = api_url or os.environ.get(env_url, default_url)
            print(
                "Upload failed. "
                f"Check {env_url} ({effective_url}) and that the server is running."
            )
            return 1
        print("Uploaded 1 row via data_aggregator API.")
        return 0

    if dataset == DATASET_PRUNING:
        default_url = DEFAULT_PRUNING_EXPERIMENTS_API_URL
        env_url = "DATA_AGGREGATOR_PRUNING_EXPERIMENTS_URL"

        total = 0
        for chunk in chunk_rows(rows, chunk_size):
            metadata = chunk[0].get("metadata", {}) if chunk else {}
            latest = chunk[0].get("latest_pruning_run", {}) if chunk else {}
            payload = {
                "metadata": metadata if isinstance(metadata, dict) else {},
                "latest_pruning_run": latest if isinstance(latest, dict) else {},
                "experiments": [row.get("experiment", {}) for row in chunk],
            }
            ok = upload_pruning_experiments(
                payload,
                url=api_url,
                profile=profile,
                dedupe=dedupe,
                timeout=timeout,
            )
            if not ok:
                effective_url = api_url or os.environ.get(env_url, default_url)
                print(
                    f"Upload failed after {total} rows. "
                    f"Check {env_url} ({effective_url}) and that the server is running."
                )
                return 1
            total += len(chunk)

        print(f"Uploaded {total} rows via data_aggregator API.")
        return 0

    print(f"Unsupported dataset: {dataset}")
    return 1


def qident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def table_exists(cur: Any, table_name: str) -> bool:
    cur.execute(
        """
        select 1
        from information_schema.tables
        where table_schema = 'public' and table_name = %s
        limit 1
        """,
        (table_name,),
    )
    return cur.fetchone() is not None


def rename_legacy_tables(cur: Any, profile: str, suffix: str, table_name: str) -> None:
    legacy_profile_name = legacy_profile_table_name(profile, suffix)
    if legacy_profile_name != table_name:
        if table_exists(cur, legacy_profile_name) and not table_exists(cur, table_name):
            cur.execute(f"ALTER TABLE {qident(legacy_profile_name)} RENAME TO {qident(table_name)}")

    if profile == DEFAULT_PROFILE.lower():
        if table_exists(cur, suffix) and not table_exists(cur, table_name):
            cur.execute(f"ALTER TABLE {qident(suffix)} RENAME TO {qident(table_name)}")


def ensure_direct_table(cur: Any, dataset: str, profile: str, table_name: str) -> None:
    if dataset == DATASET_BERT:
        suffix = BERT_TABLE_SUFFIX
        rename_legacy_tables(cur, profile, suffix, table_name)
        cur.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto")
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {qident(table_name)} (
                id uuid primary key default gen_random_uuid(),
                ingested_at timestamptz not null default now(),
                ts timestamptz not null,
                kernel text not null,
                variant text not null,
                target text not null,
                source text not null default '',
                m integer not null,
                k integer not null,
                n integer not null,
                latency_us double precision not null,
                std_us double precision not null,
                number integer,
                repeat integer,
                min_repeat_ms integer,
                iteration integer,
                total_iterations integer
            )
            """
        )

        index_key = f"{table_key(profile)}_{suffix}"
        idx_kernel_variant_ts = clamp_identifier(f"idx_{index_key}_kernel_variant_ts")
        idx_shape = clamp_identifier(f"idx_{index_key}_shape")
        uniq_row = clamp_identifier(f"uniq_{index_key}_row")

        cur.execute(
            f"CREATE INDEX IF NOT EXISTS {qident(idx_kernel_variant_ts)} "
            f"ON {qident(table_name)} (kernel, variant, ts)"
        )
        cur.execute(
            f"CREATE INDEX IF NOT EXISTS {qident(idx_shape)} "
            f"ON {qident(table_name)} (m, k, n)"
        )
        cur.execute(
            f"""
            CREATE UNIQUE INDEX IF NOT EXISTS {qident(uniq_row)}
            ON {qident(table_name)} (
                kernel,
                variant,
                target,
                source,
                m,
                k,
                n,
                ts,
                latency_us,
                std_us,
                number,
                repeat,
                min_repeat_ms,
                iteration,
                total_iterations
            ) NULLS NOT DISTINCT
            """
        )
        return

    if dataset == DATASET_BEST:
        suffix = BEST_SCHEDULES_TABLE_SUFFIX
        rename_legacy_tables(cur, profile, suffix, table_name)
        cur.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto")
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {qident(table_name)} (
                id uuid primary key default gen_random_uuid(),
                ingested_at timestamptz not null default now(),
                kernel text not null,
                m integer not null,
                k integer not null,
                n integer not null,
                latency_us double precision not null,
                std_us double precision not null,
                trace text not null,
                decisions jsonb not null default '[]'::jsonb
            )
            """
        )

        index_key = f"{table_key(profile)}_{suffix}"
        idx_kernel_shape = clamp_identifier(f"idx_{index_key}_kernel_shape")
        idx_latency = clamp_identifier(f"idx_{index_key}_latency")
        uniq_row = clamp_identifier(f"uniq_{index_key}_row")

        cur.execute(
            f"CREATE INDEX IF NOT EXISTS {qident(idx_kernel_shape)} "
            f"ON {qident(table_name)} (kernel, m, k, n)"
        )
        cur.execute(
            f"CREATE INDEX IF NOT EXISTS {qident(idx_latency)} "
            f"ON {qident(table_name)} (latency_us)"
        )
        cur.execute(
            f"""
            CREATE UNIQUE INDEX IF NOT EXISTS {qident(uniq_row)}
            ON {qident(table_name)} (
                kernel,
                m,
                k,
                n,
                latency_us,
                std_us,
                trace,
                decisions
            )
            """
        )
        return

    if dataset == DATASET_BEST_PRUNED:
        suffix = BEST_PRUNED_CONFIG_TABLE_SUFFIX
        rename_legacy_tables(cur, profile, suffix, table_name)
        cur.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto")
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {qident(table_name)} (
                id uuid primary key default gen_random_uuid(),
                ingested_at timestamptz not null default now(),
                ts timestamptz not null,
                target text not null default '',
                selected_config_name text not null,
                selected_state_token text not null default '',
                selection_reason text not null default '',
                latency_retention double precision,
                time_reduction double precision,
                trial_reduction double precision,
                score double precision,
                payload_hash text not null,
                payload jsonb not null
            )
            """
        )

        index_key = f"{table_key(profile)}_{suffix}"
        idx_ts = clamp_identifier(f"idx_{index_key}_ts")
        idx_cfg = clamp_identifier(f"idx_{index_key}_config")
        idx_score = clamp_identifier(f"idx_{index_key}_score")
        uniq_hash = clamp_identifier(f"uniq_{index_key}_payload_hash")

        cur.execute(
            f"CREATE INDEX IF NOT EXISTS {qident(idx_ts)} "
            f"ON {qident(table_name)} (ts)"
        )
        cur.execute(
            f"CREATE INDEX IF NOT EXISTS {qident(idx_cfg)} "
            f"ON {qident(table_name)} (selected_config_name)"
        )
        cur.execute(
            f"CREATE INDEX IF NOT EXISTS {qident(idx_score)} "
            f"ON {qident(table_name)} (score)"
        )
        cur.execute(
            f"CREATE UNIQUE INDEX IF NOT EXISTS {qident(uniq_hash)} "
            f"ON {qident(table_name)} (payload_hash)"
        )
        return

    if dataset == DATASET_PRUNING:
        suffix = PRUNING_EXPERIMENTS_TABLE_SUFFIX
        rename_legacy_tables(cur, profile, suffix, table_name)
        cur.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto")
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {qident(table_name)} (
                id uuid primary key default gen_random_uuid(),
                ingested_at timestamptz not null default now(),
                run_id text not null,
                ts timestamptz not null,
                mode text not null,
                iteration integer not null,
                config_name text not null,
                config_hash text not null,
                tasks_signature text not null default '',
                is_baseline boolean not null default false,
                benchmark_only boolean not null default false,
                num_tasks integer,
                num_successful_tasks integer,
                all_tasks_succeeded boolean,
                latency_geomean_us double precision,
                total_tuning_time_sec double precision,
                total_trials integer,
                latency_retention double precision,
                time_reduction double precision,
                trial_reduction double precision,
                score double precision,
                metadata jsonb not null default '{{}}'::jsonb,
                latest_pruning_run jsonb not null default '{{}}'::jsonb,
                experiment jsonb not null
            )
            """
        )

        index_key = f"{table_key(profile)}_{suffix}"
        idx_ts = clamp_identifier(f"idx_{index_key}_ts")
        idx_cfg_iter = clamp_identifier(f"idx_{index_key}_cfg_iter")
        idx_score = clamp_identifier(f"idx_{index_key}_score")
        uniq_run_id = clamp_identifier(f"uniq_{index_key}_run_id")

        cur.execute(
            f"CREATE INDEX IF NOT EXISTS {qident(idx_ts)} "
            f"ON {qident(table_name)} (ts)"
        )
        cur.execute(
            f"CREATE INDEX IF NOT EXISTS {qident(idx_cfg_iter)} "
            f"ON {qident(table_name)} (config_name, iteration)"
        )
        cur.execute(
            f"CREATE INDEX IF NOT EXISTS {qident(idx_score)} "
            f"ON {qident(table_name)} (score)"
        )
        cur.execute(
            f"CREATE UNIQUE INDEX IF NOT EXISTS {qident(uniq_run_id)} "
            f"ON {qident(table_name)} (run_id)"
        )
        return

    raise ValueError(f"Unsupported dataset: {dataset}")


def run_direct_import_bert(
    rows: List[Dict[str, Any]],
    db_url: str,
    table_name: str,
    profile: str,
    chunk_size: int,
    dedupe: bool,
) -> int:
    try:
        import psycopg
    except ImportError:
        print("psycopg is required. Install with: pip install psycopg[binary]")
        return 1

    select_sql = f"""
        SELECT 1
        FROM {qident(table_name)}
        WHERE kernel = %(kernel)s
          AND variant = %(variant)s
          AND target = %(target)s
          AND source = %(source)s
          AND m = %(m)s
          AND k = %(k)s
          AND n = %(n)s
          AND ts = %(ts)s
          AND latency_us = %(latency_us)s
          AND std_us = %(std_us)s
          AND number IS NOT DISTINCT FROM %(number)s
          AND repeat IS NOT DISTINCT FROM %(repeat)s
          AND min_repeat_ms IS NOT DISTINCT FROM %(min_repeat_ms)s
          AND iteration IS NOT DISTINCT FROM %(iteration)s
          AND total_iterations IS NOT DISTINCT FROM %(total_iterations)s
        LIMIT 1
    """

    insert_sql = f"""
        INSERT INTO {qident(table_name)} (
            ts,
            kernel,
            variant,
            target,
            source,
            m,
            k,
            n,
            latency_us,
            std_us,
            number,
            repeat,
            min_repeat_ms,
            iteration,
            total_iterations
        )
        VALUES (
            %(ts)s,
            %(kernel)s,
            %(variant)s,
            %(target)s,
            %(source)s,
            %(m)s,
            %(k)s,
            %(n)s,
            %(latency_us)s,
            %(std_us)s,
            %(number)s,
            %(repeat)s,
            %(min_repeat_ms)s,
            %(iteration)s,
            %(total_iterations)s
        )
    """

    inserted = 0
    skipped = 0
    pending: List[Dict[str, Any]] = []

    try:
        with psycopg.connect(db_url) as conn:
            with conn.cursor() as cur:
                ensure_direct_table(cur, DATASET_BERT, profile, table_name)
                conn.commit()

                for row in rows:
                    if dedupe:
                        cur.execute(select_sql, row)
                        if cur.fetchone():
                            skipped += 1
                            continue

                    pending.append(row)
                    if len(pending) >= chunk_size:
                        cur.executemany(insert_sql, pending)
                        conn.commit()
                        inserted += len(pending)
                        pending.clear()

                if pending:
                    cur.executemany(insert_sql, pending)
                    conn.commit()
                    inserted += len(pending)
    except psycopg.OperationalError as exc:
        print(f"Connection failed: {exc}")
        if "sslrootcert" in str(exc) or "certificate" in str(exc):
            print("Hint: try --sslmode=require (Neon default) or --sslrootcert=system.")
        return 1

    print(f"Imported {inserted} rows. Skipped {skipped} duplicates.")
    return 0


def run_direct_import_best_schedules(
    rows: List[Dict[str, Any]],
    db_url: str,
    table_name: str,
    profile: str,
    chunk_size: int,
    dedupe: bool,
) -> int:
    try:
        import psycopg
    except ImportError:
        print("psycopg is required. Install with: pip install psycopg[binary]")
        return 1

    select_sql = f"""
        SELECT 1
        FROM {qident(table_name)}
        WHERE kernel = %(kernel)s
          AND m = %(m)s
          AND k = %(k)s
          AND n = %(n)s
          AND latency_us = %(latency_us)s
          AND std_us = %(std_us)s
          AND trace = %(trace)s
          AND decisions = %(decisions_json)s::jsonb
        LIMIT 1
    """

    insert_sql = f"""
        INSERT INTO {qident(table_name)} (
            kernel,
            m,
            k,
            n,
            latency_us,
            std_us,
            trace,
            decisions
        )
        VALUES (
            %(kernel)s,
            %(m)s,
            %(k)s,
            %(n)s,
            %(latency_us)s,
            %(std_us)s,
            %(trace)s,
            %(decisions_json)s::jsonb
        )
    """

    inserted = 0
    skipped = 0
    pending: List[Dict[str, Any]] = []

    try:
        with psycopg.connect(db_url) as conn:
            with conn.cursor() as cur:
                ensure_direct_table(cur, DATASET_BEST, profile, table_name)
                conn.commit()

                for row in rows:
                    if dedupe:
                        cur.execute(select_sql, row)
                        if cur.fetchone():
                            skipped += 1
                            continue

                    pending.append(row)
                    if len(pending) >= chunk_size:
                        cur.executemany(insert_sql, pending)
                        conn.commit()
                        inserted += len(pending)
                        pending.clear()

                if pending:
                    cur.executemany(insert_sql, pending)
                    conn.commit()
                    inserted += len(pending)
    except psycopg.OperationalError as exc:
        print(f"Connection failed: {exc}")
        if "sslrootcert" in str(exc) or "certificate" in str(exc):
            print("Hint: try --sslmode=require (Neon default) or --sslrootcert=system.")
        return 1

    print(f"Imported {inserted} rows. Skipped {skipped} duplicates.")
    return 0


def run_direct_import_best_pruned_config(
    rows: List[Dict[str, Any]],
    db_url: str,
    table_name: str,
    profile: str,
    chunk_size: int,
    dedupe: bool,
) -> int:
    try:
        import psycopg
    except ImportError:
        print("psycopg is required. Install with: pip install psycopg[binary]")
        return 1

    select_sql = f"""
        SELECT 1
        FROM {qident(table_name)}
        WHERE payload_hash = %(payload_hash)s
        LIMIT 1
    """

    insert_sql = f"""
        INSERT INTO {qident(table_name)} (
            ts,
            target,
            selected_config_name,
            selected_state_token,
            selection_reason,
            latency_retention,
            time_reduction,
            trial_reduction,
            score,
            payload_hash,
            payload
        )
        VALUES (
            %(ts)s,
            %(target)s,
            %(selected_config_name)s,
            %(selected_state_token)s,
            %(selection_reason)s,
            %(latency_retention)s,
            %(time_reduction)s,
            %(trial_reduction)s,
            %(score)s,
            %(payload_hash)s,
            %(payload_json)s::jsonb
        )
    """

    inserted = 0
    skipped = 0
    pending: List[Dict[str, Any]] = []

    try:
        with psycopg.connect(db_url) as conn:
            with conn.cursor() as cur:
                ensure_direct_table(cur, DATASET_BEST_PRUNED, profile, table_name)
                conn.commit()

                for row in rows:
                    if dedupe:
                        cur.execute(select_sql, row)
                        if cur.fetchone():
                            skipped += 1
                            continue

                    pending.append(row)
                    if len(pending) >= chunk_size:
                        cur.executemany(insert_sql, pending)
                        conn.commit()
                        inserted += len(pending)
                        pending.clear()

                if pending:
                    cur.executemany(insert_sql, pending)
                    conn.commit()
                    inserted += len(pending)
    except psycopg.OperationalError as exc:
        print(f"Connection failed: {exc}")
        if "sslrootcert" in str(exc) or "certificate" in str(exc):
            print("Hint: try --sslmode=require (Neon default) or --sslrootcert=system.")
        return 1

    print(f"Imported {inserted} rows. Skipped {skipped} duplicates.")
    return 0


def run_direct_import_pruning_experiments(
    rows: List[Dict[str, Any]],
    db_url: str,
    table_name: str,
    profile: str,
    chunk_size: int,
    dedupe: bool,
) -> int:
    try:
        import psycopg
    except ImportError:
        print("psycopg is required. Install with: pip install psycopg[binary]")
        return 1

    select_sql = f"""
        SELECT 1
        FROM {qident(table_name)}
        WHERE run_id = %(run_id)s
        LIMIT 1
    """

    insert_sql = f"""
        INSERT INTO {qident(table_name)} (
            run_id,
            ts,
            mode,
            iteration,
            config_name,
            config_hash,
            tasks_signature,
            is_baseline,
            benchmark_only,
            num_tasks,
            num_successful_tasks,
            all_tasks_succeeded,
            latency_geomean_us,
            total_tuning_time_sec,
            total_trials,
            latency_retention,
            time_reduction,
            trial_reduction,
            score,
            metadata,
            latest_pruning_run,
            experiment
        )
        VALUES (
            %(run_id)s,
            %(ts)s,
            %(mode)s,
            %(iteration)s,
            %(config_name)s,
            %(config_hash)s,
            %(tasks_signature)s,
            %(is_baseline)s,
            %(benchmark_only)s,
            %(num_tasks)s,
            %(num_successful_tasks)s,
            %(all_tasks_succeeded)s,
            %(latency_geomean_us)s,
            %(total_tuning_time_sec)s,
            %(total_trials)s,
            %(latency_retention)s,
            %(time_reduction)s,
            %(trial_reduction)s,
            %(score)s,
            %(metadata_json)s::jsonb,
            %(latest_pruning_run_json)s::jsonb,
            %(experiment_json)s::jsonb
        )
    """

    inserted = 0
    skipped = 0
    pending: List[Dict[str, Any]] = []

    try:
        with psycopg.connect(db_url) as conn:
            with conn.cursor() as cur:
                ensure_direct_table(cur, DATASET_PRUNING, profile, table_name)
                conn.commit()

                for row in rows:
                    if dedupe:
                        cur.execute(select_sql, row)
                        if cur.fetchone():
                            skipped += 1
                            continue

                    pending.append(row)
                    if len(pending) >= chunk_size:
                        cur.executemany(insert_sql, pending)
                        conn.commit()
                        inserted += len(pending)
                        pending.clear()

                if pending:
                    cur.executemany(insert_sql, pending)
                    conn.commit()
                    inserted += len(pending)
    except psycopg.OperationalError as exc:
        print(f"Connection failed: {exc}")
        if "sslrootcert" in str(exc) or "certificate" in str(exc):
            print("Hint: try --sslmode=require (Neon default) or --sslrootcert=system.")
        return 1

    print(f"Imported {inserted} rows. Skipped {skipped} duplicates.")
    return 0


def resolve_profile(cli_profile: Optional[str]) -> str:
    if cli_profile:
        return cli_profile
    return resolve_runtime_profile()


def add_common_args(parser: argparse.ArgumentParser, default_file: Path) -> None:
    parser.add_argument(
        "--mode",
        choices=["api", "direct"],
        default="api",
        help="Import via the data_aggregator API (default) or direct DB connection",
    )
    parser.add_argument(
        "--api-url",
        default=None,
        help="Override dataset-specific DATA_AGGREGATOR URL when using api mode",
    )
    parser.add_argument(
        "--profile",
        default=None,
        help=(
            "Override DATA_AGGREGATOR_PROFILE (used by api mode and to select the target table in direct mode)"
        ),
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_API_TIMEOUT,
        help=(
            "Upload timeout in seconds (api mode). Defaults to 300; overrides "
            "DATA_AGGREGATOR_TIMEOUT if set."
        ),
    )
    parser.add_argument(
        "--no-dedupe",
        action="store_true",
        help="Disable dedupe checks for inserts",
    )
    parser.add_argument(
        "--file",
        default=str(default_file),
        help="Path to results JSON file",
    )
    parser.add_argument(
        "--db-url",
        default=None,
        help="Override DATABASE_URL (direct mode only)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="Insert batch size",
    )
    parser.add_argument(
        "--sslmode",
        default=None,
        help="Override libpq sslmode (direct mode only, example: require, verify-full)",
    )
    parser.add_argument(
        "--sslrootcert",
        default=None,
        help="Path to a root certificate or 'system' for OS trust store (direct mode only)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and dedupe without writing to the database",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Import benchmark JSON files into Neon with profile-specific tables. "
            "Commands: bert-matmul | best-schedules | best-pruned-config | pruning-experiments"
        )
    )
    subparsers = parser.add_subparsers(dest="command")

    bert_parser = subparsers.add_parser(
        DATASET_BERT,
        help="Import research/results/bert_matmul_results.json",
    )
    add_common_args(bert_parser, DEFAULT_BERT_RESULTS)

    best_parser = subparsers.add_parser(
        DATASET_BEST,
        help="Import research/results/metaschedule/best_schedules.json",
    )
    add_common_args(best_parser, DEFAULT_BEST_SCHEDULES)

    best_pruned_parser = subparsers.add_parser(
        DATASET_BEST_PRUNED,
        help="Import research/results/metaschedule/best_pruned_config.json",
    )
    add_common_args(best_pruned_parser, DEFAULT_BEST_PRUNED_CONFIG)

    pruning_parser = subparsers.add_parser(
        DATASET_PRUNING,
        help="Import research/results/metaschedule/pruning_experiments.json",
    )
    add_common_args(pruning_parser, DEFAULT_PRUNING_EXPERIMENTS)

    return parser


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = build_parser()

    if len(argv) <= 1:
        argv = [argv[0], DATASET_BERT]
    elif argv[1] not in DATASET_CHOICES and argv[1] not in {"-h", "--help"}:
        # Backward compatibility: treat old invocation as bert-matmul command.
        argv = [argv[0], DATASET_BERT, *argv[1:]]

    args = parser.parse_args(argv[1:])
    if not getattr(args, "command", None):
        args.command = DATASET_BERT
    return args


def main() -> int:
    args = parse_args(sys.argv)
    dataset = args.command

    file_path = Path(args.file)
    rows, errors = load_entries(file_path, dataset)
    if errors:
        print("Skipped invalid entries:")
        for message in errors[:10]:
            print(f"  - {message}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")

    if not rows:
        print("No valid rows to import.")
        return 1

    if args.dry_run:
        print(f"Dry run ({dataset}): {len(rows)} normalized rows ready to import.")
        return 0

    profile_raw = resolve_profile(args.profile)
    normalized_profile = normalize_profile(profile_raw)
    if not normalized_profile:
        print(
            "Invalid profile. Allowed pattern: [A-Za-z0-9 _-], "
            f"max length {PROFILE_MAX_LEN}."
        )
        return 1

    if args.mode == "api":
        return run_api_import(
            rows,
            dataset=dataset,
            api_url=args.api_url,
            profile=normalized_profile,
            chunk_size=args.chunk_size,
            dedupe=not args.no_dedupe,
            timeout=args.timeout,
        )

    db_url = resolve_db_url(args.db_url)
    if not db_url:
        print("DATABASE_URL not set. Use --db-url or set it in the environment.")
        return 1

    db_url = apply_ssl_options(db_url, args.sslmode, args.sslrootcert)

    if dataset == DATASET_BERT:
        table_name = profile_table_name(normalized_profile, BERT_TABLE_SUFFIX)
        return run_direct_import_bert(
            rows,
            db_url=db_url,
            table_name=table_name,
            profile=normalized_profile,
            chunk_size=args.chunk_size,
            dedupe=not args.no_dedupe,
        )

    if dataset == DATASET_BEST:
        table_name = profile_table_name(normalized_profile, BEST_SCHEDULES_TABLE_SUFFIX)
        return run_direct_import_best_schedules(
            rows,
            db_url=db_url,
            table_name=table_name,
            profile=normalized_profile,
            chunk_size=args.chunk_size,
            dedupe=not args.no_dedupe,
        )

    if dataset == DATASET_BEST_PRUNED:
        table_name = profile_table_name(normalized_profile, BEST_PRUNED_CONFIG_TABLE_SUFFIX)
        return run_direct_import_best_pruned_config(
            rows,
            db_url=db_url,
            table_name=table_name,
            profile=normalized_profile,
            chunk_size=args.chunk_size,
            dedupe=not args.no_dedupe,
        )

    if dataset == DATASET_PRUNING:
        table_name = profile_table_name(normalized_profile, PRUNING_EXPERIMENTS_TABLE_SUFFIX)
        return run_direct_import_pruning_experiments(
            rows,
            db_url=db_url,
            table_name=table_name,
            profile=normalized_profile,
            chunk_size=args.chunk_size,
            dedupe=not args.no_dedupe,
        )

    print(f"Unsupported dataset: {dataset}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
