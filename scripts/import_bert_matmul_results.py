#!/usr/bin/env python3
import argparse
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
DEFAULT_ENV = ROOT / "services" / "data_aggregator" / ".env"
IST = ZoneInfo("Asia/Kolkata")
DEFAULT_API_TIMEOUT = 300

DATASET_BERT = "bert-matmul"
DATASET_BEST = "best-schedules"
DATASET_CHOICES = (DATASET_BERT, DATASET_BEST)

BERT_TABLE_SUFFIX = "bert_matmul_results"
BEST_SCHEDULES_TABLE_SUFFIX = "best_schedules"
DEFAULT_PROFILE = "i5-1235U"
PROFILE_PATTERN = re.compile(r"^[A-Za-z0-9 _-]+$")
TABLE_NAME_MAX = 63
PROFILE_MAX_LEN = max(1, TABLE_NAME_MAX - (len(BERT_TABLE_SUFFIX) + 1))

DEFAULT_BERT_API_URL = "http://localhost:3000/api/upload/bert_matmul_results"
DEFAULT_BEST_SCHEDULES_API_URL = "http://localhost:3000/api/upload/best_schedules"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.workloads.common.data_aggregator_client import (  # noqa: E402
    resolve_profile as resolve_runtime_profile,
    upload_best_schedules,
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


def load_entries(path: Path, dataset: str) -> Tuple[List[Dict[str, Any]], List[str]]:
    if not path.exists():
        return [], [f"File not found: {path}"]
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        return [], [f"Invalid JSON: {exc}"]
    if not isinstance(payload, list):
        return [], ["Results file must contain a JSON array"]

    rows: List[Dict[str, Any]] = []
    errors: List[str] = []
    seen: set[Tuple[Any, ...]] = set()

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
        else:
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
    else:
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
            "Commands: bert-matmul | best-schedules"
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

    table_name = profile_table_name(normalized_profile, BEST_SCHEDULES_TABLE_SUFFIX)
    return run_direct_import_best_schedules(
        rows,
        db_url=db_url,
        table_name=table_name,
        profile=normalized_profile,
        chunk_size=args.chunk_size,
        dedupe=not args.no_dedupe,
    )


if __name__ == "__main__":
    sys.exit(main())
