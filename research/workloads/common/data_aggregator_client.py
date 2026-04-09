import json
import os
import platform
import re
import socket
import signal
import subprocess
import uuid
import urllib.request
import urllib.error
import urllib.parse
from datetime import datetime
from typing import Iterable, Dict, Any, Tuple, Optional

TABLE_SUFFIX = "bert_matmul_results"
DEFAULT_PROFILE = "i5-1235U"
TABLE_NAME_MAX = 63
PROFILE_MAX_LEN = max(1, TABLE_NAME_MAX - (len(TABLE_SUFFIX) + 1))
PROFILE_PATTERN = re.compile(r"^[A-Za-z0-9 _-]+$")
_CACHED_PROFILE: Optional[str] = None
DEFAULT_DATA_AGGREGATOR_URL = "http://localhost:3000/api/upload/bert_matmul_results"
DEFAULT_BEST_SCHEDULES_URL = "http://localhost:3000/api/upload/best_schedules"
DEFAULT_BEST_PRUNED_CONFIG_URL = "http://localhost:3000/api/upload/best_pruned_config"
DEFAULT_PRUNING_EXPERIMENTS_URL = "http://localhost:3000/api/upload/pruning_experiments"
DEFAULT_COMPARISON_RESULTS_URL = "http://localhost:3000/api/upload/comparison_results"


def upload_results(
    entries: Iterable[Dict[str, Any]],
    url: Optional[str] = None,
    profile: Optional[str] = None,
    dedupe: bool = False,
    timeout: Optional[int] = None,
) -> bool:
    """Upload BERT matmul result entries to the data aggregator.

    Controlled by env vars:
    - DATA_AGGREGATOR_URL (default: http://localhost:3000/api/upload/bert_matmul_results)
    - DATA_AGGREGATOR_PROFILE (default: i5-1235U)
    - DATA_AGGREGATOR_TIMEOUT (seconds, default: 10)
    - DATA_AGGREGATOR_DISABLE (set to "1" to disable)
    """
    if url is None:
        url = os.environ.get("DATA_AGGREGATOR_URL", DEFAULT_DATA_AGGREGATOR_URL)
    return _upload_payload(
        list(entries),
        url=url,
        profile=profile,
        dedupe=dedupe,
        timeout=timeout,
        payload_filename="bert_matmul_results.json",
    )


def upload_best_schedules(
    entries: Iterable[Dict[str, Any]],
    url: Optional[str] = None,
    profile: Optional[str] = None,
    dedupe: bool = False,
    timeout: Optional[int] = None,
) -> bool:
    """Upload MetaSchedule best-schedule entries to the data aggregator.

    Controlled by env vars:
    - DATA_AGGREGATOR_BEST_SCHEDULES_URL (default: http://localhost:3000/api/upload/best_schedules)
    - DATA_AGGREGATOR_PROFILE (default: auto-detected)
    - DATA_AGGREGATOR_TIMEOUT (seconds, default: 10)
    - DATA_AGGREGATOR_DISABLE (set to "1" to disable)
    """
    if url is None:
        url = os.environ.get(
            "DATA_AGGREGATOR_BEST_SCHEDULES_URL",
            DEFAULT_BEST_SCHEDULES_URL,
        )
    return _upload_payload(
        list(entries),
        url=url,
        profile=profile,
        dedupe=dedupe,
        timeout=timeout,
        payload_filename="best_schedules.json",
    )


def upload_best_pruned_config(
    payload: Dict[str, Any],
    url: Optional[str] = None,
    profile: Optional[str] = None,
    dedupe: bool = False,
    timeout: Optional[int] = None,
) -> bool:
    """Upload best_pruned_config payload to the data aggregator."""
    if url is None:
        url = os.environ.get(
            "DATA_AGGREGATOR_BEST_PRUNED_CONFIG_URL",
            DEFAULT_BEST_PRUNED_CONFIG_URL,
        )
    return _upload_payload(
        payload,
        url=url,
        profile=profile,
        dedupe=dedupe,
        timeout=timeout,
        payload_filename="best_pruned_config.json",
    )


def upload_pruning_experiments(
    payload: Any,
    url: Optional[str] = None,
    profile: Optional[str] = None,
    dedupe: bool = False,
    timeout: Optional[int] = None,
) -> bool:
    """Upload pruning_experiments payload to the data aggregator."""
    if url is None:
        url = os.environ.get(
            "DATA_AGGREGATOR_PRUNING_EXPERIMENTS_URL",
            DEFAULT_PRUNING_EXPERIMENTS_URL,
        )
    return _upload_payload(
        payload,
        url=url,
        profile=profile,
        dedupe=dedupe,
        timeout=timeout,
        payload_filename="pruning_experiments.json",
    )


def upload_comparison_results(
    payload: Any,
    url: Optional[str] = None,
    profile: Optional[str] = None,
    dedupe: bool = False,
    timeout: Optional[int] = None,
) -> bool:
    """Upload comparison_results payload to refresh the comp_summary snapshot."""
    if url is None:
        url = os.environ.get(
            "DATA_AGGREGATOR_COMPARISON_RESULTS_URL",
            DEFAULT_COMPARISON_RESULTS_URL,
        )
    return _upload_payload(
        payload,
        url=url,
        profile=profile,
        dedupe=dedupe,
        timeout=timeout,
        payload_filename="comparison_results.json",
    )


def _upload_payload(
    payload: Any,
    *,
    url: str,
    profile: Optional[str],
    dedupe: bool,
    timeout: Optional[int],
    payload_filename: str,
) -> bool:
    if os.environ.get("DATA_AGGREGATOR_DISABLE") == "1":
        return False

    resolved_profile = resolve_profile(profile)
    resolved_timeout = _resolve_timeout(timeout)

    if payload is None:
        return False
    if isinstance(payload, (list, tuple, set, dict)) and not payload:
        return False

    fields = {"profile": resolved_profile}
    if dedupe:
        fields["dedupe"] = "1"

    body, content_type = _encode_multipart(
        fields=fields,
        files=[(
            "file",
            payload_filename,
            "application/json",
            json.dumps(payload, default=_json_default).encode("utf-8"),
        )],
    )

    request = urllib.request.Request(url, data=body, method="POST")
    request.add_header("Content-Type", content_type)
    request.add_header("Content-Length", str(len(body)))

    try:
        with urllib.request.urlopen(request, timeout=resolved_timeout) as response:
            return 200 <= response.status < 300
    except urllib.error.HTTPError as exc:
        print(f"[data-aggregator] Upload failed: HTTP {exc.code}")
    except urllib.error.URLError as exc:
        print(f"[data-aggregator] Upload failed: {exc.reason}")
    except Exception as exc:
        print(f"[data-aggregator] Upload failed: {exc}")

    return False


def ensure_data_aggregator_connection_or_prompt(
    runner_name: str,
    *,
    prompt_timeout: int = 30,
    probe_timeout: int = 3,
    url: Optional[str] = None,
) -> bool:
    """Ensure the aggregator server is up, or ask to continue without DB uploads.

    Returns True if the service is reachable or if the user explicitly agrees to
    continue without it. On agreement, DATA_AGGREGATOR_DISABLE is set for this process.
    """
    if os.environ.get("DATA_AGGREGATOR_DISABLE") == "1":
        print("[data-aggregator] Disabled via DATA_AGGREGATOR_DISABLE=1; skipping connection check.")
        return True

    reachable, details = _probe_data_aggregator(url=url, timeout=probe_timeout)
    if reachable:
        print(f"[data-aggregator] Connected ({details})")
        return True

    print("\n" + "=" * 88)
    print("WARNING: No active data aggregator connection detected.")
    if details:
        print(f"Connection check details: {details}")
    print("Start it in another terminal with:")
    print("  cd services/data_aggregator && npm run dev")
    print(
        f"If you continue, {runner_name} will run without DB uploads for this session."
    )
    print("=" * 88)

    answer = _input_timeout_default_no(
        "Run tests without DB connection? [y/N]: ",
        seconds=prompt_timeout,
    )

    if answer in ("y", "yes"):
        os.environ["DATA_AGGREGATOR_DISABLE"] = "1"
        print("[data-aggregator] Continuing without DB connection; uploads are disabled.")
        return True

    print("[data-aggregator] Aborting test run. Start data_aggregator with npm run dev and retry.")
    return False


def is_data_aggregator_reachable(
    url: Optional[str] = None,
    timeout: Optional[int] = None,
) -> bool:
    """Check if the expected data aggregator service is reachable."""
    reachable, _ = _probe_data_aggregator(url=url, timeout=timeout)
    return reachable


def _probe_data_aggregator(
    url: Optional[str] = None,
    timeout: Optional[int] = None,
) -> tuple[bool, str]:
    """Probe aggregator connectivity and return (is_reachable, details)."""
    if url is None:
        url = os.environ.get("DATA_AGGREGATOR_URL", DEFAULT_DATA_AGGREGATOR_URL)

    probe_timeout = _resolve_timeout(timeout)
    parsed = urllib.parse.urlsplit(url)

    if parsed.scheme and parsed.netloc:
        host = parsed.hostname or "localhost"
        if _is_local_host(host):
            port = parsed.port or (443 if parsed.scheme == "https" else 80)
            if not _is_local_port_open(host, port, probe_timeout):
                return False, f"No local listener on {host}:{port}"

    last_error = "Service did not match Apache TVM Research Aggregator signature"

    for probe_url in _resolve_probe_urls(url):
        request = urllib.request.Request(probe_url, method="GET")
        try:
            with urllib.request.urlopen(request, timeout=probe_timeout) as response:
                if not (200 <= response.status < 300):
                    last_error = f"Unexpected HTTP {response.status} from {probe_url}"
                    continue
                raw_body = response.read(8192).decode("utf-8", errors="ignore")
                if _looks_like_data_aggregator_response(probe_url, raw_body):
                    return True, probe_url
                last_error = f"Signature mismatch at {probe_url}"
        except urllib.error.HTTPError:
            last_error = f"HTTP error at {probe_url}"
            continue
        except urllib.error.URLError:
            last_error = f"Connection refused/unreachable at {probe_url}"
            continue
        except Exception:
            last_error = f"Unexpected probe failure at {probe_url}"
            continue

    return False, last_error


def _json_default(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")


def _resolve_probe_urls(upload_url: str) -> list[str]:
    parsed = urllib.parse.urlsplit(upload_url)
    if parsed.scheme and parsed.netloc:
        base = f"{parsed.scheme}://{parsed.netloc}"
    else:
        base = "http://localhost:3000"

    return [
        f"{base}/openapi.json",
        f"{base}/",
    ]


def _is_local_host(host: str) -> bool:
    normalized = host.strip().lower()
    return normalized in {"localhost", "127.0.0.1", "::1"}


def _is_local_port_open(host: str, port: int, timeout_seconds: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=max(1, int(timeout_seconds))):
            return True
    except OSError:
        return False


def _looks_like_data_aggregator_response(probe_url: str, raw_body: str) -> bool:
    body = raw_body.strip()
    if probe_url.endswith("/openapi.json"):
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            return False

        title = str(((parsed.get("info") or {}).get("title") or "")).strip().lower()
        paths = parsed.get("paths") or {}
        return (
            title == "apache tvm research aggregator"
            and "/api/upload/bert_matmul_results" in paths
            and "/api/upload/best_schedules" in paths
            and "/api/upload/best_pruned_config" in paths
            and "/api/upload/pruning_experiments" in paths
            and "/api/upload/comparison_results" in paths
        )

    return body == "Healthy!"


def _input_timeout_default_no(prompt: str, seconds: int = 30) -> str:
    seconds = max(1, int(seconds))

    if hasattr(signal, "SIGALRM"):
        def _handler(signum, frame):
            raise TimeoutError

        old_handler = signal.getsignal(signal.SIGALRM)
        signal.signal(signal.SIGALRM, _handler)
        signal.alarm(seconds)
        try:
            return input(prompt).strip().lower()
        except (EOFError, KeyboardInterrupt, TimeoutError):
            return "n"
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)

    try:
        return input(prompt).strip().lower()
    except (EOFError, KeyboardInterrupt):
        return "n"


def resolve_profile(profile: Optional[str] = None) -> str:
    """Resolve the profile to send to the data aggregator.

    Priority: explicit argument -> DATA_AGGREGATOR_PROFILE -> auto-detected CPU -> DEFAULT_PROFILE.
    Returns a normalized lowercase profile that matches the server's accepted pattern.
    """
    global _CACHED_PROFILE

    if profile:
        normalized = _normalize_profile(profile)
        if normalized:
            return normalized

    env_profile = os.environ.get("DATA_AGGREGATOR_PROFILE")
    if env_profile:
        normalized = _normalize_profile(env_profile)
        if normalized:
            return normalized

    if _CACHED_PROFILE:
        return _CACHED_PROFILE

    detected = detect_cpu_profile()
    normalized = _normalize_profile(detected) if detected else None
    if normalized:
        _CACHED_PROFILE = normalized
        return normalized

    fallback = _normalize_profile(DEFAULT_PROFILE) or "unknown"
    _CACHED_PROFILE = fallback
    return fallback


def detect_cpu_profile() -> Optional[str]:
    """Detect a CPU brand string and normalize it for profile usage."""
    raw = _detect_cpu_brand_string()
    if not raw:
        return None
    raw = raw.strip()
    model = _extract_cpu_model(raw)
    return model or raw


def _normalize_profile(raw: str) -> Optional[str]:
    if not raw:
        return None
    cleaned = raw.strip()
    if not cleaned:
        return None
    cleaned = re.sub(r"[^A-Za-z0-9 _-]+", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        return None
    cleaned = cleaned.lower()
    if len(cleaned) > PROFILE_MAX_LEN:
        cleaned = cleaned[:PROFILE_MAX_LEN].rstrip()
    if not cleaned or PROFILE_PATTERN.fullmatch(cleaned) is None:
        return None
    return cleaned


def _extract_cpu_model(raw: str) -> Optional[str]:
    """Try to extract a compact CPU model identifier (e.g., i5-1235U, ryzen7-5800x)."""
    lower = raw.lower()

    intel_match = re.search(r"\b(i[3579])[- ]?(\d{4,5}[a-z]*)\b", lower)
    if intel_match:
        return f"{intel_match.group(1)}-{intel_match.group(2)}"

    xeon_match = re.search(r"\bxeon\s+([a-z]?[- ]?\d{4,5}[a-z]*)\b", lower)
    if xeon_match:
        return f"xeon-{xeon_match.group(1).replace(' ', '')}"

    ryzen_match = re.search(r"\bryzen\s+([3579])\s+(\d{4,5}[a-z]*)\b", lower)
    if ryzen_match:
        return f"ryzen{ryzen_match.group(1)}-{ryzen_match.group(2)}"

    apple_match = re.search(r"\b(m\d)\s*(pro|max|ultra)?\b", lower)
    if apple_match:
        suffix = apple_match.group(2)
        if suffix:
            return f"{apple_match.group(1)}-{suffix}"
        return apple_match.group(1)

    return None


def _detect_cpu_brand_string() -> Optional[str]:
    system = platform.system()
    try:
        if system == "Linux":
            try:
                with open("/proc/cpuinfo", "r", encoding="utf-8") as f:
                    for line in f:
                        if line.lower().startswith("model name"):
                            return line.split(":", 1)[1].strip()
            except OSError:
                pass
            try:
                output = subprocess.run(
                    ["lscpu"],
                    check=False,
                    capture_output=True,
                    text=True,
                ).stdout
                for line in output.splitlines():
                    if "Model name" in line or "model name" in line:
                        return line.split(":", 1)[1].strip()
            except OSError:
                return None
        if system == "Darwin":
            output = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                check=False,
                capture_output=True,
                text=True,
            ).stdout
            return output.strip() or None
        if system == "Windows":
            output = subprocess.run(
                ["wmic", "cpu", "get", "Name"],
                check=False,
                capture_output=True,
                text=True,
                shell=True,
            ).stdout
            lines = [line.strip() for line in output.splitlines() if line.strip()]
            return lines[1] if len(lines) >= 2 else None
    except Exception:
        return None

    return None


def _resolve_timeout(timeout: Optional[int]) -> int:
    if timeout is not None:
        return max(1, int(timeout))
    env_timeout = os.environ.get("DATA_AGGREGATOR_TIMEOUT")
    if env_timeout:
        try:
            return max(1, int(env_timeout))
        except ValueError:
            pass
    return 10


def _encode_multipart(
    fields: Dict[str, str],
    files: Iterable[Tuple[str, str, str, bytes]],
) -> Tuple[bytes, str]:
    boundary = uuid.uuid4().hex
    boundary_bytes = boundary.encode("ascii")
    parts: list[bytes] = []

    for name, value in fields.items():
        parts.extend(
            [
                b"--" + boundary_bytes,
                f'Content-Disposition: form-data; name="{name}"'.encode("ascii"),
                b"",
                value.encode("utf-8"),
            ]
        )

    for field, filename, content_type, data in files:
        parts.extend(
            [
                b"--" + boundary_bytes,
                (
                    f'Content-Disposition: form-data; name="{field}"; '
                    f'filename="{filename}"'
                ).encode("ascii"),
                f"Content-Type: {content_type}".encode("ascii"),
                b"",
                data,
            ]
        )

    parts.append(b"--" + boundary_bytes + b"--")
    parts.append(b"")

    body = b"\r\n".join(parts)
    content_type_header = f"multipart/form-data; boundary={boundary}"
    return body, content_type_header
