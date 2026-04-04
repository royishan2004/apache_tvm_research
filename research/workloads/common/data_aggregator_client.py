import json
import os
import platform
import re
import subprocess
import uuid
import urllib.request
import urllib.error
from datetime import datetime
from typing import Iterable, Dict, Any, Tuple, Optional

TABLE_SUFFIX = "bert_matmul_results"
DEFAULT_PROFILE = "i5-1235U"
TABLE_NAME_MAX = 63
PROFILE_MAX_LEN = max(1, TABLE_NAME_MAX - (len(TABLE_SUFFIX) + 1))
PROFILE_PATTERN = re.compile(r"^[A-Za-z0-9 _-]+$")
_CACHED_PROFILE: Optional[str] = None


def upload_results(
    entries: Iterable[Dict[str, Any]],
    url: Optional[str] = None,
    profile: Optional[str] = None,
    dedupe: bool = False,
    timeout: Optional[int] = None,
) -> bool:
    """Upload a list of result entries to the data aggregator.

    Controlled by env vars:
    - DATA_AGGREGATOR_URL (default: http://localhost:3000/api/upload/bert_matmul_results)
    - DATA_AGGREGATOR_PROFILE (default: i5-1235U)
    - DATA_AGGREGATOR_TIMEOUT (seconds, default: 10)
    - DATA_AGGREGATOR_DISABLE (set to "1" to disable)
    """
    if os.environ.get("DATA_AGGREGATOR_DISABLE") == "1":
        return False

    if url is None:
        url = os.environ.get(
            "DATA_AGGREGATOR_URL",
            "http://localhost:3000/api/upload/bert_matmul_results",
        )
    profile = resolve_profile(profile)

    timeout = _resolve_timeout(timeout)

    payload = list(entries)
    if not payload:
        return False

    fields = {"profile": profile}
    if dedupe:
        fields["dedupe"] = "1"

    body, content_type = _encode_multipart(
        fields=fields,
        files=[(
            "file",
            "results.json",
            "application/json",
            json.dumps(payload, default=_json_default).encode("utf-8"),
        )],
    )

    request = urllib.request.Request(url, data=body, method="POST")
    request.add_header("Content-Type", content_type)
    request.add_header("Content-Length", str(len(body)))

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return 200 <= response.status < 300
    except urllib.error.HTTPError as exc:
        print(f"[data-aggregator] Upload failed: HTTP {exc.code}")
    except urllib.error.URLError as exc:
        print(f"[data-aggregator] Upload failed: {exc.reason}")
    except Exception as exc:
        print(f"[data-aggregator] Upload failed: {exc}")

    return False


def _json_default(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")


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
