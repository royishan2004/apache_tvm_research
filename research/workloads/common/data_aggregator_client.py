import json
import os
import uuid
import urllib.request
import urllib.error
from datetime import datetime
from typing import Iterable, Dict, Any, Tuple, Optional


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
    if profile is None:
        profile = os.environ.get("DATA_AGGREGATOR_PROFILE", "i5-1235U")

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
