import os
import signal
from typing import Optional

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

def ensure_benchmark_settings_or_prompt(
    runner_name: str,
    *,
    prompt_timeout: int = 30,
) -> bool:
    """Ensure benchmark settings are active, or ask to continue without them."""
    if os.environ.get("BENCHMARK_SETTINGS_ACTIVE") == "1":
        return True

    if os.environ.get("BENCHMARK_SETTINGS_DISABLE") == "1":
        print("[benchmark-env] Disabled via BENCHMARK_SETTINGS_DISABLE=1; skipping check.")
        return True

    print("\n" + "=" * 88)
    print("WARNING: benchmark_settings.sh has not been loaded.")
    print("Execution latencies and thread mapping will be significantly degraded or inaccurate.")
    print("Load the settings in the current context before running with:")
    print("  source scripts/benchmark_settings.sh")
    print(
        f"If you continue, {runner_name} will run with default system threading."
    )
    print("=" * 88)

    answer = _input_timeout_default_no(
        "Run tests without optimal benchmark thread settings? [y/N]: ",
        seconds=prompt_timeout,
    )

    if answer in ("y", "yes"):
        os.environ["BENCHMARK_SETTINGS_DISABLE"] = "1"
        print("[benchmark-env] Continuing without optimal thread configurations.")
        return True

    print("[benchmark-env] Aborting test run. Run 'source scripts/benchmark_settings.sh' first.")
    return False
