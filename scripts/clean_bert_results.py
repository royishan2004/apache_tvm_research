#!/usr/bin/env python3
import json
import shutil
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "research" / "results" / "bert_matmul_results.json"

def backup(path: Path) -> Path:
    ts = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    bak = path.with_suffix(path.suffix + f".bak.{ts}")
    shutil.copy2(path, bak)
    return bak

def main():
    if not RESULTS.exists():
        print(f"File not found: {RESULTS}")
        return
    bak = backup(RESULTS)
    data = json.loads(RESULTS.read_text())
    before = len(data)
    filtered = [r for r in data if r.get("variant") != "rule_based" and r.get("variant") != "rule_based_ms4"]
    after = len(filtered)
    RESULTS.write_text(json.dumps(filtered, indent=2))
    print(f"Backed up {RESULTS} -> {bak}")
    print(f"Removed {before - after} records (before: {before}, after: {after})")

if __name__ == '__main__':
    main()
