#!/usr/bin/env bash
set -euo pipefail

find_listener_pid() {
  if command -v ss >/dev/null 2>&1; then
    ss -ltnp '( sport = :3000 )' 2>/dev/null \
      | sed -n 's/.*pid=\([0-9]\+\).*/\1/p' \
      | head -n 1
    return 0
  fi

  if command -v lsof >/dev/null 2>&1; then
    lsof -t -i:3000 2>/dev/null | head -n 1
    return 0
  fi

  if command -v fuser >/dev/null 2>&1; then
    fuser -n tcp 3000 2>/dev/null | awk '{print $1}'
    return 0
  fi

  return 0
}

pid="$(find_listener_pid || true)"
if [[ -n "${pid}" ]]; then
  echo "Killing existing process ${pid} on port 3000"
  kill "${pid}" || true
  sleep 1
fi

echo "Starting data aggregator on port 3000"
exec tsx src/index.ts
