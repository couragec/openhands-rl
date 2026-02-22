#!/usr/bin/env bash
set -euo pipefail

RUNTIME_ENV_JSON=".aider_fsm/runtime_env.json"

if [ ! -f "$RUNTIME_ENV_JSON" ]; then
    echo "ERROR: runtime_env.json not found" >&2
    exit 1
fi

echo "Health check: runtime_env.json exists"
exit 0
