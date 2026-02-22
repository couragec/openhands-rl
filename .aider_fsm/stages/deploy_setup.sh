#!/usr/bin/env bash
set -euo pipefail

TRAINED_MODEL_PATH="${TRAINED_MODEL_PATH:-${AIDER_TRAINED_MODEL_DIR:-}}"
RUNTIME_ENV_JSON=".aider_fsm/runtime_env.json"

if [ -z "$TRAINED_MODEL_PATH" ]; then
    echo "ERROR: TRAINED_MODEL_PATH not set" >&2
    exit 1
fi

if [ ! -d "$TRAINED_MODEL_PATH" ]; then
    echo "ERROR: model directory not found: $TRAINED_MODEL_PATH" >&2
    exit 1
fi

echo "Deploy: model=$TRAINED_MODEL_PATH (direct-inference mode)"

MODEL_NAME=$(basename "$TRAINED_MODEL_PATH")

cat > "$RUNTIME_ENV_JSON" <<ENVJSON
{
    "inference": {
        "mode": "direct",
        "model": "${MODEL_NAME}",
        "model_path": "${TRAINED_MODEL_PATH}"
    },
    "deploy_engine": "direct",
    "model_path": "${TRAINED_MODEL_PATH}"
}
ENVJSON

echo "runtime_env.json written (direct mode)"
