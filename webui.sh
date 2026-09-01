#!/usr/bin/env sh
set -eu

PROJECT_DIR=$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd)
cd "$PROJECT_DIR"

if ! command -v uv >/dev/null 2>&1; then
    echo "uv is required. Install it from https://docs.astral.sh/uv/getting-started/installation/" >&2
    exit 127
fi

exec uv run --frozen streamlit run \
    "$@" \
    --server.address 127.0.0.1 \
    src/ai_vocab_video_generator/webui.py
