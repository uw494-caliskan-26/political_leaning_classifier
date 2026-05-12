#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

if ! command -v uv >/dev/null 2>&1; then
  echo "Install uv first: https://docs.astral.sh/uv/"
  exit 1
fi

echo "Syncing dependencies (uv)..."
uv sync

echo "Launching Political Leaning Classifier WebUI..."
uv run python app.py
