#!/usr/bin/env bash
set -euo pipefail

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"

curl -sS "http://${HOST}:${PORT}/zero-shot" \
  -H "Content-Type: application/json" \
  -d @payload.json \
  | jq .
