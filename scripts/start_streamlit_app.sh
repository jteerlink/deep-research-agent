#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${PORT:-8501}"
HOST="${HOST:-localhost}"
BIND_ADDR="${BIND_ADDR:-0.0.0.0}"
URL="http://${HOST}:${PORT}"
LOG_FILE="${LOG_FILE:-/tmp/deep-research-agent-streamlit.log}"
PID_FILE="${PID_FILE:-/tmp/deep-research-agent-streamlit.pid}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-30}"
LAUNCH_LABEL="${LAUNCH_LABEL:-deep-research-agent.streamlit.${PORT}}"

if curl -fsS -I "$URL" >/dev/null 2>&1; then
  echo "Streamlit app is already running: $URL"
  exit 0
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required but was not found on PATH." >&2
  exit 127
fi

rm -f "$PID_FILE"
uv_path="$(command -v uv)"

if command -v launchctl >/dev/null 2>&1 && [[ "$(uname -s)" == "Darwin" ]]; then
  launchctl submit -l "$LAUNCH_LABEL" -o "$LOG_FILE" -e "$LOG_FILE" -- \
    /bin/bash -lc '
      echo $$ > "$1"
      cd "$2"
      exec "$3" --native-tls run --extra ui streamlit run deep_research_agent/ui.py \
        --server.headless true \
        --server.address "$4" \
        --server.port "$5"
    ' bash "$PID_FILE" "$ROOT_DIR" "$uv_path" "$BIND_ADDR" "$PORT"
else
  cd "$ROOT_DIR"
  nohup "$uv_path" --native-tls run --extra ui streamlit run deep_research_agent/ui.py \
    --server.headless true \
    --server.address "$BIND_ADDR" \
    --server.port "$PORT" \
    < /dev/null > "$LOG_FILE" 2>&1 &

  echo "$!" > "$PID_FILE"
fi

deadline=$((SECONDS + STARTUP_TIMEOUT))
while (( SECONDS < deadline )); do
  pid=""
  if [[ -s "$PID_FILE" ]]; then
    pid="$(cat "$PID_FILE")"
  fi

  if curl -fsS -I "$URL" >/dev/null 2>&1; then
    sleep 1
    if [[ -n "$pid" ]] && ! kill -0 "$pid" >/dev/null 2>&1; then
      continue
    fi

    if ! curl -fsS -I "$URL" >/dev/null 2>&1; then
      continue
    fi

    echo "Streamlit app started: $URL"
    [[ -n "$pid" ]] && echo "PID: $pid"
    echo "Log: $LOG_FILE"
    echo "PID file: $PID_FILE"
    exit 0
  fi

  if [[ -n "$pid" ]] && ! kill -0 "$pid" >/dev/null 2>&1; then
    echo "Streamlit app exited before it was ready. Recent log output:" >&2
    tail -n 80 "$LOG_FILE" >&2 || true
    exit 1
  fi

  sleep 1
done

echo "Timed out waiting for Streamlit app at $URL. Recent log output:" >&2
tail -n 80 "$LOG_FILE" >&2 || true
exit 1
