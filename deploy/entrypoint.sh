#!/usr/bin/env bash
# Run the API when SERVICE=api, otherwise the Streamlit UI.
set -e
PORT="${PORT:-8080}"

if [ "$SERVICE" = "api" ]; then
  exec uvicorn copilot.api.main:app --host 0.0.0.0 --port "$PORT"
else
  exec streamlit run src/copilot/app/demo.py \
    --server.port="$PORT" \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --browser.gatherUsageStats=false
fi
