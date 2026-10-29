#!/usr/bin/env bash
# hcTools 启动脚本：LLM-first 意图解析服务（POST /api/predict）
set -euo pipefail
cd "$(dirname "$0")"

PORT="${HC_PORT:-8084}"
HOST="${HC_HOST:-0.0.0.0}"

# 时间基准：置 1 → sports 相对日期（今天/明天/周X）用真实当天；默认评测双锚零漂移。
export HC_SPORTS_RUNTIME_TODAY="${HC_SPORTS_RUNTIME_TODAY:-1}"

# 先杀掉占用同一端口的旧进程（往往是上次没退干净的 uvicorn）
if command -v lsof >/dev/null 2>&1; then
  pids="$(lsof -t -iTCP:"$PORT" -sTCP:LISTEN 2>/dev/null || true)"
  if [ -n "$pids" ]; then
    echo "[run.sh] 端口 $PORT 被以下进程占用，先结束：$pids"
    kill $pids 2>/dev/null || true
    sleep 1
  fi
fi

exec uvicorn app.main:app --host "$HOST" --port "$PORT" --reload