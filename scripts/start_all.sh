#!/bin/bash
# Start the whole fraud system with one command (added in Part 2 · Step 6; Notification added in Step 7).
#   bash scripts/start_all.sh     start everything (skips anything already running)
#   bash scripts/status.sh        what is running, and each program's last log line
#   bash scripts/stop_all.sh      stop everything (Docker keeps running)
# Each program runs in the background; its output goes to data/logs/<name>.log.

cd "$(dirname "$0")/.." || exit 1
PY=.venv/bin/python
mkdir -p data/logs data/run

# name | command  (kept in this order: services first, then agents, then the Monitor)
PROGRAMS="
gateway|$PY -m uvicorn fraud_agents.services.gateway.app:app --port 8000
sms-provider|$PY -m uvicorn fraud_agents.services.sms.provider:app --port 8100
audit|$PY -m fraud_agents.services.audit.service
risk-scoring|$PY -m fraud_agents.agents.risk_scoring.agent
decision|$PY -m fraud_agents.agents.decision.agent
investigation|$PY -m fraud_agents.agents.investigation.agent
verification|$PY -m fraud_agents.agents.verification.agent
notification|$PY -m fraud_agents.agents.notification.agent
monitor|$PY -m streamlit run src/fraud_agents/monitor/app.py
"

# 1. Docker (Redis and PostgreSQL)
if [ -z "$SKIP_DOCKER" ]; then
  if ! docker info >/dev/null 2>&1; then
    echo "Starting Docker Desktop ..."
    open -a Docker
    for i in $(seq 1 60); do docker info >/dev/null 2>&1 && break; sleep 2; done
  fi
  docker compose up -d >/dev/null 2>&1 || { echo "Could not start Redis/PostgreSQL (docker compose up -d)"; exit 1; }
  echo -n "Waiting for Redis and PostgreSQL to be healthy "
  for i in $(seq 1 30); do
    [ "$(docker compose ps --format '{{.Health}}' | grep -c healthy)" -ge 2 ] && break
    echo -n "."; sleep 2
  done
  echo " ok"
fi

# 2. Ollama (Llama for the AI agents)
if [ -z "$SKIP_OLLAMA" ] && ! curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
  echo "Starting Ollama ..."
  open -a Ollama
  for i in $(seq 1 30); do curl -s http://localhost:11434/api/tags >/dev/null 2>&1 && break; sleep 1; done
fi

# 3. Our programs
echo "$PROGRAMS" | while IFS='|' read -r name cmd; do
  [ -z "$name" ] && continue
  pidfile="data/run/$name.pid"
  if [ -f "$pidfile" ] && kill -0 "$(cat "$pidfile")" 2>/dev/null; then
    echo "  $name: already running"
    continue
  fi
  nohup $cmd >"data/logs/$name.log" 2>&1 &
  echo $! >"$pidfile"
  echo "  $name: started (log: data/logs/$name.log)"
  sleep 1
done

sleep 6
echo
bash scripts/status.sh
echo
echo "Monitor: http://localhost:8501   (open it with: open http://localhost:8501)"
