#!/bin/bash
# Stop every program of the fraud system (Docker and Ollama keep running).
# Nothing is lost: an event not finished yet stays in Redis and is picked up on the next start;
# paused verifications stay saved in PostgreSQL.
cd "$(dirname "$0")/.." || exit 1
for pidfile in data/run/*.pid; do
  [ -f "$pidfile" ] || continue
  name=$(basename "$pidfile" .pid)
  pid=$(cat "$pidfile")
  if kill -0 "$pid" 2>/dev/null; then
    kill -TERM "$pid" 2>/dev/null         # ask it to stop
    for i in 1 2 3 4 5 6 7 8 9 10; do kill -0 "$pid" 2>/dev/null || break; sleep 0.5; done
    kill -0 "$pid" 2>/dev/null && kill -KILL "$pid" 2>/dev/null   # still running after 5 s: force it
    echo "  $name: stopped"
  fi
  rm -f "$pidfile"
done
echo "All stopped. (Docker is still running; to stop it too: docker compose stop)"
