#!/bin/bash
# Show which programs of the fraud system are running, with each one's last log line.
cd "$(dirname "$0")/.." || exit 1
for name in gateway sms-provider audit risk-scoring decision investigation verification monitor; do
  pidfile="data/run/$name.pid"
  if [ -f "$pidfile" ] && kill -0 "$(cat "$pidfile")" 2>/dev/null; then state="running"; else state="STOPPED"; fi
  last=$(tail -n 1 "data/logs/$name.log" 2>/dev/null | cut -c1-110)
  printf "%-14s %-8s %s\n" "$name" "$state" "$last"
done
