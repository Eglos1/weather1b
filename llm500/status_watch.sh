#!/bin/bash
LOG=/tmp/llm_status_watch.log
echo "[watch] $(date -Is) status watcher started" > "$LOG"
while true; do
  /home/user/weather1b/venv-gpu/bin/python gen_status.py 2>>"$LOG"
  sleep 30
done
