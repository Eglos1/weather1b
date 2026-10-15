#!/bin/bash
# Watchdog: keeps the NOAA ISD-Lite hourly harvest alive until finished.
# Relaunches fetch_hourly_all.py (resumable) every time it dies; exits when
# the state file says there is nothing left to fetch.
cd /home/user/weather1b || exit 1
LOG=/tmp/hourly_super.log
STATE=/mnt/biggiecheese/weather1b/hourly/isdfetch_state.json
echo "[super] $(date -Is) hourly supervisor started" >> "$LOG"
while true; do
  if ! pgrep -f "fetch_hourly_all.py" >/dev/null; then
    if [ -f "$STATE" ] && grep -q '"done"' "$STATE"; then
      # check completeness heuristically: state has stations count vs task count
      :
    fi
    echo "[super] $(date -Is) launching harvest" >> "$LOG"
    nice -n 10 /home/user/weather1b/venv-gpu/bin/python \
      /home/user/weather1b/fetch_hourly_all.py >> /tmp/hourly_fetch.log 2>&1
    rc=$?
    if [ "$rc" -eq 0 ]; then
      echo "[super] $(date -Is) harvest COMPLETE rc=0" >> "$LOG"
      break
    fi
    echo "[super] $(date -Is) harvest exited rc=$rc (restart in 30s)" >> "$LOG"
    sleep 30
  else
    sleep 60
  fi
done