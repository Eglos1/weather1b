#!/usr/bin/env bash
# Continuous visualizer upgrader.
# Watches runs/weather-gpu/{latest.pt,final.pt}; every time a checkpoint with a
# NEWER mtime appears (latest.pt after each epoch, final.pt at the end) it
# regenerates viz/index.html from the best available checkpoint.
cd /home/user/weather1b || exit 1
dir=runs/weather-gpu
last=""
while true; do
  pick=$(ls -t "$dir"/latest.pt "$dir"/final.pt 2>/dev/null | head -1)
  if [ -n "$pick" ] && [ -f "$pick" ]; then
    mt=$(stat -c %Y "$pick")
    if [ "$mt" != "$last" ]; then
      echo "[vizwatch] $(date -Is) regen from $pick (mt=$mt)" >> /tmp/viz_watch.log
      HSA_OVERRIDE_GFX_VERSION=10.3.0 venv-gpu/bin/python viz_dump.py \
        "$pick" data/gsod_daily_full.zip viz/index.html >> /tmp/viz_watch.log 2>&1
      echo "[vizwatch] done rc=$? at $(date -Is)" >> /tmp/viz_watch.log
      last=$mt
    fi
  fi
  sleep 45
done