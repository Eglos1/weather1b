#!/usr/bin/env bash
# supervise_weather_cpu.sh
# ========================
# Supervisor for the CPU-only WeatherNet trainer.  Waits until the hourly
# cache exists (build_hourly_cache.py), then launches train_weather_cpu.py
# in the POLITEST mode:
#     nice -n 15 ionice -c 3
# (the LLM trainer gets nice 10 / GPU; this job yields to both the user's
#  foreground apps and the LLM, and only touches CPU cores).
#
# Reruns forever until runs/weather-cpu/DONE appears, exactly like the LLM
# supervisor.  Safe against crashes/reboots thanks to --resume auto.
#
# Launch once, leave running:
#     bash supervise_weather_cpu.sh
cd /home/user/weather1b || exit 1
LOG=/tmp/weather_cpu_super.log
TLOG=/tmp/weather_cpu_train.log
OUT=runs/weather-cpu
CACHE=hourly_cache
PY=/home/user/weather1b/venv-gpu/bin/python

echo "[super] $(date -Is) weather-cpu supervisor started" >> "$LOG"
FAILS=0

cpu_running() { pgrep -f "venv-gpu/bin/python -u train_weather_cpu.py" >/dev/null; }
build_running() { pgrep -f "python -u build_hourly_cache.py" >/dev/null; }

while true; do
  if [ -f "$OUT/DONE" ]; then
    echo "[super] $(date -Is) DONE -- exiting" >> "$LOG"; exit 0
  fi
  if [ ! -f "$CACHE/manifest.json" ]; then
    if build_running; then
      echo "[super] $(date -Is) cache still building, waiting 60s" >> "$LOG"
    else
      echo "[super] $(date -Is) cache missing and no build running -- starting it" >> "$LOG"
      nice -n 15 ionice -c 3 "$PY" -u build_hourly_cache.py \
        --workers 14 --cache-dir "$CACHE" >> /tmp/weather_cache_build.log 2>&1
      sleep 60
    fi
    sleep 60
    continue
  fi
  if cpu_running; then
    sleep 90
    continue
  fi
  echo "[super] $(date -Is) launching weather-cpu trainer (resume auto)" >> "$LOG"
  nice -n 15 ionice -c 3 "$PY" -u train_weather_cpu.py \
    --cache-dir "$CACHE" --out "$OUT" --resume auto \
    --threads 12 --batch-size 32 --grad-accum 1 \
    --optim adafactor --ada-lr 0.01 \
    --init-ckpt runs/weather-gpu/final.pt --eval-limit 2000 \
    --max-steps 250000 --save-every 500 --val-every 250 \
    >> "$TLOG" 2>&1
  rc=$?
  if [ "$rc" -ne 0 ]; then FAILS=$((FAILS+1)); else FAILS=0; fi
  echo "[super] $(date -Is) trainer exited rc=$rc (fails=$FAILS)" >> "$LOG"
  sleep $(( FAILS > 0 ? 300 : 30 ))
done