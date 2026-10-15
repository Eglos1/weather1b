#!/usr/bin/env bash
# WeatherNet supervisor: keeps the stage-2 (1M-window) training alive across
# crashes and reboots by re-launching with --resume auto until a DONE marker
# appears. Safe to start at login/boot.
cd /home/user/weather1b || exit 1
LOG=/tmp/supervise2.log
OUT=runs/weather-gpu
echo "[super] $(date -Is) supervisor started" >> "$LOG"
while true; do
  if [ -f "$OUT/DONE" ]; then
    echo "[super] $(date -Is) DONE marker present -- exiting" >> "$LOG"
    exit 0
  fi
  echo "[super] $(date -Is) launching (auto-resume)" >> "$LOG"
  env HSA_OVERRIDE_GFX_VERSION=10.3.0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    venv-gpu/bin/python -u train.py --data data/gsod_daily_full.zip --stream \
    --data-limit 1000000 --holdout 4000 --optim adafactor --ada-lr 0.01 \
    --precision bf16 --batch-size 64 --grad-accum 1 --chunk-size 512 \
    --epochs 2 --save-every 2000 --resume auto --out "$OUT" \
    >> /tmp/gpu_big2.log 2>&1
  rc=$?
  echo "[super] $(date -Is) train exited rc=$rc" >> "$LOG"
  sleep 25
done