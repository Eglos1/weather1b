#!/usr/bin/env bash
# launch GPU training (Adafactor eps fixed) + viz auto-upgrader, fully detached
cd /home/user/weather1b || exit 1
export HSA_OVERRIDE_GFX_VERSION=10.3.0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

nohup setsid ./vizwatch.sh > /tmp/viz_watch2.log 2>&1 < /dev/null &
W=$!
nohup setsid venv-gpu/bin/python -u train.py \
  --data data/gsod_daily_full.zip --stream --data-limit 12000 \
  --optim adafactor --ada-lr 0.01 --precision bf16 \
  --batch-size 4 --grad-accum 8 --chunk-size 512 --epochs 3 \
  --out runs/weather-gpu > /tmp/gpu_train2.log 2>&1 < /dev/null &
T=$!
sleep 2
echo "watcher_pid=$W"
echo "train_pid=$T"
pgrep -af "[t]rain.py --data" | head -2
exit 0