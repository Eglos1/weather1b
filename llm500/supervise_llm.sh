#!/usr/bin/env bash
# LLM-500 supervisor: waits until the GPU is actually free (weather run
# finished or no weather job is hogging it) and free VRAM >= 3.0 GB, then
# launches cuikal the LLM training with --resume auto. Reruns forever until
# a DONE marker appears. Never touches the weather run.
cd /home/user/weather1b/llm500 || exit 1
LOG=/tmp/llm_super.log
OUT=runs/llm500
echo "[super] $(date -Is) llm supervisor started" >> "$LOG"
FAILS=0

vram_free() {
  local tot="" used=""
  for f in /sys/class/drm/card*/device/mem_info_vram_total; do
    [ -f "$f" ] || continue
    tot=$(cat "$f")
    used=$(cat "${f%_total}_used")
    # first card that responds to sysfs
    if [ "${tot:-0}" -gt 0 ] 2>/dev/null; then break; fi
  done
  if [ "${tot:-0}" -gt 0 ] && [ "${used:-0}" -ge 0 ] 2>/dev/null; then
    local free_mb=$(( (tot - used) / 1024 / 1024 ))
    [ "$free_mb" -ge 5000 ] && echo 1 || { echo "[diag] vram free ${free_mb}MB (needs >=5000)" >> "$LOG"; echo 0; }
  else
    echo 1   # no readable vram counter -> assume free
  fi
}
weather_busy() { pgrep -f "venv-gpu/bin/python -u train_llm.py" >/dev/null && return 1; \
                 pgrep -f "venv-gpu/bin/python -u train.py" >/dev/null; }
llm_running() { pgrep -f "venv-gpu/bin/python -u train_llm.py" >/dev/null; }

while true; do
  if [ -f "$OUT/DONE" ]; then
    echo "[super] $(date -Is) DONE -- exiting" >> "$LOG"; exit 0
  fi
  if [ ! -x /home/user/weather1b/venv-gpu/bin/python ]; then
    sleep 30; continue
  fi
  free_ok=$(vram_free)
  if [ "$free_ok" = "1" ] && ! weather_busy && ! llm_running; then
    # wait until there is a real corpus, then rebuild BPE/train.bin
    txt_mb=$(du -sm data/text 2>/dev/null | cut -f1)
    if [ "${txt_mb:-0}" -lt 30 ]; then
      echo "[super] $(date -Is) corpus only ${txt_mb}MB, waiting 60s" >> "$LOG"
      sleep 60
      continue
    fi
    echo "[super] $(date -Is) rebuilding corpus+vocab, then launching" >> "$LOG"
    /home/user/weather1b/venv-gpu/bin/python build_bpe.py --incremental >> /tmp/llm_bpe.log 2>&1
    if [ "$?" -ne 0 ]; then
      echo "[super] $(date -Is) build_bpe FAILED, will retry in 60s" >> "$LOG"
      sleep 60
      continue
    fi
    echo "[super] $(date -Is) GPU free, launching llm (resume auto)" >> "$LOG"
    nice -n 10 ionice -c 3 env HSA_OVERRIDE_GFX_VERSION=10.3.0 \
      PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      /home/user/weather1b/venv-gpu/bin/python -u train_llm.py \
      --out "$OUT" --batch-size 4 --grad-accum 8 --seq 512 --lr 3e-4 \
      --max-steps 250000 --save-every 500 --val-every 250 \
      >> /tmp/llm_train.log 2>&1
    rc=$?
    if [ "$rc" -ne 0 ]; then
      FAILS=$((FAILS+1))
    else
      FAILS=0
    fi
    sleep $(( FAILS > 0 ? 300 : 0 ))
    echo "[super] $(date -Is) llm exited rc=$rc (fails=$FAILS)" >> "$LOG"
  else
    echo "[super] $(date -Is) GPU busy (weather), waiting 90s" >> "$LOG"
    sleep 90
  fi
  if [ -f "$OUT/DONE" ]; then
    echo "[super] $(date -Is) DONE -- exiting" >> "$LOG"; exit 0
  fi
done