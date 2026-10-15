#!/usr/bin/env bash
# Watchdog: probes Open-Meteo archive API; when unblocked, launches the
# sequential hourly bulk fetch. Runs forever silently until unblocked.
cd /home/user/weather1b || exit 1
PY=.venv/bin/python
probe() {
  $PY - "$@" <<'EOF'
import sys, urllib.request
try:
    req = urllib.request.Request(
        "https://archive-api.open-meteo.com/v1/archive?latitude=40.7794"
        "&longitude=-73.9692&start_date=1940-01-01&end_date=1940-12-31"
        "&hourly=temperature_2m,relative_humidity_2m,pressure_msl,wind_speed_10m"
        "&wind_speed_unit=ms&timezone=GMT",
        headers={"User-Agent": "weather1b-probe/0.2"})
    with urllib.request.urlopen(req, timeout=30) as r:
        r.read()
    print("UNBLOCKED")
except Exception:
    print("BLOCKED")
EOF
}
intervals=0
while :; do
  r=$(probe)
  if [ "$r" = "UNBLOCKED" ]; then
    echo "$(date +%FT%T) probe OK -- launching hourly fetch"
    exec $PY fetch_all.py --only hourly --workers 1
    break
  fi
  intervals=$((intervals+1))
  if [ $((intervals % 15)) -eq 0 ]; then
    echo "$(date +%FT%T) still blocked (interval $intervals)"
  fi
  sleep 60
done