import html, json, os
def tail(p,n=10):
    try: return open(p,encoding='utf-8',errors='replace').read().splitlines()[-n:]
    except Exception: return ["(no log yet)"]
def sysfs(card,kind,node):
    p=f"/sys/class/drm/card{card}/device/mem_info_vram_{kind.replace('-','_')}"
    try: return int(open(p).read())
    except Exception: return 0
used=0; tot=0; busy=0
for c in (1,0):
    u=sysfs(c,'used',''); t=sysfs(c,'total','')
    if t>0: used,tot=u,t; break
try: busy=int(open("/sys/class/drm/card1/device/gpu_busy_percent").read())
except Exception: busy=-1
fs={}
try: fs=json.load(open("data/fetch_state.json"))
except Exception: pass
off=fs.get("clean:train",{}).get("offset",0); tot_r=fs.get("clean:train",{}).get("total","?") or "?"
tlog=html.escape("<br>".join(tail("/tmp/llm_train.log")))
slog=html.escape("<br>".join(tail("/tmp/llm_super.log")))
tbinom="?"; 
import glob, numpy as np
bins=glob.glob("data/bin/train.bin")
if bins:
    tbinom=f"{os.path.getsize(bins[0])//2/1e6:.1f}M tok"
page=f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<meta http-equiv="refresh" content="12"><title>TS-LLM — People's Speech Text (500M) · LIVE</title>
<style>body{{background:#0e1117;color:#d6deeb;font:14px/1.5 ui-monospace,Menlo,monospace;padding:36px 24px}}
.box{{max-width:880px;margin:0 auto;background:#151a25;border:1px solid #2a3444;border-radius:12px;padding:22px 26px}}
h1{{font-size:20px;color:#7aa2ff;margin:0 0 8px}} .live{{color:{'#7aa2ff' if busy>60 else '#ff9f6e'};animation:blink 2s infinite;font-size:13px}}
@keyframes blink{{50%{{opacity:.25}}}}
.card{{background:#0b0f16;border:1px solid #2a3444;border-radius:10px;padding:12px 16px;margin:10px 0}}
.card b{{font-size:21px;color:#7aa2ff}} pre{{background:#0b0f16;border:1px solid #2a3444;border-radius:8px;padding:12px;font-size:12px;white-space:pre-wrap}}
.mut{{color:#8aa0b8;font-size:12.5px}}</style></head><body><div class="box">
<h1>People's-Speech-Text LLM — 483.8M params</h1>
<div class="live">● GPU busy {busy}% &nbsp;·&nbsp; VRAM {used/2**30:.1f}/{tot/2**30:.1f} GiB &nbsp;·&nbsp; train.bin {tbinom} &nbsp;·&nbsp; transcripts {(off/1e3):,.0f}k / {int(tot_r):,} train</div>
<div class="card"><b>{"TRAINING NOW" if busy>60 else "idle"}</b> &nbsp;24L x1280 · bf16 · Adafactor · auto-resume + crash-guard supervisor</div>
<h3>trainer log</h3><pre>{tlog}</pre>
<h3>supervisor log</h3><pre>{slog}</pre>
<p class="mut">WeatherNet stage-2 finished 18:38 — RMSE 3.70 °C. LLM took the GPU over at 06:54. (Watchdog: 30 s.)</p>
</div></body></html>"""
open("status/index.html","w").write(page)
