"""
viz_dump.py
===========
Dumps a trained WeatherNet checkpoint into a SELF-CONTAINED interactive
HTML visualizer (no CDN, works from file://).

The hero view is a HEATMAP OF ALL MODEL WEIGHTS: every ternary weight matrix
(encoder + 14 latent blocks + projector) is binned into a 120x120 grid and
painted as a heat tile -- green where weights fire +1, red where they fire -1,
dark where they are dead (below the ternary threshold). Click a tile to zoom,
or press play to auto-cycle through every layer. Secondary panels show the
mean |activation| firing profile per layer, the ternary structure table, and
real station-window forecasts (inputs vs prediction vs ground truth).

Usage:  python viz_dump.py [checkpoint.pt] [archive.zip] [out.html]
Runs on CPU so it does not disturb a running GPU training job.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data import WeatherZipCSV, Normalizer          # noqa: E402
from model import WeatherNet, WeatherNetConfig       # noqa: E402
from layers import BitLinear                         # noqa: E402


def layer_info(layer: BitLinear) -> dict:
    """Ternary structure of one BitLinear latent weight."""
    w = layer.weight.detach()
    beta = float(w.abs().mean().item())
    thr = float(layer.ternary_threshold) * beta if layer.ternary_threshold > 0 else 0.0
    pos = int((w > thr).sum().item())
    neg = int((w < -thr).sum().item())
    total = w.numel()
    return {
        "beta": round(beta, 5),
        "thr": round(thr, 7),
        "pos": pos, "neg": neg, "zero": total - pos - neg,
        "active": (pos + neg) / total,
    }


def binned_heat(w: torch.Tensor, thr: float, rows: int = 120, cols: int = 120) -> str:
    """
    Bin a weight matrix into a `rows`x`cols` grid and pack two metrics per cell:
      * signed firing bias  s = (n_pos - n_neg)/n   in [-1, 1]
      * firing density     d = (n_pos + n_neg)/n   in [0, 1]
    packed interleaved as sgn8, den8 bytes, base64-encoded.
    """
    a = w.detach().cpu().float().numpy()
    nr, nc = a.shape
    ri = np.minimum(np.arange(nr) * rows // nr, rows - 1)
    ci = np.minimum(np.arange(nc) * cols // nc, cols - 1)
    grid = ri[:, None] * cols + ci[None, :]
    cnt = np.bincount(grid.ravel(), minlength=rows * cols).reshape(rows, cols).astype(np.float64)
    p = a > thr
    n = a < -thr
    pc = np.bincount(grid[p], minlength=rows * cols).reshape(rows, cols) if p.any() \
        else np.zeros((rows, cols), np.float64)
    nc_ = np.bincount(grid[n], minlength=rows * cols).reshape(rows, cols) if n.any() \
        else np.zeros((rows, cols), np.float64)
    den = np.zeros_like(cnt)
    sgn = np.zeros_like(cnt)
    m = cnt > 0
    den[m] = (pc[m] + nc_[m]) / cnt[m]
    sgn[m] = (pc[m] - nc_[m]) / cnt[m]
    sgn8 = np.round((sgn * 0.5 + 0.5) * 255).astype(np.uint8)
    den8 = np.round(den * 255).astype(np.uint8)
    inter = np.empty((rows * cols * 2,), np.uint8)
    inter[0::2] = sgn8.ravel()
    inter[1::2] = den8.ravel()
    return base64.b64encode(inter.tobytes()).decode()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", nargs="?", default="runs/weather/final.pt")
    ap.add_argument("archive", nargs="?", default="data/gsod_daily_full.zip")
    ap.add_argument("out", nargs="?", default="viz/index.html")
    ap.add_argument("--windows", type=int, default=6)
    args = ap.parse_args()

    ckpt_path = os.path.abspath(args.checkpoint)
    archive_path = os.path.abspath(args.archive)
    if not os.path.exists(archive_path):
        raise SystemExit(f"archive not found: {archive_path}")

    dev = torch.device("cpu")
    ckpt = torch.load(ckpt_path, map_location=dev, weights_only=False)
    cfg = WeatherNetConfig(**ckpt["config"])
    model = WeatherNet(cfg)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    normalizer = Normalizer()
    normalizer.load_state_dict(ckpt["normalizer"])

    ds = WeatherZipCSV(archive_path)
    raw = [w for w in ds.iter_windows(limit=args.windows * 40)][:args.windows]
    if not raw:
        raise SystemExit("no windows found in archive")

    order = [("embed", model.feature_embed), ("encoder", model.encoder)]
    for i, blk in enumerate(model.blocks):
        order.append((f"block{i:02d}", blk.bit_linear))
    order.append(("proj", model.projector))
    tags = [t for t, _ in order]

    fire: dict[str, list[float]] = {}

    def make_hook(tag: str):
        def hook(_mod, _inp, out):
            fire.setdefault(tag, []).append(float(out.detach().float().abs().mean().item()))
        return hook

    heat_layers = []          # ternary layers only: base64 heat grid + stats
    fire_index = []           # tags of ternary layers (for heat tiles)
    for tag, mod in order:
        mod.register_forward_hook(make_hook(tag))
        if isinstance(mod, BitLinear):
            w = mod.weight.detach()
            beta = float(w.abs().mean().item())
            thr = float(mod.ternary_threshold) * beta if mod.ternary_threshold > 0 else 0.0
            heat_layers.append({
                "tag": tag,
                "shape": f"{w.shape[1]}x{w.shape[0]}",
                "beta": round(beta, 5),
                "thr": round(thr, 7),
                "rows": 120, "cols": 120,
                "b64": binned_heat(w, thr),
                "info": layer_info(mod),
            })
            fire_index.append(tag)

    t0 = time.time()
    samples = []
    with torch.no_grad():
        for x, ctx, y in raw:
            xw_t = torch.tensor(normalizer.transform_x(x), dtype=torch.float32)
            ctx_t = torch.tensor(normalizer.transform_ctx(ctx), dtype=torch.float32)
            pred = model(xw_t.unsqueeze(0), ctx_t.unsqueeze(0)).squeeze(0)
            pred_phys = normalizer.inverse_y(pred.numpy())
            y_phys = normalizer.inverse_y(y)
            rmse = float(np.sqrt(np.mean((pred_phys - y_phys) ** 2)))
            corr = float(np.corrcoef(pred_phys, y_phys)[0, 1]) if y_phys.std() > 0 else 0.0
            samples.append({
                "x": [float(v) for v in x[:, 0]],
                "x_pressure": [float(v) for v in x[:, 1]],
                "pred": [float(v) for v in pred_phys],
                "target": [float(v) for v in y_phys],
                "rmse": round(rmse, 3),
                "corr": round(corr, 3),
            })
    secs = round(time.time() - t0, 1)

    status = "no log"
    try:
        with open("/tmp/gpu_train.log") as fh:
            lines = [l.rstrip("\n") for l in fh]
        picks = [l for l in lines if l.startswith(("[run]", "[epoch"))]
        status = (" | ".join(picks[-3:])) if picks else "mid epoch 0"
    except OSError:
        pass

    profile = []
    for tag in fire_index:
        v = fire.get(tag) or []
        profile.append({"tag": tag, "mean": float(np.mean(v)) if v else 0.0,
                        "peak": float(np.max(v)) if v else 0.0})

    nfn = sum(p.numel() for p in model.parameters() if not torch.isfinite(p).all())
    warn = ""
    if nfn > 0:
        warn = (f"CAUTION: {nfn:,}/{sum(p.numel() for p in model.parameters()):,} "
                f"parameters in this checkpoint are NaN (training diverged). "
                f"Heatmaps/forecasts below are meaningless for this checkpoint.")

    data = {
        "ckpt": os.path.basename(ckpt_path),
        "archive": os.path.basename(archive_path),
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "infer_secs": secs,
        "n_params": sum(p.numel() for p in model.parameters()),
        "n_ternary": sum(p.numel() for m in model.modules() if isinstance(m, BitLinear)
                         for p in (m.weight,)),
        "blocks": cfg.num_hidden_blocks,
        "width": cfg.d_hidden,
        "heat": heat_layers,
        "fire_index": fire_index,
        "profile": profile,
        "samples": samples,
        "train_status": status,
        "warn": warn,
        "active_overall": float(np.mean([l["info"]["active"] for l in heat_layers])),
    }

    html = render_html(data)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(html)
    mb = len(html) / 1e6
    print(f"wrote {os.path.abspath(args.out)} ({mb:.1f} MB, inference {secs}s)")


JS = r"""
const D = __DATA__;
const $ = id => document.getElementById(id);
const fmt = (n, d=2) => (typeof n==='number' ? n.toFixed(d) : n);
const prop = pct => pct.toFixed(1) + '%';
const BG = [28, 34, 48];
const POS = [0x5a, 0xd1, 0xb0];
const NEG = [0xff, 0x6b, 0x6b];

/* ---------- decode base64-packed heat grid ---------- */
function decodeHeat(b64, rows, cols){
  const bin = atob(b64), n = rows*cols;
  const sgn = new Float32Array(n), den = new Float32Array(n);
  for(let i=0;i<n;i++){
    sgn[i] = bin.charCodeAt(2*i)/255 - 0.5;
    den[i] = bin.charCodeAt(2*i+1)/255;
  }
  return {sgn, den, rows, cols};
}
function paintHeat(ctx, h, px){
  const {sgn, den, rows, cols} = h;
  const img = ctx.createImageData(cols, rows);
  for(let y=0;y<rows;y++){
    for(let x=0;x<cols;x++){
      const k = y*cols + x, s = sgn[k], f = den[k];
      const sc = 0.55 + 0.45*f;             // dead zones stay dark
      let r=BG[0], g=BG[1], b=BG[2];
      if(s >= 0){ const t = Math.min(s/0.5,1);
        r = BG[0]+(POS[0]-BG[0])*t; g = BG[1]+(POS[1]-BG[1])*t; b = BG[2]+(POS[2]-BG[2])*t; }
      else { const t = Math.min(-s/0.5,1);
        r = BG[0]+(NEG[0]-BG[0])*t; g = BG[1]+(NEG[1]-BG[1])*t; b = BG[2]+(NEG[2]-BG[2])*t; }
      const o = (y*cols + x)*4;
      img.data[o]=(isFinite(r)?r*sc:0); img.data[o+1]=(isFinite(g)?g*sc:0);
      img.data[o+2]=(isFinite(b)?b*sc:0); img.data[o+3]=255;
    }
  }
  ctx.putImageData(img, 0, 0);
}

/* ---------- cards ---------- */
function initCards(){
  $('cParams').innerHTML = `<b>${fmt(D.n_params/1e9,2)}B</b><span>PARAMS</span>`;
  $('cTernary').innerHTML = `<b>${fmt(D.n_ternary/1e6,0)}M</b><span>TERNARY WEIGHTS</span>`;
  $('cActive').innerHTML = `<b>${prop(D.active_overall*100)}</b><span>GLOBAL ACTIVE (≠0)</span>`;
  $('cInfer').innerHTML = `<b>${D.infer_secs}s</b><span>CPU INFERENCE</span>`;
  $('cTrain').innerHTML = `<b style="font-size:13px">${D.train_status}</b><span>GPU TRAIN</span>`;
}

/* ---------- hero: heatmap of every ternary weight matrix ---------- */
function buildColors(){
  const c = document.createElement('canvas'); c.width=c.height=1;
  const g = c.getContext('2d');
  const grad = g.createLinearGradient(0,0,256,0);
  grad.addColorStop(0,  '#ff6b6b'); grad.addColorStop(0.5,'#1c2230');
  grad.addColorStop(1,  '#5ad1b0');
  g.fillStyle = grad; g.fillRect(0,0,256,1);
  const img = g.getImageData(0,0,256,1).data;
  for(let i=0;i<256;i++){
    const s=document.createElement('span'); s.className='sw';
    s.style.background=`rgb(${img[i*4]},${img[i*4+1]},${img[i*4+2]})`;
    $('legend').appendChild(s);
  }
  const lab = ['−1 (negative)', '0 (dead zone)', '+1 (positive)'];
  lab.forEach((t,i)=>{
    const s=document.createElement('span'); s.textContent=t; s.style.marginLeft=i? '8px':'4px';
    $('legend').appendChild(s);
  });
}

function tileGlow(meanAct, maxAct){
  const t = maxAct>0 ? meanAct/maxAct : 0;
  const col = t>0.75? '34,215,184' : t>0.35? '108,154,140' : '42,52,68';
  return `0 0 ${6+16*t}px rgba(${col},${0.25+0.45*t})`;
}

function buildTiles(){
  const wrap = $('heatmaps');
  if(D.warn){
    const d = document.createElement('div');
    d.className = 'panel'; d.style.gridColumn = '1 / -1';
    d.innerHTML = `<b style="color:var(--hot)">No live signal to paint.</b> ${D.warn}<br>`+
      `<span style="color:var(--mut)">The heatmap grid re-appears automatically when a clean `+
      `checkpoint lands — this page is regenerated each epoch and the tab reloads it.</span>`;
    wrap.appendChild(d);
    return;
  }
  const maxAct = Math.max(...D.profile.map(p=>p.mean), 1e-9);
  D.heat.forEach((L, idx)=>{
    const h = decodeHeat(L.b64, L.rows, L.cols);
    const act = D.profile[idx].mean;
    const tile = document.createElement('div');
    tile.className = 'tile';
    tile.style.boxShadow = tileGlow(act, maxAct);
    const cv = document.createElement('canvas');
    cv.width = L.cols; cv.height = L.rows;
    paintHeat(cv.getContext('2d'), h, 1);
    tile.appendChild(cv);
    const label = document.createElement('div');
    label.className = 'tilelabel';
    tile.appendChild(label);
    tile.onclick = () => zoom(idx, tile);
    wrap.appendChild(tile);
  });
}
function fixLabels(){ // simpler re-render of label counts to avoid the inline mess
  document.querySelectorAll('.tile').forEach((tile, idx)=>{
    const L = D.heat[idx], inf = L.info, tot = inf.pos+inf.neg+inf.zero;
    tile.querySelector('.tilelabel').innerHTML =
      `<b>${L.tag}</b> <span>${L.shape}</span>` +
      `<span>β ${L.beta} · thr ${L.thr}</span>` +
      `<span>|act| ${D.profile[idx].mean.toFixed(4)}</span>` +
      `<span>+1 ${prop(inf.pos/tot*100)} · −1 ${prop(inf.neg/tot*100)} · 0 ${prop(inf.zero/tot*100)}</span>` +
      `<span>active ${prop(inf.active*100)}</span>`;
  });
}

/* ---------- lightbox zoom + auto-cycle ---------- */
let cur = 0, playing = true, timer = null;
function pause(){
  playing = false;
  const b = $('playBtn'); if (b) b.textContent = '▶ play';
}
function zoom(idx, tile){
  cur = idx; pause();
  const L = D.heat[cur], h = decodeHeat(L.b64, L.rows, L.cols);
  const cv = $('lens'); const px = 4;
  cv.width = h.cols*px; cv.height = h.rows*px;
  const ctx = cv.getContext('2d');
  const c2 = document.createElement('canvas'); c2.width=h.cols; c2.height=h.rows;
  paintHeat(c2.getContext('2d'), h, 1);
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(c2, 0, 0, h.cols, h.rows, 0, 0, h.cols*px, h.rows*px);
  const o = document.getElementById('lensBox');
  o.classList.add('open');
  document.getElementById('lensTitle').innerHTML =
    `<b>${L.tag}</b> — ${L.shape} · β ${L.beta} · active ${prop(L.info.active*100)} ` +
    `· +1 ${prop((L.info.pos/(L.info.pos+L.info.neg+L.info.zero))*100)} / −1 ${prop((L.info.neg/(L.info.pos+L.info.neg+L.info.zero))*100)}`;
}
function play(){
  playing = !playing;
  $('playBtn').textContent = playing ? '⏸ pause' : '▶ play';
  if(playing) step();
}
function step(){
  if(!playing) return;
  const tile = document.querySelectorAll('.tile')[cur];
  zoom(cur, tile);
  cur = (cur+1) % D.heat.length;
}
document.addEventListener('DOMContentLoaded', ()=>{
  if(D.warn){ const b=$('warn'); b.textContent=D.warn; b.classList.add('show'); }
  initCards(); buildColors(); buildTiles(); fixLabels(); buildProfile(); buildTable(); buildSamples();
  $('playBtn').addEventListener('click', play);
  timer = setInterval(step, 2000);
});

/* ---------- firing profile (mean |activation| across windows) ---------- */
function buildProfile(){
  const wrap = $('profile'), max = Math.max(...D.profile.map(p=>p.mean), 1e-9);
  D.profile.forEach(p=>{
    const bar = document.createElement('div');
    bar.className = 'bar'; bar.style.height = `${p.mean/max*100}%`;
    bar.title = `${p.tag}: mean|act| = ${p.mean.toFixed(4)}`;
    wrap.appendChild(bar);
  });
}

/* ---------- full weight-structure table ---------- */
function buildTable(){
  const rows = D.heat.map((L,i)=>{
    const inf = L.info, tot = inf.pos+inf.neg+inf.zero;
    return `<tr><td>${L.tag}</td><td>${L.shape}</td><td>${L.beta}</td>`+
      `<td>${prop(inf.active*100)}</td><td>${prop(inf.pos/tot*100)}</td>`+
      `<td>${prop(inf.neg/tot*100)}</td><td>${prop(inf.zero/tot*100)}</td></tr>`;
  }).join('');
  $('layerTable').innerHTML =
    `<tr><th>Layer</th><th>shape</th><th>β</th><th>active</th><th>+1</th><th>−1</th><th>0</th></tr>`+rows;
}

/* ---------- sample windows: inputs vs forecast vs truth ---------- */
function buildSamples(){
  const wrap = $('sampleCharts');
  D.samples.forEach((s,i)=>{
    const div = document.createElement('div');
    div.className = 'panel';
    div.innerHTML =
      `<h3>Window #${i+1} — RMSE <b class="pos">${s.rmse}°C</b> · corr <b>${s.corr}</b></h3>`+
      `<canvas id="cv${i}"></canvas>`;
    wrap.appendChild(div);
  });
  requestAnimationFrame(()=>D.samples.forEach((s,i)=>drawSample(s,i)));
}
function drawSample(s, i){
  const cv = document.getElementById('cv'+i);
  if(!cv) return;
  const W = cv.clientWidth || 740, H = 190;
  cv.width = W; cv.height = H;
  const ctx = cv.getContext('2d');
  const pad = {l:36, r:10, t:14, b:26};
  ctx.clearRect(0,0,W,H);
  if(!isFinite(s.rmse)){
    ctx.fillStyle = '#8aa0b8'; ctx.font='13px monospace';
    ctx.fillText('model output non-finite — this checkpoint collapsed. see banner.', pad.l, H/2);
    return;
  }
  const all = [...s.pred, ...s.target];
  const lo = Math.min(...s.x, ...all) - 2, hi = Math.max(...s.x, ...all) + 2;
  const sx = t => pad.l + t/(71-1)*(W-pad.l-pad.r);
  const syv = v => pad.t + (1-(v-lo)/(hi-lo))*(H-pad.t-pad.b);
  ctx.strokeStyle = '#2a3444'; ctx.fillStyle = '#8aa0b8';
  ctx.strokeRect(pad.l, pad.t, W-pad.l-pad.r, H-pad.t-pad.b);
  ctx.font = '10px monospace';
  for(let g=0; g<=4; g++){ const v = lo+(hi-lo)*g/4; ctx.fillText(v.toFixed(0), 4, syv(v)+3); }
  ctx.beginPath();
  s.x.forEach((v,t)=> t===0 ? ctx.moveTo(sx(t), syv(v)) : ctx.lineTo(sx(t), syv(v)));
  ctx.strokeStyle = '#8aa0b8'; ctx.lineWidth = 1.4; ctx.stroke();
  ctx.beginPath();
  s.pred.forEach((v,t)=> t===0 ? ctx.moveTo(sx(t+24), syv(v)) : ctx.lineTo(sx(t+24), syv(v)));
  ctx.strokeStyle = '#5ad1b0'; ctx.lineWidth = 1.8; ctx.stroke();
  ctx.beginPath();
  s.target.forEach((v,t)=> t===0 ? ctx.moveTo(sx(t+24), syv(v)) : ctx.lineTo(sx(t+24), syv(v)));
  ctx.strokeStyle = '#ffb454'; ctx.lineWidth = 1.8; ctx.setLineDash([4,4]); ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = '#8aa0b8'; ctx.fillText('24h input', pad.l+4, H-6);
  ctx.fillStyle = '#5ad1b0'; ctx.fillText('prediction', W*0.5, H-6);
  ctx.fillStyle = '#ffb454'; ctx.fillText('ground truth', W*0.68, H-6);
}
"""


TEM = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>WeatherNet 1.58-bit — full weight heatmap</title>
<style>
 :root { --bg:#0e1117; --panel:#151a25; --panel2:#1c2230; --line:#2a3444;
         --txt:#d6deeb; --mut:#8aa0b8; --acc:#5ad1b0; --warm:#ffb454; --hot:#ff6b6b; }
 * { box-sizing:border-box; margin:0; }
 body { background:var(--bg); color:var(--txt);
        font:14px/1.45 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; }
 header { padding:18px 24px; border-bottom:1px solid var(--line);
          background:linear-gradient(90deg,#0f1220,#131a30); }
 h1 { font-size:19px; letter-spacing:.4px; }
 h1 small { color:var(--warm); }
 .sub { color:var(--mut); font-size:12.5px; margin-top:6px; }
 .cards { display:flex; gap:12px; flex-wrap:wrap; padding:16px 24px 4px; }
 .card { background:var(--panel); border:1px solid var(--line); border-radius:10px;
         padding:12px 16px; min-width:180px; }
 .card b { display:block; font-size:19px; color:var(--acc); padding-bottom:4px; }
 .card span { color:var(--mut); font-size:11px; letter-spacing:.6px; }
 section { padding:8px 24px 28px; }
 h2 { font-size:15px; color:var(--warm); border-bottom:1px dashed var(--line);
      padding:16px 0 6px; margin-bottom:12px; }
 h2 small { color:var(--mut); font-weight:normal; }
 .panel { background:var(--panel); border:1px solid var(--line); border-radius:10px;
          padding:14px 16px; }
 #heatmaps { display:grid; grid-template-columns:repeat(auto-fill,minmax(280px,1fr));
             gap:16px; }
 .tile { background:var(--panel); border:1px solid var(--line); border-radius:10px;
         padding:10px; cursor:zoom-in; transition:transform .12s, box-shadow .2s; }
 .tile:hover { transform:translateY(-2px); border-color:var(--acc); }
 .tile canvas { width:100%; display:block; image-rendering:pixelated; border-radius:6px;
                border:1px solid var(--line); }
 .tilelabel { display:flex; flex-wrap:wrap; gap:4px 14px; margin-top:8px;
              font-size:11.5px; color:var(--mut); }
 .tilelabel b { color:var(--txt); font-size:12.5px; }
 .legend { display:flex; align-items:center; gap:2px; margin:10px 0 16px;
           font-size:11.5px; color:var(--mut); }
 .legend .sw { width:30px; height:12px; display:inline-block; border-radius:3px; }
 .legend .sw:last-of-type, .legend span:last-of-type { margin-right:14px; }
 .cols { display:grid; grid-template-columns:1fr 1fr; gap:16px; }
 @media (max-width:1000px){ .cols { grid-template-columns:1fr; } }
 table { border-collapse:collapse; width:100%; font-size:12px; }
 th,td { padding:4px 8px; border-bottom:1px solid var(--line); text-align:right; }
 th:first-child,td:first-child { text-align:left; }
 th { color:var(--mut); font-weight:normal; }
 .pos { color:var(--acc); } .neg { color:var(--hot); }
 .bars { display:flex; align-items:flex-end; gap:3px; height:130px; }
 .bar { flex:1; background:var(--acc); border-radius:2px 2px 0 0; min-width:3px; }
 .profile { color:var(--mut); font-size:11px; margin-top:6px; }
 #sampleCharts { display:grid; grid-template-columns:1fr 1fr; gap:12px; }
 @media (max-width:1000px){ #sampleCharts { grid-template-columns:1fr; } }
 #sampleCharts canvas { width:100%; }
 h3 { font-weight:normal; font-size:13px; margin:0 0 6px; }
 /* lightbox */
 #lensBox { position:fixed; inset:0; background:rgba(8,10,16,.78); display:none;
            align-items:center; justify-content:center; z-index:50; flex-direction:column; }
 #lensBox.open { display:flex; }
 #lensCard { background:var(--panel); border:1px solid var(--line); border-radius:12px;
             padding:18px; max-width:92vw; }
 #lensCard canvas { width:100%; image-rendering:pixelated; border-radius:8px;
                    border:1px solid var(--line); }
 #lensTitle { margin:10px 2px 0; font-size:13px; color:var(--mut); }
 #lensTitle b { color:var(--txt); }
 #lensBox .close { margin-top:10px; }
 .btn { background:var(--acc); color:#0b1210; border:none; border-radius:8px;
        padding:9px 18px; font:700 13px ui-monospace,monospace; cursor:pointer; }
 .btn:hover { filter:brightness(1.1); }
 #topbar { display:flex; align-items:center; gap:14px; flex-wrap:wrap; }
 #warn { display:none; background:#2a1418; border:1px solid var(--hot); color:#ffb7b7;
         border-radius:10px; padding:12px 16px; margin:14px 24px 0; font-size:12.5px; }
 #warn.show { display:block; }
</style></head>
<body>
<header>
  <h1>WeatherNet 1.58-bit <small>— heatmap of every latent weight</small></h1>
  <div class="sub">
    checkpoint: <b>@@CKPT@@</b> · archive: <b>@@ARCHIVE@@</b> ·
    generated: @@GENERATED@@ · inference: @@SECS@@s (CPU) ·
    <b>@@BLOCKS@@ × @@WIDTH@@</b> blocks · @@NHEAT@@ ternary matrices shown
  </div>
</header>

<div id="warn"></div>

<div class="cards">
  <div class="card" id="cParams"></div>
  <div class="card" id="cTernary"></div>
  <div class="card" id="cActive"></div>
  <div class="card" id="cInfer"></div>
  <div class="card" id="cTrain"></div>
</div>

<section>
  <h2>All weights, binned 120×120 each — green fires +1, red fires −1, dark is dead</h2>
  <div id="topbar">
    <button id="playBtn" class="btn">⏸ pause</button>
    <div class="legend" id="legend"></div>
  </div>
  <div id="heatmaps"></div>
</section>

<section class="cols">
  <div>
    <h2>Layer weight structure</h2>
    <div class="panel"><table id="layerTable"></table></div>
  </div>
  <div>
    <h2>Firing profile — mean |activation| per layer</h2>
    <div class="panel">
      <div class="bars" id="profile"></div>
      <div class="profile">averaged over @@SAMPLES@@ sample windows · each bar = one transformer layer</div>
    </div>
  </div>
</section>

<section>
  <h2>Inputs vs outputs — 24h forecasts on real station windows</h2>
  <div id="sampleCharts"></div>
</section>

<!-- lightbox -->
<div id="lensBox">
  <div id="lensCard">
    <canvas id="lens"></canvas>
    <div id="lensTitle"></div>
    <button class="btn" onclick="document.getElementById('lensBox').classList.remove('open')">close</button>
  </div>
</div>

<script>
@@JS@@
</script>
</body></html>"""


def render_html(d: dict) -> str:
    js = JS.replace("__DATA__", json.dumps(d))
    return (TEM
            .replace("@@CKPT@@", d["ckpt"])
            .replace("@@ARCHIVE@@", d["archive"])
            .replace("@@GENERATED@@", d["generated"])
            .replace("@@SECS@@", str(d["infer_secs"]))
            .replace("@@BLOCKS@@", str(d["blocks"]))
            .replace("@@WIDTH@@", str(d["width"]))
            .replace("@@NHEAT@@", str(len(d["heat"])))
            .replace("@@SAMPLES@@", str(len(d["samples"])))
            .replace("@@JS@@", js))


if __name__ == "__main__":
    main()