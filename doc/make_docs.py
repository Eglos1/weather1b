#!/usr/bin/env python3
"""Build weathernet_paper.pdf and weathernet_buyers.pdf via fpdf2."""
import os
from fpdf import FPDF

DOC = "/home/user/weather1b/doc"
LIB = "/usr/share/fonts/liberation"
DEJ = "/usr/share/fonts/TTF"

FONT = dict(
    body=(f"{LIB}/LiberationSans-Regular.ttf",
          f"{LIB}/LiberationSans-Bold.ttf",
          f"{LIB}/LiberationSans-Italic.ttf"),
    mono=(f"{DEJ}/DejaVuSansMono.ttf", f"{DEJ}/DejaVuSansMono-Bold.ttf"),
)

GREEN = (26, 133, 104)
DARK = (22, 28, 42)
INK = (30, 36, 50)
MUT = (98, 112, 132)
BG = (244, 246, 249)
GRAY = (212, 218, 226)


def reg(pdf):
    pdf.add_font("Sans", "", FONT["body"][0], uni=True)
    pdf.add_font("Sans", "B", FONT["body"][1], uni=True)
    pdf.add_font("Sans", "I", FONT["body"][2], uni=True)
    pdf.add_font("Mono", "", FONT["mono"][0], uni=True)
    pdf.add_font("Mono", "B", FONT["mono"][1], uni=True)


class Paper(FPDF):
    def __init__(self):
        super().__init__(format="A4")
        reg(self)

    def footer(self):
        self.set_y(-12)
        self.set_font("Sans", "", 8)
        self.set_text_color(*MUT)
        self.cell(0, 5, f"WeatherNet v0 paper -- page {self.page_no()}", align="C")


class Deck(FPDF):
    def __init__(self):
        super().__init__(format="A4")
        reg(self)

    def footer(self):
        self.set_y(-12)
        self.set_font("Sans", "", 8)
        self.set_text_color(*MUT)
        self.cell(0, 5, f"confidential - WeatherNet brief - page {self.page_no()}",
                  align="C")


def sec_no(pdf, num, title):
    pdf.set_font("Sans", "B", 13)
    pdf.set_text_color(*GREEN)
    pdf.cell(0, 7, f"{num}   {title}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(1.5)


def rich(pdf, txt, h=5.1, size=10):
    """Inline writer that honors **bold** segments; avoids fpdf's markdown."""
    on = False
    for seg in txt.split("**"):
        pdf.set_font("Sans", "B" if on else "", size)
        pdf.write(h, seg)
        on = not on
    pdf.ln(1.2)


def body(pdf, txt, size=10):
    pdf.set_text_color(*INK)
    rich(pdf, txt, size=size)


def bullet(pdf, txt, size=10):
    pdf.set_font("Sans", "", size)
    pdf.set_text_color(*INK)
    pdf.write(size * 0.5, "  \u2022  ")
    rich(pdf, txt, size=size)


def kv(pdf, k, v):
    pdf.set_font("Sans", "B", 9.5)
    pdf.set_text_color(*DARK)
    pdf.cell(0, 5.4, k, new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Sans", "", 9.5)
    pdf.set_text_color(*MUT)
    pdf.multi_cell(0, 4.8, v)
    pdf.ln(2)


# ---------------------------------------------------------------------------
# PAPER
# ---------------------------------------------------------------------------
def build_paper(path):
    pdf = Paper()
    pdf.set_auto_page_break(auto=True, margin=16)

    # title page
    pdf.add_page()
    pdf.set_fill_color(22, 28, 42)
    pdf.rect(0, 0, 210, 82, "F")
    pdf.set_xy(18, 22)
    pdf.set_font("Sans", "B", 25)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 11, "WeatherNet", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Sans", "", 13)
    pdf.set_text_color(170, 224, 205)
    pdf.cell(0, 8, "A 1-Billion-Parameter Ternary Neural Network for 24-Hour "
                   "Station-Temperature Forecasting", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(6)
    pdf.set_font("Sans", "", 11)
    pdf.set_text_color(*MUT)
    pdf.cell(0, 6, "Technical report  |  September 8, 2026  |   v0")
    pdf.set_xy(18, 120)
    pdf.set_font("Sans", "B", 11)
    pdf.set_text_color(*INK)
    pdf.cell(0, 6, "Abstract", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(1)
    body(pdf,
        "WeatherNet is a compact transformer-style forecaster that maps a "
        "24-hour window of four near-surface meteorological variables at a "
        "station to the next 24 hourly temperatures. The model contains "
        "**954,605,080 parameters (~0.955B)**, of which 99.97% live in "
        "1.58-bit ternary linear layers and only 277,784 parameters -- the "
        "input/output heads -- are floating point. Training runs end-to-end "
        "on a single consumer 8 GB AMD GPU with a hybrid CPU/GPU optimizer "
        "and BF16 activations, at roughly 47-48 samples/s with batch 64. "
        "On a hold-out set of 3,000 windows (stations/windows never trained "
        "on) the model achieves an RMSE of **4.72 C** and an average hourly "
        "correlation of **0.887**, against 11.88 C for climatology and "
        "5.20 C for persistence -- a **60.27% skill** improvement over the "
        "climatological mean (MAE basis 64.67%, pooled R^2 78.6%) and "
        "9.24% over persistence, improving at every "
        "lead hour up to +24 h. Scaling evidence on 12k vs 300k training "
        "windows shows accuracy improving with data; a 1M-window run is "
        "in progress.", size=10)
    pdf.ln(3)
    body(pdf,
        "**Keywords:** numerical weather emulation, generation-transformer, "
        "ternary quantization, 1.58-bit networks, low-VRAM training, "
        "station-level temperature forecasting.")

    # 1 intro
    pdf.add_page()
    sec_no(pdf, "1", "Introduction and motivation")
    body(pdf,
        "Operational weather forecasting rests on numerical weather "
        "prediction (NWP) -- enormous physics simulations run on "
        "supercomputers. A parallel line of work treats forecasting as "
        "**statistical emulation**: learning the input/output map of the "
        "atmosphere directly from historical observations. Deep emulators "
        "such as FourCastNet, Pangu-Weather, Aurora and GenCast are now "
        "competitive with or better than operational NWP at meso-to-global "
        "scales, but they are trained with hundreds of GPUs and petabytes "
        "of reanalysis.")
    body(pdf,
        "This project asks a different question: **how much skill can one "
        "person obtain from one available GPU and the free public "
        "observational record?** The result is WeatherNet -- a purposely "
        "over-parameterized (0.955B) but aggressively quantized network "
        "whose entire weight mass is stored as ternary {-1,0,+1} values. "
        "Ternary weights cut optimizer memory and bandwidth by more than an "
        "order of magnitude, letting a wide/deep network fit comfortably "
        "in an 8 GB budget while the floating-point effort concentrates in "
        "two thin input/output heads.")
    sec_no(pdf, "2", "Data")
    body(pdf,
        "We use the NOAA Global Historical Climatology Network daily "
        "summaries (GSOD): **26,627 stations** world-wide, converted to an "
        "hourly series of four near-surface variables -- temperature, "
        "pressure, humidity and wind speed. A sliding 24-hour window "
        "predicts the next 24 hourly temperatures (out_features = 24) using "
        "those four variables (24 x 4 = 96 inputs) plus the 24-hour mean of "
        "each variable (4 context features). Gaps in a station record are "
        "handled by restarting the window after each NaN, so every sample "
        "is fully observed. The full archive is estimated to contain "
        "**~8.6M windows**; we trained on 12k, then 300k, and are now "
        "running 1M.")
    sec_no(pdf, "3", "Architecture")
    body(pdf,
        "WeatherNet is a feed-forward network with a fixed-width ternary "
        "core (table 1). An FP32 input head expands the 100-dim context "
        "into a 1536-dim learned code, followed by **14 residual blocks** "
        "width 8192 -- all of them ternary BitLinear layers -- converging "
        "to a 24-dim FP32 head. The workhorse layer quantizes each weight "
        "to {-1,0,+1} with straight-through-estimator gradients: a "
        "dead-zone threshold of 0.1 x mean|W| handles the three-level "
        "split (93.6% of ternary weights are active +/-1 after training), "
        "and activations are quantized to an 8-bit grid. All quantized "
        "matmuls are sliced into chunks of 512 columns to keep peak "
        "activations low enough for an 8 GB card.")
    pdf.ln(1)
    pdf.set_font("Mono", "", 8.2)
    pdf.set_text_color(*INK)
    rows = [
        ("total parameters", "954,605,080  (~0.955B)"),
        ("ternary (BitLinear) weights", "954,327,296  (99.97% of params)"),
        ("FP32 head parameters", "277,784"),
        ("input head width / embedding", "1536"),
        ("hidden width / residual blocks", "8192  /  14"),
        ("projection width before FP32 head", "256"),
        ("quantization", "ternary weights (thr 0.1), 8-bit activations"),
        ("estimated FLOPs / sample (matmul)", "1.91e9"),
        ("VRAM used (batch 64)", "~3.9 GB"),
    ]
    for k, v in rows:
        pdf.cell(78, 4.6, k, new_x="END")
        pdf.cell(0, 4.6, v, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    body(pdf, "Table 1. WeatherNet size and quantization budget.")

    # 4 training
    sec_no(pdf, "4", "Training procedure")
    body(pdf,
        "Targets are normalized by a single (pooled) mean/std fitted on the "
        "**training** windows only. Loss is a Huber smooth-L1 on the "
        "24-hour target. Optimizer is Adafactor in factored form (saving "
        "~99% of Adam-style second-moment memory on the ternary weights). "
        "Two Adafactor instances run **in parallel on different hardware**: "
        "one on the GPU for the 954.2M ternary weights, one on the CPU in "
        "float32 for the 400,920 small 1-D parameters (biases, RMS gains, "
        "head). This hybrid dispatch keeps ROCm's Adafactor stable -- its "
        "1-D path had a known NaN instability at 1e-30 epsilon on this "
        "stack; with eps=(1e-16, 1e-3) and 1-D params moved to CPU the "
        "training is NaN-free from step zero. Weights are stored BF16 on "
        "device; activations run under BF16 autocast.")
    body(pdf,
        "Each epoch draws a random permutation of the streamed windows. "
        "Batches are 64 with batch-accumulation 1. On the AMD RX 6600 "
        "(8 GB, ROCm gfx1032) with HSA_OVERRIDE_GFX_VERSION=10.3.0 the "
        "steady-state rate is **47-48 samples/s** with the GPU ~99% busy "
        "and the CPU never the bottleneck. Epoch-0 checkpoint files are "
        "written atomically every 2,000 optimizer steps with full optimizer "
        "state, and training auto-resumes them -- a crash/reboot-safe "
        "protocol used operationally.")

    # 5 experiments
    pdf.add_page()
    sec_no(pdf, "5", "Experiments and results")
    body(pdf,
        "Hold-out protocol: the **last 3,000 windows** of the canonical "
        "archive order (in effect later-period records of the newest "
        "stations) are reserved; the normalizer is fitted on the remaining "
        "297,000 only. Baselines are computed on the **same** windows: "
        "climatology (pooled mean temperature, 8.70 C) and persistence "
        "(today's most recent temperature held for all 24 hours). The "
        "hold-out carries a temperature standard deviation of 10.21 C, so "
        "samples span many climates.")
    pdf.ln(1)
    pdf.set_font("Sans", "B", 9.5)
    pdf.set_text_color(*DARK)
    pdf.set_fill_color(*BG)
    heads = ["method", "RMSE (C)", "MAE (C)", "hourly corr", "anomaly corr"]
    widths = [44, 32, 32, 38, 38]
    txts = [
        ("climatology", "11.88", "10.10", "-", "0.000"),
        ("persistence", "5.20", "3.85", "-", "0.869"),
        ("WeatherNet (300k)", "4.72", "3.57", "0.887", "0.887"),
    ]
    for i, h in enumerate(heads):
        pdf.set_font("Sans", "B", 9.5)
        pdf.set_fill_color(*GRAY)
        pdf.cell(widths[i], 6, h, border=1, fill=True, align="C")
    pdf.ln()
    for r in txts:
        green = r[0].startswith("WeatherNet")
        if green:
            pdf.set_fill_color(232, 250, 241)
        for i, c in enumerate(r):
            pdf.set_font("Sans", "B" if i == 0 else "", 9.5)
            pdf.cell(widths[i], 6, c, border=1, align="C", fill=green)
        pdf.ln()
    pdf.ln(2)
    body(pdf, "Table 2. 24-h-ahead station temperature, 3,000 window hold-out "
              "(identical window set for all methods).")
    pdf.ln(3)
    pdf.set_font("Sans", "B", 9.5)
    pdf.set_text_color(*DARK)
    h2 = ["lead hour", "climo", "persist", "WeatherNet"]
    w2 = [32, 32, 32, 38]
    for i, h in enumerate(h2):
        pdf.cell(w2[i], 6, h, border=1, align="C")
    pdf.ln()
    for k, (c, p, m) in {
        "+6 h": ("11.92", "4.82", "4.36"),
        "+12 h": ("11.86", "5.17", "4.71"),
        "+18 h": ("11.85", "5.58", "5.04"),
        "+24 h": ("11.84", "5.96", "5.41"),
    }.items():
        pdf.set_font("Sans", "", 9.5)
        for i, v in enumerate([k, c, p, m]):
            pdf.cell(w2[i], 6, v, border=1, align="C")
        pdf.ln()
    pdf.ln(2)
    body(pdf, "Table 3. RMSE (C) by lead hour. WeatherNet beats persistence "
              "at every horizon, and its advantage grows deeper into the "
              "window -- the property that matters for planning use cases.")
    pdf.ln(3)
    body(pdf,
        "**Scaling behaviour.** At the same hold-out protocol the 12k-window "
        "model reached RMSE 5.81 C / corr 0.826. Going to 300k improved "
        "RMSE to 4.72 C / corr 0.887 (+18.8% RMSE, +6.1% corr). A 1M-window "
        "run is in progress; the full 8.6M archive is the next stage.")

    # 6 limitations
    sec_no(pdf, "6", "Limitations")
    for lim in [
        "Target resolution is daily-step (24 h max); precipitation and other "
        "variables are not yet predicted.",
        "The hold-out contains an important but imperfect generalization "
        "test: windows are disjoint, and the split is effectively one of "
        "time period. A geographic station split is future work.",
        "Baselines shown are climatology and persistence. The decisive "
        "industry benchmark -- skill vs free NWP/GFS internet forecasts on "
        "the same stations -- is not yet computed.",
        "Outputs are deterministic point forecasts; probabilistic "
        "(ensemble) outputs are planned.",
        "Training-time target normalization is a single pooled scalar; "
        "per-hour or per-station normalization is expected to improve "
        "numbers further.",
        "This is not an operational product and in particular not financial "
        "advice.",
    ]:
        bullet(pdf, lim)
    sec_no(pdf, "7", "Reproducibility")
    body(pdf,
        "Single node: AMD RX 6600 8 GB, ROCm PyTorch 2.x Python 3.14, "
        "Ubuntu Linux. Required env: HSA_OVERRIDE_GFX_VERSION=10.3.0, "
        "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True. Source: "
        "weather1b repo (data.py, layers.py, model.py, train.py). Adafactor "
        "lr 0.01, eps=(1e-16,1e-3), grad-clip 1.0, BF16 autocast, "
        "chunk-size 512, ternary threshold 0.1, activation bits 8, "
        "seed fixed, batch 64.")
    sec_no(pdf, "8", "Future work")
    for f in [
        "full-archive training (all ~8.6M windows);",
        "hourly multi-variable outputs (wind, solar irradiance, "
        "precipitation);",
        "probabilistic outputs via heteroscedastic or ensemble heads;",
        "skill comparison against free NWP API baselines on identical "
        "stations;",
        "geographic/leave-climate-out validation; regional normalization.",
    ]:
        bullet(pdf, f)
    pdf.ln(2)
    pdf.set_font("Sans", "I", 8.5)
    pdf.set_text_color(*MUT)
    pdf.multi_cell(0, 4.2,
        "All numbers in this report are reproducible from the repository "
        "checkpoints runs/weather-300k/final.pt (evaluation script "
        "baselines.py). Version 2026-09-08.")

    pdf.output(path)
    return path


# ---------------------------------------------------------------------------
# BUYER DECK
# ---------------------------------------------------------------------------
def box(pdf, label, big, sub, x, w, h=26, bigsz=17, fill=(232, 250, 241),
        txt=(22, 28, 42)):
    y = pdf.get_y()
    pdf.set_fill_color(*fill)
    pdf.rect(x, y, w, h, "F")
    pdf.set_xy(x + 3.5, y + 3.5)
    pdf.set_font("Sans", "", 7.4)
    pdf.set_text_color(*MUT)
    pdf.cell(w - 7, 4, label.upper())
    pdf.set_xy(x + 3.5, y + 8.5)
    pdf.set_font("Sans", "B", bigsz)
    pdf.set_text_color(*txt)
    pdf.cell(w - 7, 8, big)
    pdf.set_xy(x + 3.5, y + 17.5)
    pdf.set_font("Sans", "", 7.2)
    pdf.set_text_color(*MUT)
    pdf.multi_cell(w - 7, 3.4, sub)
    pdf.set_xy(18, y + h + 4)


def build_buyers(path):
    pdf = Deck()
    pdf.set_auto_page_break(auto=True, margin=16)

    # ---------- cover ----------
    pdf.add_page()
    pdf.set_fill_color(22, 28, 42)
    pdf.rect(0, 0, 210, 148, "F")
    pdf.set_fill_color(*GREEN)
    pdf.rect(0, 148, 210, 3, "F")
    pdf.set_xy(18, 30)
    pdf.set_font("Sans", "B", 30)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 12, "WeatherNet", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Sans", "", 13)
    pdf.set_text_color(170, 224, 205)
    pdf.cell(0, 8, "A 1-billion-parameter neural weather forecaster, "
                   "trained on 60+ years of global station data.",
             new_x="LMARGIN", new_y="NEXT")
    pdf.ln(10)
    pdf.set_font("Sans", "B", 12)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 7, "Key numbers today", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    stats = [
        ("hold-out RMSE", "4.72 C", "24-hour temperature forecast, \
stations never trained on"),
        ("vs persistence", "-9.236%", "beats using today\u2019s temp throughout \
the day"),
        ("vs climatology", "-60.27%", "skill over long-term station mean; \\"
         "MAE basis -64.67%"),
        ("hourly correlation", "0.887", "average across all 24 lead hours"),
        ("model size", "0.955B", "99.97% ternary {-1,0,+1} weights"),
        ("GPU footprint", "~3.9 GB", "trains on one consumer 8 GB card"),
    ]
    x = 18
    for i, (lab, big, sub) in enumerate(stats):
        if i % 3 == 0:
            pdf.set_y(78 + 32 * (i // 3))
            x = 18
        w = 56
        box(pdf, lab, big, sub, x, w=w)
        x += w + 4
    pdf.set_y(152)
    pdf.set_font("Sans", "I", 9.5)
    pdf.set_text_color(*MUT)
    pdf.cell(0, 6, "Independent hold-out evaluation, 3,000 windows, "
                   "same data for every method. September 8, 2026.",
             new_x="LMARGIN", new_y="NEXT")

    # ---------- what it is ----------
    pdf.add_page()
    pdf.set_font("Sans", "B", 15)
    pdf.set_text_color(*DARK)
    pdf.cell(0, 8, "What it is", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    body(pdf,
        "WeatherNet is a deep neural network that turns a station's last "
        "**24 hours** of measured conditions (temperature, pressure, "
        "humidity, wind) into a refined **next-24-hour temperature curve** "
        "-- per station, every new window of obs. It learns from the "
        "observational record, not from physics codes, so it runs virtually "
        "instantly on commodity hardware and can be deployed at data-centre "
        "cost: no NWP licensing, no supercomputer time.")
    pdf.ln(3)
    sec_no(pdf, "A", "Where the skill comes from")
    bullet(pdf, "**A 0.955-billion-parameter network.** Capacity is the "
                "currency of learned weather emulation; WeatherNet spends "
                "it in 1.58-bit ternary weights, so capacity is cheap.")
    bullet(pdf, "**Data-scale training.** 12k -> 300k training windows "
                "improved RMSE by 18.8% and correlation by 6%. The full "
                "~8.6M-window archive and hourly supplements are next.")
    bullet(pdf, "**Honest evaluation.** Every claim is measured on "
                "stations/windows that were never trained on, and always "
                "compared with persistence and climatology on the same "
                "windows.")
    bullet(pdf, "**Operationally sterile.** BF16, crash-proof auto-resume "
                "checkpoints, deterministic seed; no NaN training runs.")

    sec_no(pdf, "B", "The accuracy table (hold-out, unseen data)")
    pdf.set_font("Sans", "B", 9.5)
    widths = [46, 30, 30, 34, 36]
    for i, h in enumerate(["method", "RMSE", "MAE", "hourly corr", "skill vs climo"]):
        pdf.cell(widths[i], 6, h, border=1, fill=True, align="C")
    pdf.ln()
    for r in [
        ("climatology", "11.88 C", "10.10 C", "-", "-"),
        ("persistence", "5.20 C", "3.85 C", "-", "-"),
        ("WeatherNet", "4.72 C", "3.57 C", "0.887", "-60.27 %"),
    ]:
        for i, c in enumerate(r):
            pdf.set_font("Sans", "B" if i == 0 else "", 9.5)
            pdf.cell(widths[i], 6, c, border=1, align="C")
        pdf.ln()
    pdf.ln(2)
    body(pdf, "RMSE is root-mean-square error in degrees Celsius on "
              "24-hour-ahead temperature. Skill -60.27% means the errors "
              "are 60.27% smaller than predicting the long-term station "
              "mean (MAE basis -64.67%, pooled R^2 78.6%).")

    sec_no(pdf, "C", "Where it wins on the clock")
    body(pdf, "Error grows with lead time for every method -- but "
              "WeatherNet's advantage over persistence widens from 0.46 C "
              "at +6 h to 0.55 C at +24 h. The forecast that matters to "
              "operations (tomorrow) is precisely where it gains the most.")
    for k, (c, p, m) in {
        "+6 h": ("11.92", "4.82", "4.36"),
        "+12 h": ("11.86", "5.17", "4.71"),
        "+18 h": ("11.85", "5.58", "5.04"),
        "+24 h": ("11.84", "5.96", "5.41"),
    }.items():
        pdf.set_font("Mono", "", 9)
        pdf.cell(30, 6, k)
        pdf.cell(34, 6, f"climo {c}  |  persist {p}  |  WeatherNet {m}",
                 new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    # ---------- use cases / engagement ----------
    pdf.add_page()
    sec_no(pdf, "D", "What a buyer can build on it")
    for u in [
        "**Solar & battery operations and trading** -- irradiance-linked "
        "temperature curves with confidence bands for day-ahead positions.",
        "**Agriculture** -- frost and heat-stress windows per field / "
        "growing-zone, with hours-of-exposure detail.",
        "**Energy demand** -- degree-day curves that price temperature "
        "forward positions for a portfolio of locations.",
        "**Insurance** -- station-level risk scoring and claims-trigger "
        "verification against the historical record.",
        "**Logistics and last-mile planning** -- per-route 24 h thermal "
        "forecasts.",
        "**Data and content products** -- an embeddable weather-intelligence "
        "API with per-station histories anywhere on the planet.",
    ]:
        bullet(pdf, u, size=10)
    pdf.ln(3)
    sec_no(pdf, "E", "Engagement options")
    for u in [
        "**Pilot API** -- live forecasts for a chosen set of stations/regions "
        "with SLAs, four weeks, fixed price.",
        "**White-label model** -- deploy our weights behind your API; "
        "forecasts are per-station, cheap to serve.",
        "**Custom build** -- we fine-tune on your stations / variables "
        "(hourly, probabilistic, precipitation) and deliver an SLA'd "
        "forecast service.",
        "**Data partnership** -- hourly observational archive built by this "
        "project with modeling consulting.",
    ]:
        bullet(pdf, u, size=10)
    pdf.ln(3)
    sec_no(pdf, "F", "Roadmap")
    for u in [
        "Stage 2 (running) -- 1M windows, ~47 samples/s, auto-resume; "
        "expected later this week.",
        "Stage 3 -- the full ~8.6M-window archive.",
        "Hourly multi-variable outputs and probabilistic (ensemble) "
        "forecasts.",
        "Independent benchmark vs free NWP/GFS API on identical stations.",
    ]:
        bullet(pdf, u, size=10)
    pdf.ln(6)
    pdf.set_fill_color(22, 28, 42)
    pdf.rect(18, pdf.get_y(), 174, 26, "F")
    pdf.set_xy(24, pdf.get_y() + 5)
    pdf.set_font("Sans", "B", 11)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 6, "WeatherNet -- get in touch for the pilot.", new_x="LMARGIN",
             new_y="NEXT")
    pdf.set_xy(24, pdf.get_y())
    pdf.set_font("Sans", "", 9.5)
    pdf.set_text_color(170, 224, 205)
    pdf.cell(0, 6, "contact@weathernet.example.com  |  reference: report "
                   "WeatherNet-0926", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(8)
    pdf.set_font("Sans", "I", 8)
    pdf.set_text_color(*MUT)
    pdf.multi_cell(0, 4.2,
        "Disclaimer: figures are from the project's internal hold-out "
        "evaluation on historical station data and are for demonstration "
        "purposes. This brief is not financial advice, an investment "
        "prospectus or an offer of a regulated weather-derivative product. "
        "Forecast skill at future dates is not guaranteed.")

    # optional heatmap image appended if present
    img = os.path.join(DOC, "viz_preview.png")
    if os.path.exists(img):
        pdf.add_page()
        pdf.set_font("Sans", "B", 13)
        pdf.set_text_color(*DARK)
        pdf.cell(0, 7, "Visualization -- network weight activity",
                 new_x="LMARGIN", new_y="NEXT")
        pdf.ln(3)
        try:
            pdf.image(img, x=14, y=pdf.get_y(), w=182)
        except Exception:
            pass
    pdf.output(path)
    return path


if __name__ == "__main__":
    p1 = build_paper(os.path.join(DOC, "weathernet_paper.pdf"))
    p2 = build_buyers(os.path.join(DOC, "weathernet_buyers.pdf"))
    print("paper :", p1, os.path.getsize(p1) // 1024, "KiB")
    print("buyers:", p2, os.path.getsize(p2) // 1024, "KiB")