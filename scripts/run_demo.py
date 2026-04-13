"""
TradingApp - Interactive Demo Dashboard (HTTP server)
======================================================
Generates synthetic SPY 1-min bars, runs the full backtest engine,
then serves an interactive Plotly dashboard via FastAPI on localhost.
Opens the browser automatically.

Usage:
    python scripts/run_demo.py
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import math
import random
import sys
import time
import threading
import webbrowser
from pathlib import Path

# Force UTF-8 stdout on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# Project root on path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core.config import setup_logging
from src.core.models import Bar
from backtest.engine import BacktestEngine

setup_logging()
logger = logging.getLogger("tradingapp.demo")

# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------

def generate_synthetic_bars(
    symbol: str = "SPY",
    n_bars: int = 7_800,
    start_price: float = 480.0,
    seed: int = 42,
) -> list[Bar]:
    random.seed(seed)
    rng = random.Random(seed)
    bars: list[Bar] = []
    price = start_price
    epoch_start = 1_700_000_400
    bars_per_day = 390
    vol_base = 0.0001
    drift = 0.00002

    for i in range(n_bars):
        day = i // bars_per_day
        bar_in_day = i % bars_per_day
        ts = epoch_start + day * 86_400 + bar_in_day * 60
        vol = vol_base * (1 + 2 * abs(rng.gauss(0, 1)) * 0.3)
        ret = drift + vol * rng.gauss(0, 1)
        price = price * math.exp(ret)
        open_ = price / math.exp(ret)
        high = max(open_, price) * (1 + abs(rng.gauss(0, vol * 0.5)))
        low = min(open_, price) * (1 - abs(rng.gauss(0, vol * 0.5)))
        t = bar_in_day / bars_per_day
        v_profile = 2 - 4 * (t - 0.5) ** 2
        volume = max(1, int(v_profile * 500_000 * (1 + rng.gauss(0, 0.3))))
        bars.append(Bar(
            symbol=symbol,
            timestamp=float(ts),
            open=round(open_, 4),
            high=round(high, 4),
            low=round(low, 4),
            close=round(price, 4),
            volume=float(volume),
            vwap=round((high + low + price) / 3, 4),
        ))
    return bars


DEMO_STRATEGY_CONFIG: dict = {
    "dual_ma_crossover": {
        "enabled": True,
        "family": "trend_following",
        "symbols": ["SPY"],
        "params": {"fast_period": 20, "slow_period": 60, "conviction": 0.7},
    },
}

# ---------------------------------------------------------------------------
# Backtest runner
# ---------------------------------------------------------------------------

async def _run_backtest(bars: list[Bar]):
    engine = BacktestEngine(strategy_config=DEMO_STRATEGY_CONFIG, initial_nav=1_000_000)
    equity_df = await engine.run(bars)
    stats = engine.compute_stats(equity_df)
    return equity_df, stats, engine


# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------

def build_html(equity_df, stats: dict, bars: list[Bar], elapsed: float) -> str:
    # Equity curve
    if not equity_df.empty:
        ts_labels = [str(t) for t in equity_df.index.strftime("%Y-%m-%d %H:%M")]
        nav_values = [round(v, 2) for v in equity_df["nav"].tolist()]
        returns_values = [round(v * 100, 4) for v in equity_df["returns"].fillna(0).tolist()]
        cummax = equity_df["nav"].cummax()
        dd = ((equity_df["nav"] - cummax) / cummax * 100).tolist()
    else:
        ts_labels, nav_values, returns_values, dd = [], [], [], []

    # Price chart (sampled)
    sample_step = max(1, len(bars) // 600)
    sampled = bars[::sample_step]
    import datetime as _dt
    price_ts = [
        str(_dt.datetime.utcfromtimestamp(b.timestamp).strftime("%Y-%m-%d %H:%M"))
        for b in sampled
    ]
    price_open  = [b.open  for b in sampled]
    price_high  = [b.high  for b in sampled]
    price_low   = [b.low   for b in sampled]
    price_close = [b.close for b in sampled]

    # Stat cards
    stat_items = [
        ("Total Return",      f"{stats.get('total_return', 0)*100:.3f}%"),
        ("Ann. Return",       f"{stats.get('annualised_return', 0)*100:.2f}%"),
        ("Ann. Volatility",   f"{stats.get('annualised_volatility', 0)*100:.2f}%"),
        ("Sharpe Ratio",      f"{stats.get('sharpe_ratio', 0):.3f}"),
        ("Max Drawdown",      f"{stats.get('max_drawdown', 0)*100:.2f}%"),
        ("Total Fills",       str(int(stats.get("total_fills", 0)))),
        ("Final NAV",         f"${stats.get('final_nav', 0):,.0f}"),
        ("Bars Replayed",     f"{len(bars):,}"),
        ("Engine Runtime",    f"{elapsed:.1f}s"),
    ]

    def card_color(label, value):
        if label == "Max Drawdown":
            return "#ef4444"
        if label in ("Total Return", "Ann. Return"):
            return "#10b981" if not value.startswith("-") else "#ef4444"
        if label == "Sharpe Ratio":
            try:
                return "#10b981" if float(value) >= 0 else "#ef4444"
            except ValueError:
                return "#f9fafb"
        return "#a5b4fc"

    stat_cards_html = "".join(
        f'<div class="stat-card" id="stat-{label.lower().replace(" ", "-")}">'
        f'<div class="stat-label">{label}</div>'
        f'<div class="stat-value" style="color:{card_color(label, value)}">{value}</div>'
        f"</div>"
        for label, value in stat_items
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1.0"/>
<title>TradingApp &mdash; Backtest Dashboard</title>
<meta name="description" content="Interactive backtest results for the TradingApp multi-agent algorithmic trading system."/>
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet"/>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
:root{{
  --bg:#0a0d14;--panel:#111827;--border:#1f2937;
  --accent:#6366f1;--g:#10b981;--amber:#f59e0b;--red:#ef4444;
  --text:#f9fafb;--dim:#9ca3af;
  --font:'Inter',sans-serif;--mono:'JetBrains Mono',monospace;
}}
*{{box-sizing:border-box;margin:0;padding:0}}
html,body{{height:100%;background:var(--bg);color:var(--text);font-family:var(--font)}}
/* ── Header ── */
header{{
  background:linear-gradient(135deg,#0f172a 0%,#1e1b4b 100%);
  border-bottom:1px solid var(--border);
  padding:18px 40px;
  display:flex;align-items:center;justify-content:space-between;
  position:sticky;top:0;z-index:50;
}}
.logo{{display:flex;align-items:center;gap:14px}}
.logo-icon{{
  width:42px;height:42px;border-radius:10px;
  background:linear-gradient(135deg,var(--accent),#818cf8);
  display:flex;align-items:center;justify-content:center;font-size:20px;
}}
.logo-text h1{{font-size:18px;font-weight:700;letter-spacing:-.3px}}
.logo-text span{{font-size:12px;color:var(--dim)}}
.badges{{display:flex;gap:8px;align-items:center;flex-wrap:wrap}}
.badge{{
  border-radius:20px;padding:4px 12px;font-size:12px;font-weight:500;
  background:rgba(99,102,241,.15);border:1px solid rgba(99,102,241,.3);color:#a5b4fc;
}}
.badge.green{{background:rgba(16,185,129,.15);border-color:rgba(16,185,129,.3);color:#6ee7b7}}
/* ── Layout ── */
main{{max-width:1440px;margin:0 auto;padding:32px 40px}}
.section-label{{
  font-size:11px;font-weight:600;color:var(--dim);
  text-transform:uppercase;letter-spacing:1px;
  margin-bottom:14px;display:flex;align-items:center;gap:8px;
}}
.section-label::after{{content:'';flex:1;height:1px;background:var(--border)}}
/* ── Stat cards ── */
.stats-grid{{
  display:grid;
  grid-template-columns:repeat(auto-fill,minmax(150px,1fr));
  gap:14px;margin-bottom:30px;
}}
.stat-card{{
  position:relative;overflow:hidden;
  background:var(--panel);border:1px solid var(--border);
  border-radius:12px;padding:18px 16px;
  transition:border-color .2s,transform .2s;cursor:default;
}}
.stat-card::before{{
  content:'';position:absolute;top:0;left:0;right:0;height:2px;
  background:linear-gradient(90deg,var(--accent),#818cf8);
}}
.stat-card:hover{{border-color:var(--accent);transform:translateY(-2px)}}
.stat-label{{font-size:10px;color:var(--dim);text-transform:uppercase;letter-spacing:.8px;margin-bottom:7px;font-weight:500}}
.stat-value{{font-size:20px;font-weight:700;font-family:var(--mono);letter-spacing:-.5px}}
/* ── Charts ── */
.chart-full,.chart-half{{margin-bottom:20px}}
.chart-row{{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-bottom:20px}}
.chart-panel{{
  background:var(--panel);border:1px solid var(--border);
  border-radius:14px;padding:22px;
  transition:border-color .2s;
}}
.chart-panel:hover{{border-color:rgba(99,102,241,.35)}}
.chart-panel h2{{font-size:14px;font-weight:600;margin-bottom:3px}}
.chart-panel p{{font-size:11px;color:var(--dim);margin-bottom:14px}}
/* ── Pulse ── */
@keyframes pulse{{0%,100%{{opacity:1}}50%{{opacity:.3}}}}
.dot{{
  display:inline-block;width:7px;height:7px;border-radius:50%;
  background:var(--g);animation:pulse 2s infinite;margin-right:5px;
}}
/* ── Footer ── */
footer{{
  text-align:center;padding:20px;color:var(--dim);
  font-size:11px;border-top:1px solid var(--border);margin-top:8px;
}}
@media(max-width:900px){{
  main{{padding:18px}}header{{padding:14px 18px}}
  .chart-row{{grid-template-columns:1fr}}
}}
</style>
</head>
<body>
<header>
  <div class="logo">
    <div class="logo-icon">&#x1F4C8;</div>
    <div class="logo-text">
      <h1>TradingApp</h1>
      <span>Multi-Agent Algorithmic Trading System</span>
    </div>
  </div>
  <div class="badges">
    <span class="badge">Backtest Report</span>
    <span class="badge green"><span class="dot"></span>Simulation Complete</span>
    <span class="badge">SPY &bull; 1-Min Bars</span>
    <span class="badge">Dual MA Crossover</span>
  </div>
</header>

<main>
  <div class="section-label">&#x1F4CA; Performance Summary</div>
  <div class="stats-grid">{stat_cards_html}</div>

  <div class="section-label">&#x1F4B9; SPY Price Action (Sampled)</div>
  <div class="chart-panel chart-full">
    <h2>Candlestick Chart</h2>
    <p>Every {sample_step}th bar &bull; {len(bars):,} 1-min bars replayed through the agent pipeline</p>
    <div id="price-chart"></div>
  </div>

  <div class="section-label">&#x1F4C9; Equity &amp; Drawdown</div>
  <div class="chart-row">
    <div class="chart-panel">
      <h2>Equity Curve</h2>
      <p>Portfolio NAV over the backtest period</p>
      <div id="equity-chart"></div>
    </div>
    <div class="chart-panel">
      <h2>Drawdown</h2>
      <p>% decline from rolling peak NAV</p>
      <div id="dd-chart"></div>
    </div>
  </div>

  <div class="section-label">&#x1F4D0; Returns Distribution</div>
  <div class="chart-panel chart-full">
    <h2>Bar-level Returns Histogram</h2>
    <p>Distribution of per-bar NAV P&amp;L changes</p>
    <div id="hist-chart"></div>
  </div>
</main>

<footer>
  TradingApp Demo Dashboard &bull; Served via FastAPI/Uvicorn &bull;
  Generated {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())} &bull;
  Synthetic data &mdash; not financial advice.
</footer>

<script>
const LAYOUT_BASE = {{
  paper_bgcolor:'rgba(0,0,0,0)',
  plot_bgcolor:'rgba(0,0,0,0)',
  font:{{color:'#9ca3af',family:'Inter,sans-serif',size:12}},
  xaxis:{{gridcolor:'#1f2937',zerolinecolor:'#374151',showspikes:true,spikecolor:'#6366f1',spikethickness:1}},
  yaxis:{{gridcolor:'#1f2937',zerolinecolor:'#374151'}},
  margin:{{l:64,r:20,t:10,b:50}},
  autosize:true,
  legend:{{bgcolor:'rgba(0,0,0,0)',font:{{color:'#9ca3af'}}}},
}};
const CFG = {{responsive:true,displayModeBar:true,displaylogo:false,
  modeBarButtonsToRemove:['toImage','sendDataToCloud']}};

/* ── Price ── */
Plotly.newPlot('price-chart',[{{
  type:'candlestick',
  x:{json.dumps(price_ts)},
  open:{json.dumps(price_open)},
  high:{json.dumps(price_high)},
  low:{json.dumps(price_low)},
  close:{json.dumps(price_close)},
  increasing:{{line:{{color:'#10b981'}},fillcolor:'rgba(16,185,129,.2)'}},
  decreasing:{{line:{{color:'#ef4444'}},fillcolor:'rgba(239,68,68,.2)'}},
  name:'SPY',
}}],{{
  ...LAYOUT_BASE,
  height:380,
  xaxis:{{...LAYOUT_BASE.xaxis,type:'category',rangeslider:{{visible:false}},nticks:14}},
  yaxis:{{...LAYOUT_BASE.yaxis,title:'Price ($)',tickprefix:'$'}},
}},CFG);

/* ── Equity ── */
const eqTs  = {json.dumps(ts_labels)};
const eqNav = {json.dumps(nav_values)};
Plotly.newPlot('equity-chart',[{{
  type:'scatter',mode:'lines',x:eqTs,y:eqNav,
  line:{{color:'#6366f1',width:2}},
  fill:'tozeroy',fillcolor:'rgba(99,102,241,.08)',
  name:'NAV',hovertemplate:'%{{x}}<br>NAV: $%{{y:,.0f}}<extra></extra>',
}}],{{
  ...LAYOUT_BASE,height:280,
  xaxis:{{...LAYOUT_BASE.xaxis,nticks:7}},
  yaxis:{{...LAYOUT_BASE.yaxis,title:'NAV ($)',tickformat:'$,.0f'}},
}},CFG);

/* ── Drawdown ── */
const ddVals = {json.dumps([round(v,4) for v in dd])};
Plotly.newPlot('dd-chart',[{{
  type:'scatter',mode:'lines',x:eqTs,y:ddVals,
  line:{{color:'#ef4444',width:1.5}},
  fill:'tozeroy',fillcolor:'rgba(239,68,68,.08)',
  name:'Drawdown %',hovertemplate:'%{{x}}<br>DD: %{{y:.3f}}%<extra></extra>',
}}],{{
  ...LAYOUT_BASE,height:280,
  xaxis:{{...LAYOUT_BASE.xaxis,nticks:7}},
  yaxis:{{...LAYOUT_BASE.yaxis,title:'Drawdown (%)',ticksuffix:'%'}},
}},CFG);

/* ── Histogram ── */
const retVals = {json.dumps(returns_values)};
Plotly.newPlot('hist-chart',[{{
  type:'histogram',x:retVals,nbinsx:80,
  marker:{{
    color:retVals.map(v=>v>=0?'rgba(16,185,129,.75)':'rgba(239,68,68,.75)'),
    line:{{width:0}},
  }},
  name:'Returns',
  hovertemplate:'Return: %{{x:.4f}}%<br>Count: %{{y}}<extra></extra>',
}}],{{
  ...LAYOUT_BASE,height:240,bargap:.04,
  xaxis:{{...LAYOUT_BASE.xaxis,title:'Return (%)',ticksuffix:'%'}},
  yaxis:{{...LAYOUT_BASE.yaxis,title:'Frequency'}},
}},CFG);
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

def build_fastapi_app(html_content: str, stats: dict, bars: list[Bar], equity_df):
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse, JSONResponse
    from fastapi.middleware.cors import CORSMiddleware

    app = FastAPI(title="TradingApp Dashboard")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

    @app.get("/", response_class=HTMLResponse)
    async def dashboard():
        return HTMLResponse(content=html_content)

    @app.get("/api/stats")
    async def api_stats():
        return JSONResponse(content={k: (v if not isinstance(v, float) or (v == v) else None)
                                      for k, v in stats.items()})

    @app.get("/api/bars/count")
    async def api_bars_count():
        return {"count": len(bars)}

    @app.get("/health")
    async def health():
        return {"status": "ok", "bars": len(bars)}

    return app


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    PORT = 8765

    print("\n" + "=" * 62)
    print("  TradingApp - Interactive Demo Dashboard")
    print("=" * 62)

    # 1. Generate bars
    print("\n[1/4] Generating synthetic SPY 1-min bars...")
    t0 = time.time()
    bars = generate_synthetic_bars(n_bars=7_800)
    print(f"      OK  {len(bars):,} bars generated")

    # 2. Run backtest
    print("[2/4] Running backtest engine...")
    equity_df, stats, _ = asyncio.run(_run_backtest(bars))
    elapsed = time.time() - t0
    print(f"      OK  Backtest complete in {elapsed:.1f}s")

    # Print summary
    print("\n" + "-" * 42)
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"  {k:30s}: {v:>12.4f}")
        else:
            print(f"  {k:30s}: {v:>12}")
    print("-" * 42)

    # 3. Build HTML
    print("\n[3/4] Building interactive dashboard...")
    html_content = build_html(equity_df, stats, bars, elapsed)
    print("      OK  HTML rendered")

    # 4. Start server + open browser
    print(f"[4/4] Starting HTTP server on http://localhost:{PORT} ...")

    import uvicorn
    app = build_fastapi_app(html_content, stats, bars, equity_df)

    # Open browser after a short delay to let server start
    def _open():
        time.sleep(1.2)
        webbrowser.open(f"http://localhost:{PORT}")
        print(f"\n  --> Dashboard: http://localhost:{PORT}")
        print("      Press Ctrl+C to stop the server.\n")

    threading.Thread(target=_open, daemon=True).start()

    print("\n" + "=" * 62 + "\n")
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")


if __name__ == "__main__":
    main()
