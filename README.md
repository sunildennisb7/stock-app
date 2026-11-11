# Nifty 50 Analyzer

Analyses all Nifty 50 stocks and builds a dashboard (works on phone and desktop) with:

- **Market**: market mood, Nifty / Bank Nifty / VIX, breadth, sectors, gainers and losers
- **Picks**: a clear label for every stock (STRONG BUY / BUY / HOLD-WAIT / AVOID / SELL-EXIT) for **swing** and **long-term**, with the reasons, plus a swing trade plan (entry, stop-loss, target, quantity)
- **Track record**: how often each label worked over the last 5 years, plus backtests compared with buy-and-hold
- **Paper trade**: practise with ₹10 lakh of fake money
- **Portfolio**: your real holdings, P&L, Indian charges and an STCG/LTCG estimate

## Open the app

**https://sunildennisb7.github.io/stock-app/**

To install it: on iPhone, tap Share → Add to Home Screen. On Android, use Chrome ⋮ → Install app. On a Mac, use Safari → File → Add to Dock.

## Updates (automatic)

A GitHub Action (`.github/workflows/update.yml`) reruns the analysis at about 5:15 PM and 9:30 PM IST every weekday and republishes the site.
For a manual refresh, go to GitHub → Actions → **Daily update** → **Run workflow**.

## Run it locally (optional)

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python run.py
```

- `--refresh` ignores the cache and re-downloads everything
- `--no-open` skips opening the browser

## Files

| File | What it does |
|---|---|
| `core/data.py` | Nifty 50 list from NSE, prices and fundamentals from Yahoo Finance (cached in `cache/`) |
| `core/indicators.py` | Moving averages, RSI, MACD, ATR, etc. |
| `core/signals.py` | The scoring rules and labels. Change thresholds here |
| `core/backtest.py` | Backtests of the same rules |
| `core/report.py` | Puts everything together into `output/index.html` |
| `web/template.html` | The dashboard (HTML/CSS/JS) |
| `web/static/` | App icon, manifest and offline support (makes it installable) |
| `scripts/make_icons.py` | Regenerates the icons |

## Important

The labels come from fixed rules, not from a prediction. In 5 years of testing the edge was small, and trading the
signals did **not** beat simply holding all 50 stocks (see the Track record tab). This isn't personal investment
advice. Data is end-of-day from free sources, so check prices at your broker before trading.
