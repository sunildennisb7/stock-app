# Market Analyzer India

An installable app (phone and desktop) that analyses the Indian market every evening:

- **Market**: market mood, Nifty / Bank / Midcap / Smallcap / VIX, Nifty 500 breadth, sectors, gainers and losers
- **Picks**: every **Nifty 500 stock** (Nifty 50, Next 50, Midcap, Smallcap) and **~34 ETFs** (index, factor, sector, gold, silver, international, debt) get a clear label (STRONG BUY / BUY / HOLD-WAIT / AVOID / SELL-EXIT) for **long-term** and **swing**, with the reasons, fundamentals and a swing trade plan
- **Funds**: ~1,200 direct-plan mutual funds ranked within 23 categories (returns, risk, worst fall, Sharpe), plus an SIP calculator
- **Paper**: practise with ₹10 lakh of fake money
- **Portfolio**: your real stocks, ETFs and mutual funds, P&L, Indian charges and an STCG/LTCG estimate
- **Learn**: track record of every signal (by company size), F&O payoff and risk calculators (no F&O signals), guide

## Open the app

**https://sunildennisb7.github.io/stock-app/**

To install it: on iPhone, tap Share → Add to Home Screen. On Android, use Chrome ⋮ → Install app. On a Mac, use Safari → File → Add to Dock.

## Updates (automatic)

A GitHub Action (`.github/workflows/update.yml`) reruns the analysis at about 6:45 PM and 9:30 PM IST every weekday and republishes the site.
For a manual refresh, go to GitHub → Actions → **Daily update** → **Run workflow**.

## Push notifications

- **1:30 PM IST** (market days): today's buy list, with buy signals re-checked against live prices.
- **Evening**: daily summary, new STRONG BUYs, market mood changes, and watchlist alerts.

Setup: in the app open **Learn → Alerts & theme → Enable notifications**, then send the setup code to Claude.
It gets added to the `PUSH_SUBSCRIBERS` repository secret. The push signing key is the `VAPID_PRIVATE_KEY` secret
(local copy in `.secrets/`, which is never committed). On iPhone, add the app to the Home Screen first (iOS 16.4+).

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
| `core/data.py` | Nifty 500 lists from NSE, ETF list, prices and fundamentals from Yahoo Finance (cached in `cache/`) |
| `core/funds.py` | Mutual fund list (AMFI) + NAV history (mfapi.in), metrics and category ranks |
| `core/indicators.py` | Moving averages, RSI, MACD, ATR, etc. |
| `core/signals.py` | The scoring rules and labels. Change thresholds here |
| `core/backtest.py` | Backtests of the same rules |
| `core/report.py` | Builds `output/` (index.html, per-stock detail files in `d/`, `funds.json`) |
| `web/template.html` | The dashboard (HTML/CSS/JS) |
| `web/static/` | App icon, manifest and offline support (makes it installable) |
| `scripts/make_icons.py` | Regenerates the icons |

## Important

The labels come from fixed rules, not from a prediction. Over 5 years, long-term STRONG BUY stocks beat the average
Nifty 500 stock, mostly in mid and small caps. Mechanically trading the swing signals did **not** beat buy-and-hold.
All tests use today's index members, so they look better than reality (see Learn → Track record). This isn't personal investment
advice. Data is end-of-day from free sources, so check prices at your broker before trading.
