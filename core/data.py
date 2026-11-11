"""Data fetching: Nifty 50 constituents (from NSE), prices and fundamentals (Yahoo Finance).

Everything is cached in ./cache so repeated runs are fast.
"""
from __future__ import annotations

import io
import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "cache"
CACHE.mkdir(exist_ok=True)

NIFTY50_URL = "https://nsearchives.nseindia.com/content/indices/ind_nifty50list.csv"
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}

# Market-wide indices shown on the Market tab (Yahoo symbols)
INDICES = {
    "^NSEI": "Nifty 50",
    "^NSEBANK": "Nifty Bank",
    "^INDIAVIX": "India VIX",
}
SECTORS = {
    "^CNXIT": "IT",
    "^CNXAUTO": "Auto",
    "^CNXPHARMA": "Pharma",
    "^CNXFMCG": "FMCG",
    "^CNXMETAL": "Metal",
    "^CNXENERGY": "Energy",
    "^CNXREALTY": "Realty",
    "^CNXPSUBANK": "PSU Bank",
    "^CNXMEDIA": "Media",
    "^CNXINFRA": "Infra",
    "NIFTY_FIN_SERVICE.NS": "Financial Services",
}


def _fresh(path: Path, max_age: timedelta) -> bool:
    return path.exists() and datetime.now() - datetime.fromtimestamp(path.stat().st_mtime) < max_age


def get_constituents(refresh: bool = False) -> pd.DataFrame:
    """Current Nifty 50 list from NSE; falls back to the last saved copy if NSE is unreachable."""
    path = CACHE / "nifty50.csv"
    if not refresh and _fresh(path, timedelta(days=7)):
        return pd.read_csv(path)
    try:
        r = requests.get(NIFTY50_URL, headers=UA, timeout=20)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        if len(df) < 45:
            raise ValueError("unexpected constituent list")
        df.to_csv(path, index=False)
        return df
    except Exception as e:  # noqa: BLE001
        if path.exists():
            print(f"  ! Could not refresh Nifty 50 list from NSE ({e}); using saved copy.")
            return pd.read_csv(path)
        raise


def get_prices(symbols: list[str], refresh: bool = False, years: int = 6) -> dict[str, pd.DataFrame]:
    """Daily OHLCV for each Yahoo symbol. Cached for 6 hours."""
    path = CACHE / "prices.pkl"
    if not refresh and _fresh(path, timedelta(hours=6)):
        cached = pd.read_pickle(path)
        if all(s in cached for s in symbols):
            return cached

    raw = None
    for attempt in range(3):
        try:
            raw = yf.download(symbols, period=f"{years}y", interval="1d", group_by="ticker",
                              auto_adjust=True, progress=False, threads=True)
            break
        except Exception as e:  # noqa: BLE001
            print(f"  ! download attempt {attempt + 1} failed: {e}")
            time.sleep(3)
    if raw is None or raw.empty:
        if path.exists():
            print("  ! Using previously saved prices.")
            return pd.read_pickle(path)
        raise RuntimeError("Could not download price data. Check your internet connection.")

    out: dict[str, pd.DataFrame] = {}
    for s in symbols:
        try:
            df = raw[s].dropna(subset=["Close"]).copy()
        except KeyError:
            continue
        if len(df) < 60:
            continue
        df.index = pd.to_datetime(df.index).tz_localize(None)
        out[s] = df[["Open", "High", "Low", "Close", "Volume"]]
    pd.to_pickle(out, path)
    return out


FUND_FIELDS = ["trailingPE", "forwardPE", "priceToBook", "returnOnEquity", "debtToEquity",
               "profitMargins", "revenueGrowth", "earningsGrowth", "dividendYield",
               "marketCap", "sector", "industry", "longName"]


def _one_info(sym: str) -> tuple[str, dict]:
    for attempt in range(2):
        try:
            info = yf.Ticker(sym).info or {}
            return sym, {k: info.get(k) for k in FUND_FIELDS}
        except Exception:  # noqa: BLE001
            time.sleep(2)
    return sym, {}


def get_fundamentals(symbols: list[str], refresh: bool = False) -> dict[str, dict]:
    """Key fundamentals per stock. These change quarterly, so cached for 7 days."""
    path = CACHE / "fundamentals.json"
    cached: dict[str, dict] = {}
    if path.exists():
        cached = json.loads(path.read_text())
    if not refresh and _fresh(path, timedelta(days=7)) and all(s in cached and cached[s] for s in symbols):
        return cached
    todo = [s for s in symbols if refresh or not cached.get(s) or not _fresh(path, timedelta(days=7))]
    with ThreadPoolExecutor(max_workers=6) as ex:
        for sym, info in ex.map(_one_info, todo):
            if info:
                cached[sym] = info
    path.write_text(json.dumps(cached, indent=1))
    return cached
