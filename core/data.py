"""Data fetching: index constituents (NSE), prices and fundamentals (Yahoo Finance).

Everything is cached in ./cache so repeated runs are fast. Constituent CSVs are also
committed to the repo as a fallback for when NSE blocks cloud servers.
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

UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
NSE_LIST = "https://nsearchives.nseindia.com/content/indices/{}.csv"

# Size buckets: Nifty 500 = Nifty 50 + Next 50 + Midcap 150 + Smallcap 250
BUCKETS = [("ind_nifty50list", "Nifty 50"), ("ind_niftynext50list", "Next 50"),
           ("ind_niftymidcap150list", "Midcap"), ("ind_niftysmallcap250list", "Smallcap")]

INDICES = {
    "^NSEI": "Nifty 50",
    "^NSEBANK": "Nifty Bank",
    "NIFTYMIDCAP150.NS": "Midcap 150",
    "NIFTYSMLCAP250.NS": "Smallcap 250",
    "^INDIAVIX": "India VIX",
}
SECTORS = {
    "^CNXIT": "IT", "^CNXAUTO": "Auto", "^CNXPHARMA": "Pharma", "^CNXFMCG": "FMCG",
    "^CNXMETAL": "Metal", "^CNXENERGY": "Energy", "^CNXREALTY": "Realty",
    "^CNXPSUBANK": "PSU Bank", "^CNXINFRA": "Infra", "NIFTY_FIN_SERVICE.NS": "Financial Services",
}

# Exchange-traded funds on NSE: symbol -> (name, category)
ETFS = {
    "NIFTYBEES": ("Nippon Nifty 50 BeES", "Index"), "SETFNIF50": ("SBI Nifty 50 ETF", "Index"),
    "JUNIORBEES": ("Nippon Nifty Next 50", "Index"),
    "BSE500IETF": ("ICICI BSE 500 ETF", "Index"), "BANKBEES": ("Nippon Nifty Bank BeES", "Index"),
    "MIDCAPETF": ("Mirae Midcap 150", "Mid & Small"),
    "MOM100": ("Motilal Midcap 100", "Mid & Small"), "HDFCSML250": ("HDFC Smallcap 250", "Mid & Small"),
    "SMALLCAP": ("Mirae Smallcap 250", "Mid & Small"),
    "MOMOMENTUM": ("Motilal Momentum 30", "Factor"), "ALPHA": ("Kotak Alpha 50", "Factor"),
    "MOVALUE": ("Motilal Value 50", "Factor"), "NV20IETF": ("ICICI Nifty Value 20", "Factor"),
    "LOWVOLIETF": ("ICICI Low Vol 30", "Factor"),
    "ITBEES": ("Nippon IT BeES", "Sector"), "PSUBNKBEES": ("Nippon PSU Bank BeES", "Sector"),
    "PHARMABEES": ("Nippon Pharma BeES", "Sector"), "HEALTHIETF": ("ICICI Healthcare ETF", "Sector"),
    "AUTOBEES": ("Nippon Auto BeES", "Sector"), "INFRABEES": ("Nippon Infra BeES", "Sector"),
    "CONSUMBEES": ("Nippon Consumption BeES", "Sector"), "CPSEETF": ("Nippon CPSE ETF", "Sector"),
    "MODEFENCE": ("Motilal Defence ETF", "Sector"), "EVINDIA": ("Mirae EV & New Age Auto", "Sector"),
    "GOLDBEES": ("Nippon Gold BeES", "Gold & Silver"), "GOLDIETF": ("ICICI Gold ETF", "Gold & Silver"),
    "SILVERBEES": ("Nippon Silver BeES", "Gold & Silver"), "SILVERIETF": ("ICICI Silver ETF", "Gold & Silver"),
    "MON100": ("Motilal Nasdaq 100", "International"), "MAFANG": ("Mirae NYSE FANG+", "International"),
    "MASPTOP50": ("Mirae S&P 500 Top 50", "International"), "HNGSNGBEES": ("Nippon Hang Seng BeES", "International"),
    "LIQUIDBEES": ("Nippon Liquid BeES", "Debt & Liquid"), "GILT5YBEES": ("Nippon 5Y Gilt", "Debt & Liquid"),
    "LTGILTBEES": ("Nippon Long Term Gilt", "Debt & Liquid"),
}


def _fresh(path: Path, max_age: timedelta) -> bool:
    return path.exists() and datetime.now() - datetime.fromtimestamp(path.stat().st_mtime) < max_age


def _nse_csv(name: str, refresh: bool) -> pd.DataFrame:
    path = CACHE / f"{name}.csv"
    if not refresh and _fresh(path, timedelta(days=7)):
        return pd.read_csv(path)
    try:
        r = requests.get(NSE_LIST.format(name), headers=UA, timeout=20)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        if len(df) < 40:
            raise ValueError("unexpected constituent list")
        df.to_csv(path, index=False)
        return df
    except Exception as e:  # noqa: BLE001
        if path.exists():
            print(f"  ! Could not refresh {name} from NSE ({type(e).__name__}); using saved copy.")
            return pd.read_csv(path)
        raise


def get_universe(refresh: bool = False) -> pd.DataFrame:
    """Nifty 500 stocks with size bucket + the ETF list, as one table."""
    rows = []
    seen = set()
    for fname, bucket in BUCKETS:
        for _, r in _nse_csv(fname, refresh).iterrows():
            sym = str(r["Symbol"]).strip()
            if sym in seen:
                continue
            seen.add(sym)
            rows.append({"sym": sym, "yf": sym + ".NS", "name": r["Company Name"], "ind": r["Industry"],
                         "cap": bucket, "kind": "stock"})
    # anything in the Nifty 500 list not covered above (rare, during index changes)
    try:
        for _, r in _nse_csv("ind_nifty500list", refresh).iterrows():
            sym = str(r["Symbol"]).strip()
            if sym not in seen:
                seen.add(sym)
                rows.append({"sym": sym, "yf": sym + ".NS", "name": r["Company Name"], "ind": r["Industry"],
                             "cap": "Smallcap", "kind": "stock"})
    except Exception:  # noqa: BLE001
        pass
    for sym, (name, cat) in ETFS.items():
        rows.append({"sym": sym, "yf": sym + ".NS", "name": name, "ind": cat, "cap": "ETF", "kind": "etf"})
    return pd.DataFrame(rows)


def get_prices(symbols: list[str], refresh: bool = False, years: int = 6) -> dict[str, pd.DataFrame]:
    """Daily OHLCV for each Yahoo symbol, downloaded in batches. Cached for 6 hours."""
    path = CACHE / "prices.pkl"
    cached: dict[str, pd.DataFrame] = pd.read_pickle(path) if path.exists() else {}
    if not refresh and _fresh(path, timedelta(hours=6)) and all(s in cached for s in symbols):
        return cached

    out: dict[str, pd.DataFrame] = {}

    def _take(s, df):
        df = df.dropna(subset=["Close"]).copy()
        if len(df) >= 60:
            df.index = pd.to_datetime(df.index).tz_localize(None)
            out[s] = df[["Open", "High", "Low", "Close", "Volume"]]

    for i in range(0, len(symbols), 100):
        batch = symbols[i:i + 100]
        for attempt in range(3):
            try:
                raw = yf.download(batch, period=f"{years}y", interval="1d", group_by="ticker",
                                  auto_adjust=True, progress=False, threads=True)
                for s in batch:
                    try:
                        _take(s, raw[s])
                    except KeyError:
                        pass
                break
            except Exception as e:  # noqa: BLE001
                print(f"  ! batch download failed ({e}); retrying")
                time.sleep(3)
    # yfinance's parallel download sometimes drops symbols ("database is locked"); retry one by one
    missing = [s for s in symbols if s not in out]
    for s in missing:
        for attempt in range(2):
            try:
                one = yf.download(s, period=f"{years}y", interval="1d", auto_adjust=True,
                                  progress=False, threads=False, multi_level_index=False)
                if not one.empty:
                    _take(s, one)
                    break
            except Exception:  # noqa: BLE001
                pass
            time.sleep(1)
        if s not in out and s in cached:
            out[s] = cached[s]  # fall back to the last good copy
    still = [s for s in symbols if s not in out]
    if still:
        print(f"  ! no price data for {len(still)} symbols: {', '.join(x.replace('.NS', '') for x in still[:15])}"
              f"{' ...' if len(still) > 15 else ''}")
    if not out:
        raise RuntimeError("Could not download price data. Check your internet connection.")
    pd.to_pickle(out, path)
    return out


FUND_FIELDS = ["trailingPE", "forwardPE", "priceToBook", "returnOnEquity", "debtToEquity",
               "profitMargins", "revenueGrowth", "earningsGrowth", "dividendYield", "marketCap"]


def _one_info(sym: str) -> tuple[str, dict]:
    for attempt in range(2):
        try:
            info = yf.Ticker(sym).info or {}
            if info.get("marketCap") or info.get("trailingPE") or info.get("returnOnEquity"):
                return sym, {k: info.get(k) for k in FUND_FIELDS}
        except Exception:  # noqa: BLE001
            pass
        time.sleep(1.5)
    return sym, {}


def get_fundamentals(symbols: list[str], refresh: bool = False, max_per_run: int = 600,
                     max_age_days: int = 7, log=print) -> dict[str, dict]:
    """Key fundamentals per stock (change quarterly). Each stock is refreshed when older than 7 days,
    at most `max_per_run` per run so Yahoo doesn't throttle us."""
    path = CACHE / "fundamentals.json"
    cached: dict[str, dict] = json.loads(path.read_text()) if path.exists() else {}
    now = time.time()

    def age(s):
        return (now - cached.get(s, {}).get("_ts", 0)) / 86400

    todo = [s for s in symbols if refresh or age(s) > max_age_days or not cached.get(s, {}).get("_ok")]
    todo.sort(key=lambda s: -age(s))
    todo = todo[:max_per_run]
    if todo:
        log(f"     fetching fundamentals for {len(todo)} stocks...")
        with ThreadPoolExecutor(max_workers=8) as ex:
            for sym, info in ex.map(_one_info, todo):
                if info:
                    cached[sym] = {**info, "_ts": now, "_ok": True}
                elif sym not in cached:
                    cached[sym] = {"_ts": now - (max_age_days - 1) * 86400, "_ok": False}  # retry tomorrow
        path.write_text(json.dumps(cached))
    return {s: {k: v for k, v in cached.get(s, {}).items() if not k.startswith("_")} for s in symbols}
