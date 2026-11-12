"""Mutual fund comparison: every open-ended Direct-Growth fund in the main categories,
ranked within its category on past returns, risk and drawdown.

Sources: AMFI scheme list (portal.amfiindia.com) + NAV history from api.mfapi.in.
NAV histories are cached for 7 days (cache/fund_navs.pkl).
"""
from __future__ import annotations

import math
import re
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests

from .data import CACHE, UA

AMFI_URL = "https://portal.amfiindia.com/spages/NAVAll.txt"
MFAPI = "https://api.mfapi.in/mf/{}"
RISK_FREE = 0.065  # rough Indian risk-free rate (T-bill) for the Sharpe ratio

# (my category, group, AMFI category keywords). First match wins.
CATS = [
    ("Large Cap", "Equity", ["- Large Cap"]),
    ("Large & Mid Cap", "Equity", ["Large & Mid Cap"]),
    ("Mid Cap", "Equity", ["- Mid Cap"]),
    ("Small Cap", "Equity", ["- Small Cap"]),
    ("Flexi Cap", "Equity", ["Flexi Cap"]),
    ("Multi Cap", "Equity", ["Multi Cap"]),
    ("ELSS (Tax saver)", "Equity", ["ELSS"]),
    ("Focused", "Equity", ["Focused"]),
    ("Value & Contra", "Equity", ["Value Fund", "Contra"]),
    ("Dividend Yield", "Equity", ["Dividend Yield"]),
    ("Sectoral & Thematic", "Equity", ["Sectoral", "Thematic"]),
    ("Index Funds", "Index", ["Index Funds"]),
    ("International", "Other", ["FoF Overseas", "investing overseas"]),
    ("Gold & Silver", "Other", ["Gold", "Silver"]),
    ("Aggressive Hybrid", "Hybrid", ["Aggressive Hybrid"]),
    ("Balanced Advantage", "Hybrid", ["Balanced Advantage", "Dynamic Asset Allocation"]),
    ("Multi Asset", "Hybrid", ["Multi Asset"]),
    ("Equity Savings", "Hybrid", ["Equity Savings"]),
    ("Arbitrage", "Hybrid", ["Arbitrage"]),
    ("Liquid & Money Market", "Debt", ["Liquid Fund", "Money Market", "Overnight"]),
    ("Short Duration Debt", "Debt", ["Short Duration", "Short Term Fund", "Ultra Short", "Low Duration"]),
    ("Corporate & PSU Bond", "Debt", ["Corporate Bond", "Banking and PSU"]),
    ("Gilt", "Debt", ["Gilt"]),
]
DEBT_INDEX = re.compile(r"SDL|Gilt|G-Sec|GSec|IBX|CRISIL|Bond|PSU Bond|Target Maturity|Debt|T-Bill|Liquid|1D Rate|Money Market", re.I)
SKIP = re.compile(r"Close Ended|Interval|Fixed Term|FMP|Series|Segregated", re.I)


def _category(amfi_cat: str, name: str) -> tuple[str, str] | None:
    if SKIP.search(amfi_cat) or SKIP.search(name):
        return None
    if "FoF Domestic" in amfi_cat or "Fund of Funds Scheme (Domestic)" in amfi_cat:
        if re.search(r"gold|silver|precious", name, re.I):
            return "Gold & Silver", "Other"
        return None
    if "ETF" in amfi_cat:
        return None  # ETFs are covered on the Picks tab with live prices
    for cat, group, keys in CATS:
        if any(k in amfi_cat for k in keys):
            if cat == "Index Funds" and DEBT_INDEX.search(name):
                return None
            if cat == "Gold & Silver" and not re.search(r"gold|silver", name, re.I):
                continue
            return cat, group
    return None


def scheme_list() -> pd.DataFrame:
    path = CACHE / "amfi_navall.txt"
    try:
        r = requests.get(AMFI_URL, headers=UA, timeout=60)
        r.raise_for_status()
        path.write_text(r.text)
        text = r.text
    except Exception as e:  # noqa: BLE001
        if not path.exists():
            raise
        print(f"  ! AMFI list unavailable ({type(e).__name__}); using saved copy.")
        text = path.read_text()
    rows, cat, house = [], "", ""
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(("Open Ended", "Close Ended", "Interval")):
            cat = line
            continue
        p = line.split(";")
        if len(p) < 8 or not p[0].isdigit():
            if len(p) == 1:
                house = line
            continue
        if not cat.startswith("Open Ended") or "Direct" not in p[4] or "Growth" not in p[5]:
            continue
        c = _category(cat, p[3])
        if c:
            rows.append({"code": int(p[0]), "name": p[3].strip(), "house": house, "cat": c[0], "group": c[1],
                         "nav": float(p[6]) if p[6].replace(".", "", 1).isdigit() else None, "navdate": p[7]})
    return pd.DataFrame(rows).drop_duplicates("code")


def _fetch_nav(code: int) -> tuple[int, pd.Series | None]:
    for attempt in range(3):
        try:
            r = requests.get(MFAPI.format(code), headers=UA, timeout=30)
            r.raise_for_status()
            d = r.json().get("data") or []
            if not d:
                return code, None
            s = pd.Series({pd.to_datetime(x["date"], format="%d-%m-%Y"): float(x["nav"]) for x in d}).sort_index()
            s = s[s > 0]
            return code, s.resample("W-FRI").last().dropna()  # weekly is plenty and keeps the cache small
        except Exception:  # noqa: BLE001
            time.sleep(1 + attempt)
    return code, None


def nav_histories(codes: list[int], log=print) -> dict[int, pd.Series]:
    path = CACHE / "fund_navs.pkl"
    cached: dict = pd.read_pickle(path) if path.exists() else {}
    navs: dict[int, pd.Series] = cached.get("navs", {})
    ts: dict[int, float] = cached.get("ts", {})
    now = time.time()
    todo = [c for c in codes if now - ts.get(c, 0) > 7 * 86400]
    if todo:
        log(f"     fetching NAV history for {len(todo)} funds...")
        with ThreadPoolExecutor(max_workers=10) as ex:
            for code, s in ex.map(_fetch_nav, todo):
                if s is not None and len(s) > 10:
                    navs[code] = s
                    ts[code] = now
        pd.to_pickle({"navs": navs, "ts": ts}, path)
    return navs


def _metrics(s: pd.Series, latest_nav: float | None) -> dict:
    s = s.copy()
    if latest_nav:
        s.iloc[-1] = latest_nav
    end = s.index[-1]

    def cagr(years):
        start = end - pd.Timedelta(days=int(365.25 * years))
        past = s[s.index <= start]
        if past.empty or (s.index[0] > start + pd.Timedelta(days=10)):
            return None
        return (s.iloc[-1] / past.iloc[-1]) ** (1 / years) - 1

    r3 = s[s.index >= end - pd.Timedelta(days=int(365.25 * 3))].pct_change().dropna()
    vol = float(r3.std() * math.sqrt(52)) if len(r3) > 50 else None
    c3 = cagr(3)
    sharpe = (c3 - RISK_FREE) / vol if c3 is not None and vol else None
    w5 = s[s.index >= end - pd.Timedelta(days=int(365.25 * 5))]
    dd = float((w5 / w5.cummax() - 1).min()) if len(w5) > 20 else None
    # consistency: share of rolling 1-year windows (last 3 years) with a positive return
    yr = s.pct_change(52).dropna()
    yr = yr[yr.index >= end - pd.Timedelta(days=int(365.25 * 3))]
    cons = float((yr > 0).mean()) if len(yr) > 20 else None
    age = (end - s.index[0]).days / 365.25
    return {"r1": cagr(1), "r3": c3, "r5": cagr(5), "r10": cagr(10), "vol": vol, "sharpe": sharpe,
            "dd": dd, "cons": cons, "age": age}


def build_funds(log=print) -> dict:
    sl = scheme_list()
    log(f"     {len(sl)} direct-growth funds in {sl['cat'].nunique()} categories")
    navs = nav_histories(sl["code"].tolist(), log)
    rows = []
    for _, f in sl.iterrows():
        s = navs.get(f["code"])
        if s is None or len(s) < 30:
            continue
        m = _metrics(s, f["nav"])
        rows.append({**f.to_dict(), **m})
    df = pd.DataFrame(rows)

    # Score within category: needs 3+ years of history.
    def pr(x, higher=True):
        return x.rank(pct=True, ascending=higher)
    df["score"] = np.nan
    for cat, g in df.groupby("cat"):
        ok = g[g["r3"].notna()]
        if len(ok) < 3:
            continue
        r5 = pr(ok["r5"]).fillna(pr(ok["r3"]))
        sc = 0.35 * pr(ok["r3"]) + 0.20 * r5 + 0.25 * pr(ok["sharpe"]).fillna(0.5) + 0.20 * pr(ok["dd"]).fillna(0.5)
        df.loc[ok.index, "score"] = (sc.rank(pct=True) * 100).round()
        df.loc[ok.index, "rank"] = sc.rank(ascending=False, method="min").astype(int)
        df.loc[ok.index, "of"] = len(ok)

    def rnd(v, n=4):
        return None if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))) else round(float(v), n)

    out = []
    for _, r in df.iterrows():
        out.append({"c": int(r["code"]), "n": r["name"], "h": r["house"], "cat": r["cat"], "g": r["group"],
                    "nav": rnd(r["nav"], 3), "r1": rnd(r["r1"]), "r3": rnd(r["r3"]), "r5": rnd(r["r5"]),
                    "r10": rnd(r["r10"]), "vol": rnd(r["vol"]), "sh": rnd(r["sharpe"], 2), "dd": rnd(r["dd"]),
                    "cons": rnd(r["cons"], 3), "age": rnd(r["age"], 1),
                    "score": None if pd.isna(r["score"]) else int(r["score"]),
                    "rank": None if pd.isna(r.get("rank")) else int(r["rank"]),
                    "of": None if pd.isna(r.get("of")) else int(r["of"])})
    cats = [{"cat": c, "g": g} for c, g, _ in CATS if c in set(df["cat"])]
    return {"funds": out, "cats": cats, "navdate": sl["navdate"].mode().iat[0] if len(sl) else ""}
