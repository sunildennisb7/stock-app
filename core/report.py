"""Runs the whole analysis and writes the dashboard (output/index.html)."""
from __future__ import annotations

import json
import math
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from . import backtest, data, signals
from .indicators import add_indicators

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = ROOT / "web" / "template.html"
OUT_DIR = ROOT / "output"


def _r(x, n=2):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(x) or math.isinf(x) else round(x, n)


def _pct(a, b):
    return _r(a / b - 1, 4) if a and b else None


def _perf(close: pd.Series) -> dict:
    c = close.dropna()
    last = c.iloc[-1]
    def back(n):
        return _pct(last, c.iloc[-1 - n]) if len(c) > n else None
    return {"last": _r(last), "d1": back(1), "w1": back(5), "m1": back(21), "m3": back(63), "y1": back(252)}


def build(refresh: bool = False, log=print) -> Path:
    log("1/5  Getting the Nifty 50 list from NSE...")
    cons = data.get_constituents(refresh)
    cons["yf"] = cons["Symbol"].astype(str) + ".NS"
    syms = cons["yf"].tolist()
    industry = dict(zip(cons["yf"], cons["Industry"]))
    names = dict(zip(cons["yf"], cons["Company Name"]))

    log("2/5  Downloading ~6 years of daily prices (stocks + indices)...")
    allsyms = syms + list(data.INDICES) + list(data.SECTORS)
    prices = data.get_prices(allsyms, refresh)
    nifty = prices["^NSEI"]

    log("3/5  Fetching fundamentals (P/E, ROE, debt, growth)...")
    funds = data.get_fundamentals([s for s in syms if s in prices], refresh)

    # sector P/E medians for the valuation check
    pe_by_ind: dict[str, list[float]] = {}
    for s in syms:
        pe = signals._num((funds.get(s) or {}).get("trailingPE"))
        if pe and pe > 0:
            pe_by_ind.setdefault(industry[s], []).append(pe)
    all_pes = [p for v in pe_by_ind.values() for p in v]
    ind_median = {k: float(np.median(v)) if len(v) >= 3 else float(np.median(all_pes)) for k, v in pe_by_ind.items()}

    log("4/5  Scoring every stock and measuring past hit rates...")
    ind_frames = {}
    for s in syms:
        if s not in prices:
            log(f"     ! no price data for {s}, skipped")
            continue
        ind_frames[s] = add_indicators(prices[s])
    # 12-month momentum skipping the latest month, ranked across the Nifty 50 each day (0 = worst, 1 = best)
    closes = pd.DataFrame({s: f["Close"] for s, f in ind_frames.items()})
    mom = closes.shift(21) / closes.shift(252) - 1
    mom_rank = mom.rank(axis=1, pct=True)

    frames, stocks, swing_hist, long_hist = {}, [], [], []
    for s, d in ind_frames.items():
        sw = signals.swing_scores(d, nifty, mom_rank[s])
        lt = signals.long_tech_scores(d, nifty, mom_rank[s])
        d["swing_score"] = sw["score"]
        d["long_tech"] = lt["tech"]
        d["fwd_s"] = signals.forward_returns(d["Close"], signals.SWING_HORIZON)
        d["fwd_l"] = signals.forward_returns(d["Close"], signals.LONG_HORIZON)
        frames[s] = d
        swing_hist.append(pd.DataFrame({"date": d.index, "score": d["swing_score"].values, "fwd": d["fwd_s"].values}))
        long_hist.append(pd.DataFrame({"date": d.index, "score": (d["long_tech"] * 100 / 60).values, "fwd": d["fwd_l"].values}))

        last, swl, ltl = d.iloc[-1], sw.iloc[-1], lt.iloc[-1]
        if pd.isna(swl["score"]) or pd.isna(ltl["tech"]):
            log(f"     ! not enough history for {s}, skipped")
            continue

        # swing
        s_score = float(swl["score"])
        s_label, s_action = signals.label_for(s_score)
        atr = float(last["ATR"])
        entry = float(last["Close"])
        stop = entry - 2 * atr
        target = entry + 4 * atr
        own = pd.DataFrame({"score": d["swing_score"], "fwd": d["fwd_s"]}).dropna()
        own = own[own.index >= d.index[-1] - pd.Timedelta(days=365 * 5)]
        own_same = own[signals.label_series(own["score"]) == s_label]

        # long term
        f_score, f_reasons, metrics = signals.fundamental_score(funds.get(s, {}), industry[s], ind_median.get(industry[s]))
        tech = float(ltl["tech"])
        l_score = tech + f_score if f_score is not None else tech * 100 / 60
        l_label, l_action = signals.label_for(l_score)

        tail = d.iloc[-260:]
        stocks.append({
            "sym": s.replace(".NS", ""), "name": names[s], "ind": industry[s],
            "price": _r(entry), "chg": _pct(entry, d["Close"].iloc[-2]),
            "perf": _perf(d["Close"]),
            "rsi": _r(last["RSI"], 1), "sma50": _r(last["SMA50"]), "sma200": _r(last["SMA200"]),
            "hi52": _r(last["High252"]), "lo52": _r(last["Low252"]), "atr": _r(atr),
            "vol": _r(last["Vol252"], 3), "mom": _r(mom[s].iloc[-1], 4), "mom_rank": _r(mom_rank[s].iloc[-1], 3),
            "swing": {"score": round(s_score), "label": s_label, "action": s_action,
                      "reasons": signals.swing_reasons(swl, last),
                      "entry": _r(entry), "stop": _r(stop), "target": _r(target),
                      "stop_pct": _pct(stop, entry), "target_pct": _pct(target, entry),
                      "hist": {"n": int(len(own_same)), "win": _r((own_same["fwd"] > 0).mean(), 3) if len(own_same) else None,
                               "avg": _r(own_same["fwd"].mean(), 4) if len(own_same) else None}},
            "long": {"score": round(l_score), "label": l_label, "action": l_action,
                     "tech": round(tech), "fund": round(f_score) if f_score is not None else None,
                     "reasons": signals.long_reasons(ltl, last) + f_reasons,
                     "zone_lo": _r(max(float(last["SMA50"]), entry * 0.95) if last["SMA50"] < entry else entry * 0.97),
                     "zone_hi": _r(entry),
                     "review": _r(float(last["SMA200"]) * 0.97)},
            "metrics": {k: _r(v, 4) for k, v in metrics.items()},
            "chart": {"d": [x.strftime("%Y-%m-%d") for x in tail.index],
                      "c": [_r(v) for v in tail["Close"]], "a": [_r(v) for v in tail["SMA50"]],
                      "b": [_r(v) for v in tail["SMA200"]]},
        })

    hit = {"swing": signals.hit_rate_table(swing_hist, "score", "fwd"),
           "long": signals.hit_rate_table(long_hist, "score", "fwd"),
           "swing_days": signals.SWING_HORIZON, "long_days": signals.LONG_HORIZON}

    # ------------------------------------------------------------- market overview
    nd = add_indicators(nifty)
    nl = nd.iloc[-1]
    above50, above200 = nl["Close"] > nl["SMA50"], nl["Close"] > nl["SMA200"]
    if above200 and above50 and nl["SMA50"] > nl["SMA200"]:
        regime = ("BULLISH", "Nifty is in an uptrend. Buy signals are more reliable in this phase.")
    elif above200 and above50:
        regime = ("CAUTIOUS", "Nifty has bounced above its 50-day and 200-day averages, but the 50-day is still below the 200-day, "
                              "so the longer uptrend isn't confirmed yet. Be selective.")
    elif above200:
        regime = ("CAUTIOUS", "Nifty is above its long-term average but has slipped short-term. Be selective; use smaller positions.")
    else:
        regime = ("BEARISH", "Nifty is below its 200-day average. Most buy signals fail more often now. Prefer cash and only the strongest stocks.")
    br50 = np.mean([f["Close"].iloc[-1] > f["SMA50"].iloc[-1] for f in frames.values()])
    br200 = np.mean([f["Close"].iloc[-1] > f["SMA200"].iloc[-1] for f in frames.values()])
    adv = sum(1 for st in stocks if (st["chg"] or 0) > 0)
    dec = sum(1 for st in stocks if (st["chg"] or 0) < 0)
    idx = {name: _perf(prices[sym]["Close"]) for sym, name in data.INDICES.items() if sym in prices}
    sectors = []
    for sym, name in data.SECTORS.items():
        if sym in prices:
            sd = add_indicators(prices[sym])
            p = _perf(sd["Close"])
            p.update({"name": name, "trend": "Up" if sd["Close"].iloc[-1] > sd["SMA50"].iloc[-1] > sd["SMA200"].iloc[-1]
                      else "Down" if sd["Close"].iloc[-1] < sd["SMA50"].iloc[-1] < sd["SMA200"].iloc[-1] else "Mixed"})
            sectors.append(p)
    ntail = nd.iloc[-260:]
    market = {
        "regime": regime[0], "regime_text": regime[1], "indices": idx,
        "nifty": {"close": _r(nl["Close"]), "sma50": _r(nl["SMA50"]), "sma200": _r(nl["SMA200"]), "rsi": _r(nl["RSI"], 1),
                  "chart": {"d": [x.strftime("%Y-%m-%d") for x in ntail.index], "c": [_r(v) for v in ntail["Close"]],
                            "a": [_r(v) for v in ntail["SMA50"]], "b": [_r(v) for v in ntail["SMA200"]]}},
        "breadth50": _r(br50, 3), "breadth200": _r(br200, 3), "adv": adv, "dec": dec,
        "sectors": sorted(sectors, key=lambda x: -(x["m1"] or -9)),
    }

    log("5/5  Running backtests of the same rules...")
    bt = {"swing": backtest.swing_backtest(frames, nifty), "long": backtest.long_backtest(frames, nifty)}
    for k in bt:
        bt[k]["stats"]["ew"] = backtest.equal_weight_stats(frames, nifty)

    payload = {
        "generated": datetime.now().strftime("%d %b %Y, %I:%M %p"),
        "data_date": nifty.index[-1].strftime("%d %b %Y"),
        "market": market, "stocks": stocks, "hit": hit, "backtest": bt,
    }
    OUT_DIR.mkdir(exist_ok=True)
    blob = json.dumps(payload, separators=(",", ":"), allow_nan=False).replace("</", "<\\/")
    html = TEMPLATE.read_text().replace("/*__DATA__*/null", blob)
    out = OUT_DIR / "index.html"
    out.write_text(html)
    (OUT_DIR / "data.json").write_text(json.dumps(payload, indent=1, allow_nan=False))
    for f in (ROOT / "web" / "static").iterdir():
        shutil.copy2(f, OUT_DIR / f.name)
    (OUT_DIR / ".nojekyll").write_text("")
    return out
