"""Runs the whole analysis and writes the app into output/:

  index.html        app + compact data for every stock/ETF (fast first load)
  d/<SYM>.json      full detail per stock/ETF (chart, reasons, plans), loaded on tap
  funds.json        mutual fund rankings, loaded when the Funds tab opens
"""
from __future__ import annotations

import json
import math
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from . import backtest, data, funds, signals
from .indicators import add_indicators

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = ROOT / "web" / "template.html"
OUT_DIR = ROOT / "output"
LORDER = ["STRONG BUY", "BUY", "HOLD / WAIT", "AVOID", "SELL / EXIT"]
LIDX = {l: i for i, l in enumerate(LORDER)}


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


def _chart(d: pd.DataFrame) -> dict:
    t = d.iloc[-260:]
    return {"d": [x.strftime("%Y-%m-%d") for x in t.index], "c": [_r(v) for v in t["Close"]],
            "a": [_r(v) for v in t["SMA50"]], "b": [_r(v) for v in t["SMA200"]]}


def _dump(obj) -> str:
    return json.dumps(obj, separators=(",", ":"), allow_nan=False)


def build(refresh: bool = False, log=print, with_funds: bool = True) -> Path:
    log("1/6  Getting Nifty 500 + ETF lists...")
    uni = data.get_universe(refresh)
    uni = uni[~uni["sym"].str.contains("DUMMY", case=False)]

    log("2/6  Downloading ~6 years of daily prices for ~550 symbols...")
    allsyms = uni["yf"].tolist() + list(data.INDICES) + list(data.SECTORS)
    prices = data.get_prices(allsyms, refresh)
    nifty = prices["^NSEI"]
    uni = uni[uni["yf"].isin(prices)]
    stocks_u, etfs_u = uni[uni["kind"] == "stock"], uni[uni["kind"] == "etf"]

    log("3/6  Fundamentals (P/E, ROE, debt, growth)...")
    funds_d = data.get_fundamentals(stocks_u["yf"].tolist(), refresh=False, log=log)

    pe_by_ind: dict[str, list[float]] = {}
    for _, r in stocks_u.iterrows():
        pe = signals._num(funds_d.get(r["yf"], {}).get("trailingPE"))
        if pe and pe > 0:
            pe_by_ind.setdefault(r["ind"], []).append(pe)
    all_pes = [p for v in pe_by_ind.values() for p in v] or [20.0]
    ind_median = {k: float(np.median(v)) if len(v) >= 3 else float(np.median(all_pes)) for k, v in pe_by_ind.items()}

    log("4/6  Scoring every stock and ETF, measuring past hit rates...")
    ind_frames = {s: add_indicators(prices[s]) for s in uni["yf"]}
    # momentum rank: stocks ranked against the Nifty 500, ETFs against other ETFs
    mom_rank, mom = {}, {}
    for group in (stocks_u, etfs_u):
        closes = pd.DataFrame({s: ind_frames[s]["Close"] for s in group["yf"]})
        m = closes.shift(21) / closes.shift(252) - 1
        mr = m.rank(axis=1, pct=True)
        for s in group["yf"]:
            mom[s], mom_rank[s] = m[s], mr[s]

    meta = uni.set_index("yf")
    frames, items, hist_s, hist_l = {}, [], [], []
    OUT_DIR.mkdir(exist_ok=True)
    det_dir = OUT_DIR / "d"
    if det_dir.exists():
        shutil.rmtree(det_dir)
    det_dir.mkdir()

    for s, d in ind_frames.items():
        mrow = meta.loc[s]
        sw = signals.swing_scores(d, nifty, mom_rank[s])
        lt = signals.long_tech_scores(d, nifty, mom_rank[s])
        d["swing_score"] = sw["score"]
        d["long_tech"] = lt["tech"]
        d["fwd_s"] = signals.forward_returns(d["Close"], signals.SWING_HORIZON)
        d["fwd_l"] = signals.forward_returns(d["Close"], signals.LONG_HORIZON)
        bucket = mrow["cap"]
        hist_s.append(pd.DataFrame({"date": d.index, "score": d["swing_score"].values, "fwd": d["fwd_s"].values, "cap": bucket}))
        hist_l.append(pd.DataFrame({"date": d.index, "score": (d["long_tech"] * 100 / 60).values, "fwd": d["fwd_l"].values, "cap": bucket}))
        if mrow["kind"] == "stock":
            frames[s] = d

        last, swl, ltl = d.iloc[-1], sw.iloc[-1], lt.iloc[-1]
        if pd.isna(swl["score"]) or pd.isna(ltl["tech"]):
            continue  # recently listed: not enough history yet
        s_score = float(swl["score"])
        s_label, s_action = signals.label_for(s_score)
        atr, entry = float(last["ATR"]), float(last["Close"])
        stop, target = entry - 2 * atr, entry + 4 * atr
        own = pd.DataFrame({"score": d["swing_score"], "fwd": d["fwd_s"]}).dropna()
        own = own[own.index >= d.index[-1] - pd.Timedelta(days=365 * 5)]
        own_same = own[signals.label_series(own["score"]) == s_label]

        if mrow["kind"] == "stock":
            f_score, f_reasons, metrics = signals.fundamental_score(funds_d.get(s, {}), mrow["ind"], ind_median.get(mrow["ind"]))
        else:
            f_score, f_reasons, metrics = None, [{"t": "ETF: score uses price trend and momentum only", "s": 0}], {}
        tech = float(ltl["tech"])
        l_score = tech + f_score if f_score is not None else tech * 100 / 60
        l_label, l_action = signals.label_for(l_score)
        perf = _perf(d["Close"])
        sym = mrow["sym"]

        items.append({
            "s": sym, "n": mrow["name"], "i": mrow["ind"], "c": bucket, "k": "e" if mrow["kind"] == "etf" else "s",
            "p": _r(entry), "ch": perf["d1"], "w1": perf["w1"], "m1": perf["m1"], "m3": perf["m3"], "y1": perf["y1"],
            "r": _r(last["RSI"], 1), "mo": _r(mom[s].iloc[-1], 4), "mr": _r(mom_rank[s].iloc[-1], 3),
            "ss": round(s_score), "sl": LIDX[s_label], "ls": round(l_score), "ll": LIDX[l_label],
            "e": _r(entry), "st": _r(stop), "t": _r(target),
            "pe": _r(metrics.get("pe"), 1), "roe": _r(metrics.get("roe"), 3),
            "mc": _r(metrics["mcap"] / 1e7, 0) if metrics.get("mcap") else None,
            "v": _r(float((d["Close"] * d["Volume"]).iloc[-20:].mean()) / 1e7, 2),  # avg daily traded value, ₹ Cr
        })
        detail = {
            "s": sym, "perf": perf, "rsi": _r(last["RSI"], 1), "sma50": _r(last["SMA50"]), "sma200": _r(last["SMA200"]),
            "hi52": _r(last["High252"]), "lo52": _r(last["Low252"]), "atr": _r(atr), "vol": _r(last["Vol252"], 3),
            "swing": {"score": round(s_score), "label": s_label, "action": s_action,
                      "reasons": signals.swing_reasons(swl, last),
                      "entry": _r(entry), "stop": _r(stop), "target": _r(target),
                      "stop_pct": _pct(stop, entry), "target_pct": _pct(target, entry),
                      "hist": {"n": int(len(own_same)),
                               "win": _r((own_same["fwd"] > 0).mean(), 3) if len(own_same) else None,
                               "avg": _r(own_same["fwd"].mean(), 4) if len(own_same) else None}},
            "long": {"score": round(l_score), "label": l_label, "action": l_action,
                     "tech": round(tech), "fund": round(f_score) if f_score is not None else None,
                     "reasons": signals.long_reasons(ltl, last) + f_reasons,
                     "zone_lo": _r(max(float(last["SMA50"]), entry * 0.95) if last["SMA50"] < entry else entry * 0.97),
                     "zone_hi": _r(entry), "review": _r(float(last["SMA200"]) * 0.97)},
            "metrics": {k: _r(v, 4) for k, v in metrics.items()},
            "chart": _chart(d),
        }
        (det_dir / f"{sym}.json").write_text(_dump(detail))

    def hits(frames_list, years=5):
        allrows = pd.concat(frames_list, ignore_index=True)
        out = {"All stocks": signals.hit_rate_table([allrows[allrows["cap"] != "ETF"]], "score", "fwd", years)}
        for cap in ["Nifty 50", "Next 50", "Midcap", "Smallcap", "ETF"]:
            sub = allrows[allrows["cap"] == cap]
            if len(sub):
                out[cap] = signals.hit_rate_table([sub], "score", "fwd", years)
        return out
    hit = {"swing": hits(hist_s), "long": hits(hist_l),
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
    sf = list(frames.values())
    br50 = float(np.mean([f["Close"].iloc[-1] > f["SMA50"].iloc[-1] for f in sf]))
    br200 = float(np.mean([f["Close"].iloc[-1] > f["SMA200"].iloc[-1] for f in sf]))
    stock_items = [x for x in items if x["k"] == "s"]
    adv = sum(1 for x in stock_items if (x["ch"] or 0) > 0)
    dec = sum(1 for x in stock_items if (x["ch"] or 0) < 0)
    idx = {name: _perf(prices[sym]["Close"]) for sym, name in data.INDICES.items() if sym in prices}
    sectors = []
    for sym, name in data.SECTORS.items():
        if sym in prices:
            sd = add_indicators(prices[sym])
            p = _perf(sd["Close"])
            c, a, b = sd["Close"].iloc[-1], sd["SMA50"].iloc[-1], sd["SMA200"].iloc[-1]
            p.update({"name": name, "trend": "Up" if c > a > b else "Down" if c < a < b else "Mixed"})
            sectors.append(p)
    market = {
        "regime": regime[0], "regime_text": regime[1], "indices": idx,
        "nifty": {"close": _r(nl["Close"]), "sma50": _r(nl["SMA50"]), "sma200": _r(nl["SMA200"]),
                  "rsi": _r(nl["RSI"], 1), "chart": _chart(nd)},
        "breadth50": _r(br50, 3), "breadth200": _r(br200, 3), "adv": adv, "dec": dec,
        "sectors": sorted(sectors, key=lambda x: -(x["m1"] or -9)),
    }

    log("5/6  Running backtests of the same rules on the Nifty 500...")
    bt = {"swing": backtest.swing_backtest(frames, nifty, max_pos=10),
          "long": backtest.long_backtest(frames, nifty, top_n=15)}
    ew = backtest.equal_weight_stats(frames, nifty)
    for k in bt:
        bt[k]["stats"]["ew"] = ew

    fund_info = {}
    if with_funds:
        log("6/6  Ranking mutual funds...")
        try:
            fd = funds.build_funds(log)
            (OUT_DIR / "funds.json").write_text(_dump(fd))
            fund_info = {"count": len(fd["funds"]), "navdate": fd["navdate"]}
        except Exception as e:  # noqa: BLE001
            log(f"  ! mutual fund data unavailable today ({type(e).__name__}: {e})")

    payload = {
        "generated": datetime.now().strftime("%d %b %Y, %I:%M %p"),
        "data_date": nifty.index[-1].strftime("%d %b %Y"),
        "labels": LORDER, "market": market, "items": items, "hit": hit, "backtest": bt, "funds": fund_info,
    }
    blob = _dump(payload).replace("</", "<\\/")
    (OUT_DIR / "index.html").write_text(TEMPLATE.read_text().replace("/*__DATA__*/null", blob))
    for f in (ROOT / "web" / "static").iterdir():
        shutil.copy2(f, OUT_DIR / f.name)
    (OUT_DIR / ".nojekyll").write_text("")
    log(f"     {len(stock_items)} stocks, {len(items) - len(stock_items)} ETFs, index.html {len(blob)//1024} KB")
    return OUT_DIR / "index.html"
