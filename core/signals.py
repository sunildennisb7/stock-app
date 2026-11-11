"""Signal rules.

Two independent scores, each 0-100:
  * SWING     (holding days to a few weeks)  - trend + momentum + entry timing
  * LONG-TERM (holding months to years)      - long trend + relative strength + fundamentals

Scores map to clear labels. The rules are fixed and transparent: every point a stock
gets is listed as a reason in the dashboard. They are NOT a forecast: the dashboard
also shows how often each label was followed by a rise in the past (hit rate).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

LABELS = [  # (min score, label, plain-English action)
    (72, "STRONG BUY", "Rules strongly favour buying now"),
    (58, "BUY", "Rules favour buying"),
    (42, "HOLD / WAIT", "Don't buy yet. If you already own it, hold"),
    (28, "AVOID", "Don't buy"),
    (0, "SELL / EXIT", "Don't buy. If you own it, consider exiting"),
]

SWING_HORIZON = 20       # trading days (~1 month) used to measure swing hit rate
LONG_HORIZON = 126       # trading days (~6 months) used to measure long-term hit rate


def label_for(score: float) -> tuple[str, str]:
    for lo, name, action in LABELS:
        if score >= lo:
            return name, action
    return LABELS[-1][1], LABELS[-1][2]


def label_series(scores: pd.Series) -> pd.Series:
    bins = [-1, 28, 42, 58, 72, 101]
    names = ["SELL / EXIT", "AVOID", "HOLD / WAIT", "BUY", "STRONG BUY"]
    return pd.cut(scores, bins=bins, labels=names, right=False).astype(str)


# --------------------------------------------------------------------------- swing
def swing_scores(d: pd.DataFrame, nifty: pd.DataFrame, mom_rank: pd.Series) -> pd.DataFrame:
    """Vectorised swing score for every day (so we can measure past hit rates).

    mom_rank = this stock's 12-month momentum percentile among Nifty 50 stocks (0..1) each day.
    """
    c = d["Close"]
    nifty_close = nifty["Close"].reindex(d.index).ffill()
    nifty_sma50 = nifty_close.rolling(50).mean()
    mr = mom_rank.reindex(d.index)

    comp = pd.DataFrame(index=d.index)
    # Relative momentum: the best-documented edge (max 30)
    comp["momentum"] = np.select([mr >= 0.8, mr >= 0.6, mr >= 0.4], [30, 20, 10], 0)
    # Trend (max 30)
    comp["above_sma50"] = (c > d["SMA50"]) * 10
    comp["ema20_gt_sma50"] = (d["EMA20"] > d["SMA50"]) * 10
    comp["sma50_gt_sma200"] = (d["SMA50"] > d["SMA200"]) * 10
    # Short-term momentum (max 15)
    comp["macd_pos"] = (d["MACDhist"] > 0) * 10
    comp["macd_rising"] = (d["MACDhist"] > d["MACDhist"].shift(1)) * 5
    # RSI: not overbought (max 10)
    r = d["RSI"]
    comp["rsi"] = np.select([(r >= 40) & (r <= 70), r > 75, r < 30], [10, -10, -5], 0)
    # Breakout + volume (max 15)
    comp["breakout"] = (c >= 0.98 * d["High20"]) * 10
    comp["volume"] = ((c > c.shift(1)) & (d["Volume"] > 1.2 * d["VolAvg20"])) * 5
    # Weak market penalty
    comp["weak_market"] = (nifty_close < nifty_sma50) * -10

    total = comp.sum(axis=1).clip(0, 100)
    valid = d["SMA200"].notna() & d["RSI"].notna() & mr.notna()
    comp["score"] = total.where(valid)
    return comp


SWING_REASONS = {
    "above_sma50": ("Price is above its 50-day average (uptrend)", "Price is below its 50-day average"),
    "ema20_gt_sma50": ("Short-term trend (20 EMA) is above the 50-day", "Short-term trend is below the 50-day"),
    "sma50_gt_sma200": ("50-day average is above 200-day (healthy trend)", "50-day average is below 200-day (downtrend)"),
    "macd_pos": ("MACD momentum is positive", "MACD momentum is negative"),
    "macd_rising": ("Momentum is improving", "Momentum is fading"),
    "volume": ("Up move came on above-average volume", None),
    "breakout": ("Trading near its 20-day high (breakout zone)", None),
    "weak_market": (None, "Overall market (Nifty) is below its 50-day average, so risk is higher"),
}


def swing_reasons(row: pd.Series, d_last: pd.Series) -> list[dict]:
    out = []
    for key, (good, bad) in SWING_REASONS.items():
        v = row[key]
        if key == "weak_market":
            if v < 0:
                out.append({"t": bad, "s": -1})
            continue
        if v > 0 and good:
            out.append({"t": good, "s": 1})
        elif v <= 0 and bad:
            out.append({"t": bad, "s": -1})
    out.insert(0, _mom_reason(row["momentum"]))
    r = d_last["RSI"]
    if 40 <= r <= 70:
        out.append({"t": f"RSI {r:.0f}: healthy, not overbought", "s": 1})
    elif r > 75:
        out.append({"t": f"RSI {r:.0f}: overbought, a pullback is common from here", "s": -1})
    elif r > 70:
        out.append({"t": f"RSI {r:.0f}: getting stretched", "s": 0})
    else:
        out.append({"t": f"RSI {r:.0f}: weak / oversold", "s": -1})
    return out


def _mom_reason(pts: float) -> dict:
    if pts >= 30:
        return {"t": "12-month momentum in the top 20% of Nifty 50 stocks", "s": 1}
    if pts >= 20:
        return {"t": "12-month momentum above most Nifty 50 stocks", "s": 1}
    if pts >= 10:
        return {"t": "12-month momentum around the middle of the Nifty 50", "s": 0}
    return {"t": "12-month momentum in the bottom 40% of the Nifty 50", "s": -1}


# ----------------------------------------------------------------------- long term
def long_tech_scores(d: pd.DataFrame, nifty: pd.DataFrame, mom_rank: pd.Series) -> pd.DataFrame:
    """Technical half of the long-term score (max 60), vectorised for history."""
    c = d["Close"]
    mr = mom_rank.reindex(d.index)
    comp = pd.DataFrame(index=d.index)
    comp["momentum"] = np.select([mr >= 0.8, mr >= 0.6, mr >= 0.4], [30, 20, 10], 0)
    comp["above_sma200"] = (c > d["SMA200"]) * 12
    comp["golden"] = (d["SMA50"] > d["SMA200"]) * 10
    comp["sma200_rising"] = (d["SMA200"] > d["SMA200"].shift(20)) * 8
    comp["tech"] = comp.sum(axis=1).clip(0, 60)
    comp.loc[d["SMA200"].isna() | mr.isna(), "tech"] = np.nan
    return comp


LONG_REASONS = {
    "above_sma200": ("Price is above its 200-day average (long-term uptrend)", "Price is below its 200-day average (long-term downtrend)"),
    "golden": ("50-day average above 200-day", "50-day average below 200-day"),
    "sma200_rising": ("200-day average is rising", "200-day average is falling"),
}


def _num(x):
    try:
        x = float(x)
        return None if math.isnan(x) or math.isinf(x) else x
    except (TypeError, ValueError):
        return None


FIN_SECTORS = {"Financial Services"}


def fundamental_score(f: dict, industry: str, sector_pe_median: float | None) -> tuple[float | None, list[dict], dict]:
    """Fundamental half of the long-term score (max 40). Returns (score, reasons, clean metrics)."""
    if not f:
        return None, [{"t": "Fundamental data unavailable: score uses price trend only", "s": 0}], {}
    roe, eg, rg = _num(f.get("returnOnEquity")), _num(f.get("earningsGrowth")), _num(f.get("revenueGrowth"))
    pm, de = _num(f.get("profitMargins")), _num(f.get("debtToEquity"))
    pe = _num(f.get("trailingPE")) or _num(f.get("forwardPE"))
    is_fin = industry in FIN_SECTORS
    s, why = 0.0, []
    if roe is not None:
        if roe > 0.18: s += 10; why.append({"t": f"High return on equity ({roe*100:.0f}%)", "s": 1})
        elif roe > 0.12: s += 6; why.append({"t": f"Decent return on equity ({roe*100:.0f}%)", "s": 1})
        elif roe > 0.08: s += 3; why.append({"t": f"Modest return on equity ({roe*100:.0f}%)", "s": 0})
        else: why.append({"t": f"Weak return on equity ({roe*100:.0f}%)", "s": -1})
    if eg is not None:
        if eg > 0.15: s += 8; why.append({"t": f"Profits growing fast ({eg*100:+.0f}% YoY)", "s": 1})
        elif eg > 0: s += 4; why.append({"t": f"Profits growing ({eg*100:+.0f}% YoY)", "s": 1})
        elif eg < -0.10: s -= 4; why.append({"t": f"Profits falling ({eg*100:+.0f}% YoY)", "s": -1})
        else: why.append({"t": f"Profits flat/slightly down ({eg*100:+.0f}% YoY)", "s": 0})
    if rg is not None:
        if rg > 0.10: s += 6; why.append({"t": f"Revenue growing well ({rg*100:+.0f}% YoY)", "s": 1})
        elif rg > 0: s += 3; why.append({"t": f"Revenue growing ({rg*100:+.0f}% YoY)", "s": 0})
        else: why.append({"t": f"Revenue shrinking ({rg*100:+.0f}% YoY)", "s": -1})
    if pm is not None:
        if pm > 0.15: s += 6; why.append({"t": f"Strong profit margin ({pm*100:.0f}%)", "s": 1})
        elif pm > 0.08: s += 3
        elif pm < 0.03: why.append({"t": f"Thin profit margin ({pm*100:.1f}%)", "s": -1})
    if is_fin:
        s += 3  # debt/equity is not meaningful for banks & NBFCs
    elif de is not None:
        if de < 50: s += 6; why.append({"t": "Low debt", "s": 1})
        elif de < 100: s += 3
        elif de > 200: s -= 4; why.append({"t": f"High debt (D/E {de/100:.1f}x)", "s": -1})
    if pe is not None:
        if pe < 0:
            s -= 4; why.append({"t": "Company is loss-making (negative P/E)", "s": -1})
        elif sector_pe_median:
            ratio = pe / sector_pe_median
            if ratio <= 1.0: s += 4; why.append({"t": f"Valuation reasonable vs peers (P/E {pe:.0f})", "s": 1})
            elif ratio > 1.8: s -= 3; why.append({"t": f"Expensive vs peers (P/E {pe:.0f})", "s": -1})
    metrics = {"pe": pe, "pb": _num(f.get("priceToBook")), "roe": roe, "de": de, "pm": pm,
               "eg": eg, "rg": rg, "dy": _num(f.get("dividendYield")), "mcap": _num(f.get("marketCap"))}
    return max(min(s, 40), 0), why, metrics


def long_reasons(row: pd.Series, d_last: pd.Series) -> list[dict]:
    out = [_mom_reason(row["momentum"])]
    for key, (good, bad) in LONG_REASONS.items():
        out.append({"t": good, "s": 1} if row[key] > 0 else {"t": bad, "s": -1})
    off = d_last["Close"] / d_last["High252"] - 1
    out.append({"t": f"{abs(off)*100:.0f}% below its 52-week high", "s": 0})
    v = d_last["Vol252"]
    if v == v:
        out.append({"t": f"Yearly volatility {v*100:.0f}% ({'high' if v > 0.35 else 'moderate' if v > 0.22 else 'low'} risk)", "s": 0})
    return out


# ------------------------------------------------------------------- hit rates
def forward_returns(close: pd.Series, horizon: int) -> pd.Series:
    return close.shift(-horizon) / close - 1


def hit_rate_table(frames: list[pd.DataFrame], score_col: str, ret_col: str, years: int = 5) -> dict:
    """Pool every stock-day of the last `years` years and measure what happened next per label.

    'xs' = average return minus the average Nifty 50 stock over the same days, i.e. the real edge.
    """
    rows = pd.concat(frames, ignore_index=True)
    cutoff = rows["date"].max() - pd.Timedelta(days=365 * years)
    rows = rows[(rows["date"] >= cutoff)].dropna(subset=[score_col, ret_col])
    rows["xs"] = rows[ret_col] - rows.groupby("date")[ret_col].transform("mean")
    rows["label"] = label_series(rows[score_col])
    out = {}
    for name, g in list(rows.groupby("label")) + [("ALL", rows)]:
        out[name] = {"n": int(len(g)), "win": float((g[ret_col] > 0).mean()),
                     "avg": float(g[ret_col].mean()), "xs": float(g["xs"].mean()),
                     "beat": float((g["xs"] > 0).mean())}
    return out
