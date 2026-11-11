"""Backtests of the exact same rules the dashboard uses.

Caveat shown in the dashboard: these use TODAY's Nifty 50 members over the past years
(survivorship bias), so real past results would have been somewhat worse.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

COST_PER_SIDE = 0.0012   # ~0.12% per buy/sell: STT, exchange, stamp duty, GST, slippage


def _stats(equity: pd.Series) -> dict:
    equity = equity.dropna()
    yrs = max((equity.index[-1] - equity.index[0]).days / 365.25, 1e-9)
    total = equity.iloc[-1] / equity.iloc[0] - 1
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / yrs) - 1
    dd = (equity / equity.cummax() - 1).min()
    return {"total": float(total), "cagr": float(cagr), "maxdd": float(dd), "years": round(yrs, 1)}


def _curve(equity: pd.Series, bench: pd.Series, ew: pd.Series | None = None) -> dict:
    df = pd.DataFrame({"s": equity, "b": bench, "e": ew if ew is not None else bench}).dropna()
    df = df / df.iloc[0] * 100
    wk = df.resample("W-FRI").last().dropna()
    return {"d": [x.strftime("%Y-%m-%d") for x in wk.index],
            "s": [round(v, 2) for v in wk["s"]], "b": [round(v, 2) for v in wk["b"]],
            "e": [round(v, 2) for v in wk["e"]]}


def _ew_series(frames, dates):
    C = pd.DataFrame({s: f["Close"] for s, f in frames.items()}).reindex(dates).ffill().bfill()
    return (C / C.iloc[0]).mean(axis=1)


def equal_weight_stats(frames: dict[str, pd.DataFrame], nifty: pd.DataFrame, years: int = 5) -> dict:
    """Buy all current Nifty 50 stocks in equal amounts and do nothing: the honest benchmark,
    since it has the same survivorship bias as the strategies."""
    start = nifty.index[-1] - pd.Timedelta(days=int(365.25 * years))
    dates = nifty.index[nifty.index >= start]
    C = pd.DataFrame({s: f["Close"] for s, f in frames.items()}).reindex(dates).ffill().bfill()
    return _stats((C / C.iloc[0]).mean(axis=1))


def swing_backtest(frames: dict[str, pd.DataFrame], nifty: pd.DataFrame, years: int = 5,
                   capital: float = 1_000_000, risk_pct: float = 0.01, max_pos: int = 8,
                   buy_threshold: float = 58, stop_atr: float = 2.0, rr: float = 2.0,
                   max_days: int = 20) -> dict:
    """Enter next day's open when a stock's swing score crosses into BUY.
    Exit at stop (entry signal close - 2 ATR), target (2x risk), or after 20 trading days."""
    start = nifty.index[-1] - pd.Timedelta(days=int(365.25 * years))
    dates = nifty.index[nifty.index >= start]

    O = pd.DataFrame({s: f["Open"] for s, f in frames.items()}).reindex(dates)
    H = pd.DataFrame({s: f["High"] for s, f in frames.items()}).reindex(dates)
    L = pd.DataFrame({s: f["Low"] for s, f in frames.items()}).reindex(dates)
    C = pd.DataFrame({s: f["Close"] for s, f in frames.items()}).reindex(dates).ffill()
    A = pd.DataFrame({s: f["ATR"] for s, f in frames.items()}).reindex(dates)
    S = pd.DataFrame({s: f["swing_score"] for s, f in frames.items()}).reindex(dates)
    fresh = (S >= buy_threshold) & (S.shift(1) < buy_threshold)

    cash, pos, trades, eq = capital, {}, [], []
    pending: list[tuple[str, float, float]] = []
    for i, dt in enumerate(dates):
        # 1) fill yesterday's signals at today's open
        for sym, sig_close, atr in pending:
            if sym in pos or len(pos) >= max_pos:
                continue
            o = O.at[dt, sym]
            if not (o > 0) or not (atr > 0):
                continue
            stop = sig_close - stop_atr * atr
            if o <= stop:
                continue
            risk_per_share = o - stop
            equity_now = cash + sum(p["qty"] * C.at[dt, s] for s, p in pos.items())
            qty = math.floor(min(equity_now * risk_pct / risk_per_share, equity_now * 0.2 / o, cash / (o * (1 + COST_PER_SIDE))))
            if qty <= 0:
                continue
            cash -= qty * o * (1 + COST_PER_SIDE)
            pos[sym] = {"qty": qty, "entry": o, "stop": stop, "target": o + rr * risk_per_share,
                        "date": dt, "i": i}
        pending = []
        # 2) exits during the day
        for sym in list(pos):
            p = pos[sym]
            o, h, l, c = O.at[dt, sym], H.at[dt, sym], L.at[dt, sym], C.at[dt, sym]
            if not (c > 0):
                continue
            px, why = None, None
            if o > 0 and o <= p["stop"]:
                px, why = o, "stop"
            elif l > 0 and l <= p["stop"]:
                px, why = p["stop"], "stop"
            elif o > 0 and o >= p["target"]:
                px, why = o, "target"
            elif h >= p["target"]:
                px, why = p["target"], "target"
            elif i - p["i"] >= max_days:
                px, why = c, "time"
            if px is not None:
                cash += p["qty"] * px * (1 - COST_PER_SIDE)
                ret = px * (1 - COST_PER_SIDE) / (p["entry"] * (1 + COST_PER_SIDE)) - 1
                trades.append({"sym": sym.replace(".NS", ""), "in": p["date"].strftime("%Y-%m-%d"),
                               "out": dt.strftime("%Y-%m-%d"), "entry": round(p["entry"], 2),
                               "exit": round(px, 2), "ret": round(ret, 4), "why": why,
                               "pnl": round(p["qty"] * (px * (1 - COST_PER_SIDE) - p["entry"] * (1 + COST_PER_SIDE)), 0)})
                del pos[sym]
        # 3) today's fresh signals -> tomorrow's orders (strongest first)
        today = fresh.loc[dt]
        cands = [s for s in today.index[today.fillna(False).astype(bool)]]
        cands.sort(key=lambda s: -S.at[dt, s])
        pending = [(s, C.at[dt, s], A.at[dt, s]) for s in cands]
        eq.append(cash + sum(p["qty"] * C.at[dt, s] for s, p in pos.items()))

    equity = pd.Series(eq, index=dates)
    rets = np.array([t["ret"] for t in trades]) if trades else np.array([0.0])
    wins, losses = rets[rets > 0], rets[rets <= 0]
    gross_win = sum(t["pnl"] for t in trades if t["pnl"] > 0)
    gross_loss = -sum(t["pnl"] for t in trades if t["pnl"] <= 0)
    st = _stats(equity)
    st.update({
        "trades": len(trades), "win": float((rets > 0).mean()) if trades else 0.0,
        "avg_win": float(wins.mean()) if len(wins) else 0.0,
        "avg_loss": float(losses.mean()) if len(losses) else 0.0,
        "pf": float(gross_win / gross_loss) if gross_loss > 0 else None,
        "bench": _stats(nifty["Close"].reindex(dates).ffill()),
        "exits": {k: sum(1 for t in trades if t["why"] == k) for k in ("target", "stop", "time")},
    })
    return {"stats": st, "curve": _curve(equity, nifty["Close"].reindex(dates).ffill(), _ew_series(frames, dates)),
            "recent": trades[-40:][::-1],
            "rules": f"Buy at next open when swing score crosses {buy_threshold:.0f}. Stop = {stop_atr:g}x ATR below signal close; "
                     f"target = {rr:g}x risk; exit after {max_days} trading days otherwise. Risk {risk_pct*100:g}% of capital per trade, "
                     f"max {max_pos} positions, ~{COST_PER_SIDE*100:.2f}% costs per side."}


def long_backtest(frames: dict[str, pd.DataFrame], nifty: pd.DataFrame, years: int = 5,
                  top_n: int = 10, min_score: float = 58, rebalance_days: int = 21) -> dict:
    """Every month hold (equal weight) the top 10 stocks by long-term trend score that rate BUY or better."""
    start = nifty.index[-1] - pd.Timedelta(days=int(365.25 * years))
    dates = nifty.index[nifty.index >= start]
    C = pd.DataFrame({s: f["Close"] for s, f in frames.items()}).reindex(dates).ffill()
    S = pd.DataFrame({s: f["long_tech"] * 100 / 60 for s, f in frames.items()}).reindex(dates)

    value, weights, eq, history = 1.0, pd.Series(dtype=float), [], []
    for i, dt in enumerate(dates):
        # weights = fraction of portfolio value in each stock; the rest is cash
        if i > 0 and len(weights):
            r = (C.loc[dt, weights.index] / C.iloc[i - 1][weights.index] - 1).fillna(0)
            port_r = float((weights * r).sum())
            value *= 1 + port_r
            weights = weights * (1 + r) / (1 + port_r)
        if i % rebalance_days == 0:
            sc = S.loc[dt].dropna()
            pick = sc[sc >= min_score].sort_values(ascending=False).head(top_n)
            new_w = pd.Series(1.0 / top_n, index=pick.index)  # unused slots stay in cash
            allidx = weights.index.union(new_w.index)
            turnover = (new_w.reindex(allidx, fill_value=0) - weights.reindex(allidx, fill_value=0)).abs().sum()
            value *= 1 - turnover * COST_PER_SIDE
            weights = new_w
            history.append({"d": dt.strftime("%Y-%m-%d"), "hold": [s.replace(".NS", "") for s in pick.index]})
        eq.append(value)
    equity = pd.Series(eq, index=dates)
    st = _stats(equity)
    st["bench"] = _stats(nifty["Close"].reindex(dates).ffill())
    return {"stats": st, "curve": _curve(equity, nifty["Close"].reindex(dates).ffill(), _ew_series(frames, dates)),
            "history": history[-6:][::-1],
            "rules": f"Every {rebalance_days} trading days, hold the top {top_n} Nifty 50 stocks by long-term trend score "
                     f"(only those rated BUY or better, otherwise cash), equal weight, ~{COST_PER_SIDE*100:.2f}% costs per trade."}
