"""Technical indicators (pure pandas, no extra libraries)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def rsi(s: pd.Series, n: int = 14) -> pd.Series:
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = up / dn.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    return out.where(dn != 0, 100.0)


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    pc = df["Close"].shift(1)
    tr = pd.concat([df["High"] - df["Low"], (df["High"] - pc).abs(), (df["Low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of an OHLCV frame with all indicators used by the signal rules."""
    d = df.copy()
    c = d["Close"]
    d["SMA20"] = sma(c, 20)
    d["EMA20"] = ema(c, 20)
    d["SMA50"] = sma(c, 50)
    d["SMA200"] = sma(c, 200)
    d["RSI"] = rsi(c)
    m = ema(c, 12) - ema(c, 26)
    d["MACD"] = m
    d["MACDsig"] = ema(m, 9)
    d["MACDhist"] = m - d["MACDsig"]
    d["ATR"] = atr(d)
    d["VolAvg20"] = d["Volume"].rolling(20).mean()
    d["High20"] = d["High"].rolling(20).max()
    d["Low10"] = d["Low"].rolling(10).min()
    d["High252"] = d["High"].rolling(252, min_periods=120).max()
    d["Low252"] = d["Low"].rolling(252, min_periods=120).min()
    d["Ret63"] = c.pct_change(63)
    d["Ret126"] = c.pct_change(126)
    d["Ret252"] = c.pct_change(252)
    d["Vol252"] = c.pct_change().rolling(252, min_periods=120).std() * np.sqrt(252)
    return d
