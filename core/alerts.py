"""Push notifications (Web Push).

Subscribers come from the PUSH_SUBSCRIBERS secret (GitHub) or .secrets/subscribers.json (local):
    [{"n": "Sunil's iPhone", "s": <PushSubscription JSON>, "w": ["TCS", "INFY", ...]}, ...]

Alert state (yesterday's labels, what was already sent) lives in cache/alert_state.json, which
GitHub Actions carries between runs via actions/cache.
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

from .data import CACHE, ROOT

STATE_PATH = CACHE / "alert_state.json"
SITE = "https://sunildennisb7.github.io/stock-app/"
LORDER = ["STRONG BUY", "BUY", "HOLD / WAIT", "AVOID", "SELL / EXIT"]
IST = ZoneInfo("Asia/Kolkata")


def now_ist() -> datetime:
    return datetime.now(IST)


def load_state() -> dict:
    try:
        return json.loads(STATE_PATH.read_text())
    except Exception:  # noqa: BLE001
        return {}


def save_state(st: dict) -> None:
    STATE_PATH.write_text(json.dumps(st))


def subscribers() -> list[dict]:
    raw = os.environ.get("PUSH_SUBSCRIBERS")
    if not raw:
        p = ROOT / ".secrets" / "subscribers.json"
        raw = p.read_text() if p.exists() else "[]"
    try:
        subs = json.loads(raw)
        return [s for s in subs if isinstance(s, dict) and s.get("s", {}).get("endpoint")]
    except json.JSONDecodeError:
        print("  ! PUSH_SUBSCRIBERS is not valid JSON")
        return []


def _vapid():
    from py_vapid import Vapid
    key = os.environ.get("VAPID_PRIVATE_KEY")
    if not key:
        p = ROOT / ".secrets" / "vapid.json"
        key = json.loads(p.read_text())["private"] if p.exists() else None
    if not key:
        return None
    return Vapid.from_raw(key.strip().encode())


def send(sub: dict, title: str, body: str, url: str = "", tag: str = "daily", log=print) -> bool:
    """Send one notification. Returns False if the subscription is dead (phone unsubscribed)."""
    from pywebpush import WebPushException, webpush
    v = _vapid()
    if v is None:
        log("  ! no VAPID private key; skipping push")
        return True
    data = json.dumps({"title": title, "body": body, "url": SITE + url, "tag": tag})
    try:
        webpush(subscription_info=sub["s"], data=data, vapid_private_key=v,
                vapid_claims={"sub": SITE.rstrip("/")}, ttl=12 * 3600, timeout=20)
        return True
    except WebPushException as e:
        code = getattr(e.response, "status_code", None)
        log(f"  ! push to {sub.get('n', 'device')} failed ({code})")
        return code not in (404, 410)
    except Exception as e:  # noqa: BLE001
        log(f"  ! push to {sub.get('n', 'device')} failed ({type(e).__name__})")
        return True


def _names(lst, k=6):
    names = [x["s"] for x in lst[:k]]
    return ", ".join(names) + (f" +{len(lst) - k} more" if len(lst) > k else "")


# ------------------------------------------------------------------ evening alerts
def evening(payload: dict, log=print) -> None:
    st = load_state()
    date = payload["data_iso"]
    if st.get("evening_sent") == date:
        log("     alerts for this date were already sent")
        return
    items, m = payload["items"], payload["market"]
    prev, prev_regime = st.get("labels") or {}, st.get("regime")
    has_prev = bool(prev)
    cur = {x["s"]: [x["sl"], x["ll"], x["p"], x["st"], x["t"]] for x in items}
    nifty = m["indices"].get("Nifty 50", {})
    stocks = [x for x in items if x["k"] == "s"]
    top_long = sorted(stocks, key=lambda x: -x["ls"])[:5]
    top_swing = sorted(stocks, key=lambda x: -x["ss"])[:5]
    n_buy = sum(1 for x in stocks if x["ll"] <= 1)
    new_sb = [x for x in items if has_prev and x["s"] in prev and
              ((x["ll"] == 0 and prev[x["s"]][1] != 0) or (x["sl"] == 0 and prev[x["s"]][0] != 0))]
    new_sb.sort(key=lambda x: -(x["ls"] + x["ss"]))

    day = datetime.strptime(date, "%Y-%m-%d").strftime("%d %b")
    d1 = nifty.get("d1") or 0
    summary_title = f"{day}: Nifty {nifty.get('last', 0):,.0f} ({d1:+.1%}) · {m['regime']}"
    summary = (f"{n_buy} stocks rated BUY or better (long term). "
               f"Top long-term: {_names(top_long, 5)}. Top swing: {_names(top_swing, 5)}.")
    if new_sb:
        summary += f" New STRONG BUY: {_names(new_sb, 6)}."
    mood_change = has_prev and prev_regime and prev_regime != m["regime"]

    subs, alive = subscribers(), []
    for sub in subs:
        ok = send(sub, summary_title, summary, "#picks", "daily", log)
        if ok and mood_change:
            ok = send(sub, f"Market mood changed: {prev_regime} → {m['regime']}", m["regime_text"], "#market", "mood", log)
        events = []
        for sym in sub.get("w", []):
            c, p = cur.get(sym), prev.get(sym)
            if not c:
                continue
            if p:
                if c[1] != p[1]:
                    events.append(f"{sym} long-term: {LORDER[p[1]]} → {LORDER[c[1]]}")
                if c[0] != p[0] and (c[0] == 0 or p[0] == 0 or c[0] == 4):
                    events.append(f"{sym} swing: {LORDER[p[0]]} → {LORDER[c[0]]}")
                if p[3] and c[2] <= p[3]:
                    events.append(f"{sym} closed ₹{c[2]:,.2f}, below its stop-loss level ₹{p[3]:,.2f}")
                elif p[4] and c[2] >= p[4]:
                    events.append(f"{sym} closed ₹{c[2]:,.2f}, reached its target ₹{p[4]:,.2f}")
        if ok and events:
            ok = send(sub, f"Watchlist: {len(events)} alert{'s' if len(events) > 1 else ''}", " · ".join(events)[:600],
                      "#port", "watch", log)
        if ok:
            alive.append(sub)
    log(f"     sent evening alerts to {len(alive)} device(s)" + (" (first run: no 'new' comparisons yet)" if not has_prev else ""))
    if len(alive) < len(subs):
        log(f"  ! {len(subs) - len(alive)} device(s) have unsubscribed; remove them from PUSH_SUBSCRIBERS")
    st.update({"labels": cur, "regime": m["regime"], "evening_sent": date})
    save_state(st)


# ------------------------------------------------------------------ 1:30 PM buy list
def fetch_live(syms: list[str]) -> tuple[dict[str, float], str | None]:
    """Latest intraday price for each NSE symbol (Yahoo, ~delayed a few minutes). Only today's prices count."""
    import yfinance as yf
    tick = [s + ".NS" for s in syms]
    raw = yf.download(tick, period="1d", interval="5m", group_by="ticker", progress=False,
                      auto_adjust=False, threads=True)
    today = now_ist().date()
    out, last_ts = {}, None
    for s, t in zip(syms, tick):
        try:
            c = raw[t]["Close"].dropna()
        except KeyError:
            continue
        if c.empty:
            continue
        ts = c.index[-1]
        ts = ts.tz_convert(IST) if ts.tzinfo else ts.tz_localize("UTC").tz_convert(IST)
        if ts.date() != today:
            continue
        out[s] = float(c.iloc[-1])
        last_ts = max(last_ts, ts) if last_ts else ts
    return out, (last_ts.strftime("%I:%M %p") if last_ts else None)


def midday(payload: dict, log=print, send_push: bool = True) -> dict:
    today = now_ist().strftime("%Y-%m-%d")
    st = load_state()
    items = payload["items"]
    stocks = [x for x in items if x["k"] == "s"]  # the buy list is stocks only
    sw_c = sorted([x for x in stocks if x["sl"] <= 1], key=lambda x: -x["ss"])[:40]
    lt_c = sorted([x for x in stocks if x["ll"] == 0], key=lambda x: -x["ls"])[:40]
    syms = sorted({x["s"] for x in sw_c + lt_c})
    live, at = fetch_live(syms)
    if not live:
        log("     no intraday prices for today (market holiday?): no buy list")
        res = {"date": today, "closed": True}
        st["midday"] = res
        save_state(st)
        return res
    swing, longp = [], []
    for x in sw_c:
        px = live.get(x["s"])
        if px is None or not x["st"] or px <= x["st"]:
            continue
        move = px / x["e"] - 1
        if -0.03 <= move <= 0.03:
            risk = px - x["st"]
            swing.append({"s": x["s"], "live": round(px, 2), "move": round(move, 4), "stop": x["st"],
                          "target": round(px + 2 * risk, 2), "score": x["ss"], "lab": LORDER[x["sl"]]})
    for x in lt_c:
        px = live.get(x["s"])
        if px is None or not x.get("zh") or not x.get("rv"):
            continue
        if x["rv"] < px <= x["zh"] * 1.05:
            longp.append({"s": x["s"], "live": round(px, 2), "move": round(px / x["p"] - 1, 4),
                          "zl": x["zl"], "zh": x["zh"], "rv": x["rv"], "score": x["ls"]})
    swing, longp = swing[:8], longp[:8]
    res = {"date": today, "at": at, "swing": swing, "long": longp, "checked": len(syms)}
    log(f"     buy list at {at}: {len(swing)} swing, {len(longp)} long-term (checked {len(syms)})")
    if send_push and st.get("midday_sent") != today:
        if swing or longp:
            title = f"Buy list for today ({at} prices)"
            parts = []
            if swing:
                parts.append("Swing: " + ", ".join(f"{p['s']} ₹{p['live']:,.0f}" for p in swing[:5]))
            if longp:
                parts.append("Long-term: " + ", ".join(f"{p['s']} ₹{p['live']:,.0f}" for p in longp[:5]))
            body = " · ".join(parts) + ". Tap for stop-loss and targets."
        else:
            title, body = "No buys today", "None of today's buy signals is still near its signal price. Better to wait."
        subs = subscribers()
        n = sum(1 for sub in subs if send(sub, title, body, "#today", "buylist", log))
        log(f"     sent buy list to {n} device(s)")
        st["midday_sent"] = today
    st["midday"] = res
    save_state(st)
    return res


def drop_today(prices: dict[str, pd.DataFrame]) -> None:
    """During market hours Yahoo includes today's unfinished candle; signals must use completed days only."""
    today = pd.Timestamp(now_ist().date())
    for k, df in prices.items():
        if len(df) and df.index[-1] >= today:
            prices[k] = df[df.index < today]
