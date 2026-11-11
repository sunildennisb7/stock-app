#!/usr/bin/env python3
"""Nifty 50 Market Analyzer.

Usage:
    .venv/bin/python run.py              # analyse and open the dashboard
    .venv/bin/python run.py --refresh    # force fresh data download
    .venv/bin/python run.py --no-open    # don't open the browser
"""
import argparse
import sys
import time
import webbrowser

from core.report import build


def main() -> int:
    ap = argparse.ArgumentParser(description="Nifty 50 market analyzer")
    ap.add_argument("--refresh", action="store_true", help="ignore cache and download fresh data")
    ap.add_argument("--no-open", action="store_true", help="don't open the dashboard in a browser")
    args = ap.parse_args()
    t = time.time()
    out = build(refresh=args.refresh)
    print(f"\nDone in {time.time() - t:.0f}s. Dashboard: {out}")
    if not args.no_open:
        webbrowser.open(out.as_uri())
    return 0


if __name__ == "__main__":
    sys.exit(main())
