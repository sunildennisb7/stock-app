#!/usr/bin/env python3
"""Market Analyzer (Nifty 500, ETFs, mutual funds).

Usage:
    .venv/bin/python run.py              # analyse, then open the app at http://localhost:8765
    .venv/bin/python run.py --refresh    # force fresh price download
    .venv/bin/python run.py --no-open    # just build output/ (used by GitHub Actions)
    .venv/bin/python run.py --no-funds   # skip the mutual fund ranking (faster)
"""
import argparse
import functools
import http.server
import sys
import time
import webbrowser

from core.report import OUT_DIR, build


def main() -> int:
    ap = argparse.ArgumentParser(description="Indian market analyzer")
    ap.add_argument("--refresh", action="store_true", help="ignore cache and download fresh prices")
    ap.add_argument("--no-open", action="store_true", help="build only, don't start the local viewer")
    ap.add_argument("--no-funds", action="store_true", help="skip mutual fund rankings")
    ap.add_argument("--notify", action="store_true", help="send evening push alerts (used by the scheduled run)")
    ap.add_argument("--midday", action="store_true", help="1:30 PM run: today's buy list from live prices + push")
    args = ap.parse_args()
    t = time.time()
    mode = "midday" if args.midday else "evening" if args.notify else "plain"
    build(refresh=args.refresh, with_funds=not args.no_funds, mode=mode)
    print(f"\nDone in {time.time() - t:.0f}s. App files are in {OUT_DIR}")
    if args.no_open:
        return 0
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(OUT_DIR))
    handler.log_message = lambda *a, **k: None
    with http.server.ThreadingHTTPServer(("127.0.0.1", 8765), handler) as srv:
        print("Open http://localhost:8765  (press Ctrl+C to stop)")
        webbrowser.open("http://localhost:8765")
        try:
            srv.serve_forever()
        except KeyboardInterrupt:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
