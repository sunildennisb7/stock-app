"""Generates the app icons (PNG) with no image libraries. Run once: python scripts/make_icons.py"""
import math, struct, zlib
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "web" / "static"
BG, BG2 = (15, 23, 42), (30, 41, 74)
LINE, DOT = (52, 211, 153), (255, 255, 255)
PTS = [(0.18, 0.70), (0.38, 0.52), (0.52, 0.62), (0.80, 0.30)]


def png(path, size, pad=0.0):
    rows = []
    s = size
    pts = [((pad + x * (1 - 2 * pad)) * s, (pad + y * (1 - 2 * pad)) * s) for x, y in PTS]
    w = s * 0.075 * (1 - 2 * pad)
    for j in range(s):
        row = bytearray([0])
        for i in range(s):
            t = j / s
            r, g, b = (int(BG[k] + (BG2[k] - BG[k]) * t) for k in range(3))
            px, py = i + 0.5, j + 0.5
            d = 1e9
            for (x1, y1), (x2, y2) in zip(pts, pts[1:]):
                dx, dy = x2 - x1, y2 - y1
                u = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)))
                d = min(d, math.hypot(px - (x1 + u * dx), py - (y1 + u * dy)))
            a = max(0.0, min(1.0, w / 2 - d + 0.5))
            r, g, b = (int(c + (L - c) * a) for c, L in zip((r, g, b), LINE))
            ex, ey = pts[-1]
            dd = math.hypot(px - ex, py - ey)
            a = max(0.0, min(1.0, w * 0.95 - dd + 0.5))
            r, g, b = (int(c + (L - c) * a) for c, L in zip((r, g, b), DOT))
            row += bytes((r, g, b))
        rows.append(bytes(row))
    raw = zlib.compress(b"".join(rows), 9)
    def chunk(t, data):
        return struct.pack(">I", len(data)) + t + data + struct.pack(">I", zlib.crc32(t + data) & 0xFFFFFFFF)
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", s, s, 8, 2, 0, 0, 0))
                     + chunk(b"IDAT", raw) + chunk(b"IEND", b""))


OUT.mkdir(parents=True, exist_ok=True)
png(OUT / "icon-192.png", 192)
png(OUT / "icon-512.png", 512)
png(OUT / "icon-maskable-512.png", 512, pad=0.12)
png(OUT / "apple-touch-icon.png", 180)
png(OUT / "favicon-32.png", 32)
print("icons written to", OUT)
