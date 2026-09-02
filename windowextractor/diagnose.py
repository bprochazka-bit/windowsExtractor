"""Detection diagnostic / tuning helper.

Run the detector on a real screenshot from the command line and see exactly
what it does at a given click point -- without the GUI. It prints the detected
rectangle and writes an overlay image so the result can be eyeballed (and shared
back for tuning).

Usage:
    python3 -m windowextractor.diagnose SHOT.png X Y [--out overlay.png]

X Y is the click location in image pixels (top-left origin). If omitted, the
image centre is used.
"""

from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np

from . import detector


def run(path, x=None, y=None, out=None):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"error: could not read image: {path}", file=sys.stderr)
        return 2
    h, w = img.shape[:2]
    if x is None or y is None:
        x, y = w // 2, h // 2
    x, y = int(x), int(y)
    print(f"image: {w}x{h}   click: ({x}, {y})")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    total = float(w * h)
    rect_l = detector._detect_by_lines(gray, (x, y))
    rect_c = detector._detect_by_contours(gray, (x, y), 0.01 * total, 0.995 * total)
    print(f"  line-based   : {rect_l}")
    print(f"  contour-based: {rect_c}")

    result = detector.detect_window_at(img, (x, y))
    if result is None:
        print("  final        : None (no window detected -- use Select mode)")
    else:
        print(f"  final        : rect={result.rect} corner_radius={result.corner_radius}")

    overlay = img.copy()
    if rect_l:
        rx, ry, rw, rh = rect_l
        cv2.rectangle(overlay, (rx, ry), (rx + rw, ry + rh), (0, 200, 255), 2)  # amber
    if rect_c:
        rx, ry, rw, rh = rect_c
        cv2.rectangle(overlay, (rx, ry), (rx + rw, ry + rh), (255, 150, 0), 2)  # blue
    if result is not None:
        rx, ry, rw, rh = result.rect
        cv2.rectangle(overlay, (rx, ry), (rx + rw, ry + rh), (0, 230, 0), 3)  # green
    cv2.drawMarker(overlay, (x, y), (0, 0, 255), cv2.MARKER_CROSS, 30, 3)

    if out is None:
        base, _ext = os.path.splitext(path)
        out = base + ".detect.png"
    cv2.imwrite(out, overlay)
    print(f"  overlay saved: {out}")
    print("  legend: green=final  amber=line  blue=contour  red cross=click")
    return 0


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    ap = argparse.ArgumentParser(prog="windowextractor.diagnose")
    ap.add_argument("image", help="path to a screenshot (PNG/JPG/...)")
    ap.add_argument("x", nargs="?", type=int, help="click X (default: centre)")
    ap.add_argument("y", nargs="?", type=int, help="click Y (default: centre)")
    ap.add_argument("--out", help="overlay output path")
    args = ap.parse_args(argv)
    return run(args.image, args.x, args.y, args.out)


if __name__ == "__main__":
    sys.exit(main())
