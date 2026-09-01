"""Conversions between GdkPixbuf, OpenCV/NumPy arrays and files.

Kept separate from the GUI so the pixel plumbing is easy to read and test.
"""

from __future__ import annotations

import cv2
import numpy as np

import gi

gi.require_version("GdkPixbuf", "2.0")
from gi.repository import GdkPixbuf, GLib  # noqa: E402


def pixbuf_to_bgr(pixbuf: GdkPixbuf.Pixbuf) -> np.ndarray:
    """Convert a GdkPixbuf to a contiguous OpenCV BGR array."""
    w = pixbuf.get_width()
    h = pixbuf.get_height()
    stride = pixbuf.get_rowstride()
    n = pixbuf.get_n_channels()
    data = pixbuf.get_pixels()

    arr = np.frombuffer(data, dtype=np.uint8, count=h * stride)
    arr = arr.reshape(h, stride)
    arr = arr[:, : w * n].reshape(h, w, n)

    if n == 4:
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
    elif n == 3:
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    else:  # grayscale, unlikely
        bgr = cv2.cvtColor(arr.reshape(h, w), cv2.COLOR_GRAY2BGR)
    return np.ascontiguousarray(bgr)


def bgra_to_pixbuf(bgra: np.ndarray) -> GdkPixbuf.Pixbuf:
    """Convert an OpenCV BGRA array to a GdkPixbuf (with alpha)."""
    rgba = cv2.cvtColor(bgra, cv2.COLOR_BGRA2RGBA)
    rgba = np.ascontiguousarray(rgba)
    h, w = rgba.shape[:2]
    # GLib.Bytes takes a copy and owns the buffer, so the pixbuf stays valid
    # regardless of the numpy array's lifetime.
    gbytes = GLib.Bytes.new(rgba.tobytes())
    return GdkPixbuf.Pixbuf.new_from_bytes(
        gbytes,
        GdkPixbuf.Colorspace.RGB,
        True,  # has_alpha
        8,  # bits per sample
        w,
        h,
        w * 4,  # rowstride
    )


def bgr_to_pixbuf(bgr: np.ndarray) -> GdkPixbuf.Pixbuf:
    """Convert an OpenCV BGR/BGRA array to an opaque-or-alpha GdkPixbuf."""
    if bgr.ndim == 3 and bgr.shape[2] == 4:
        return bgra_to_pixbuf(bgr)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = np.ascontiguousarray(rgb)
    h, w = rgb.shape[:2]
    gbytes = GLib.Bytes.new(rgb.tobytes())
    return GdkPixbuf.Pixbuf.new_from_bytes(
        gbytes, GdkPixbuf.Colorspace.RGB, False, 8, w, h, w * 3
    )


def save_bgra_png(bgra: np.ndarray, path: str) -> None:
    """Save a BGRA array as a PNG with a transparent background."""
    bgra_to_pixbuf(bgra).savev(path, "png", [], [])
