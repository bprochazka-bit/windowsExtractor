"""Tests for the framework-agnostic detection / extraction logic.

These use a synthetic screenshot (a desktop background with a window drawn on
it) so they run headless without a display server or GTK.
"""

import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from windowextractor import detector  # noqa: E402


def make_screenshot(win=(300, 200, 640, 420)):
    """A 1280x800 'desktop' with a gradient background and one window."""
    h, w = 800, 1280
    img = np.zeros((h, w, 3), np.uint8)
    # Gradient-ish textured background so it is not a flat colour.
    xs = np.linspace(30, 90, w, dtype=np.uint8)
    img[:, :, 0] = xs[None, :]
    img[:, :, 1] = 60
    img[:, :, 2] = np.linspace(120, 40, h, dtype=np.uint8)[:, None]

    x, y, ww, wh = win
    # Drop shadow.
    cv2.rectangle(img, (x + 6, y + 8), (x + ww + 6, y + wh + 8), (10, 10, 10), -1)
    # Window body (light) with a darker title bar.
    cv2.rectangle(img, (x, y), (x + ww, y + wh), (240, 240, 240), -1)
    cv2.rectangle(img, (x, y), (x + ww, y + 36), (200, 120, 60), -1)
    # A crisp 1px border to help edge detection.
    cv2.rectangle(img, (x, y), (x + ww, y + wh), (120, 120, 120), 1)
    # Some interior widgets (should NOT be detected instead of the window).
    cv2.rectangle(img, (x + 20, y + 60), (x + 120, y + 90), (180, 180, 180), -1)
    cv2.circle(img, (x + ww - 20, y + 18), 8, (255, 255, 255), -1)
    return img, win


def test_detect_finds_window_near_actual_bounds():
    img, (x, y, ww, wh) = make_screenshot()
    # Click somewhere inside the window body (below the title bar).
    result = detector.detect_window_at(img, (x + ww // 2, y + wh // 2))
    assert result is not None, "expected a detection"
    rx, ry, rw, rh = result.rect

    # The detected box should be close to the true window bounds (within a
    # small tolerance for shadow/border), and clearly bigger than a widget.
    assert abs(rx - x) <= 12, (rx, x)
    assert abs(ry - y) <= 12, (ry, y)
    assert abs(rw - ww) <= 16, (rw, ww)
    assert abs(rh - wh) <= 16, (rh, wh)


def test_detect_outside_returns_none_for_flatish_area():
    img, win = make_screenshot()
    # Click far in the background corner: no window-sized region contains it
    # except possibly the whole image, which the max-area guard rejects.
    result = detector.detect_window_at(img, (20, 20), min_area_frac=0.02)
    if result is not None:
        # If anything is returned it must not be the window we drew.
        rx, ry, rw, rh = result.rect
        assert not (rx <= win[0] + 5 <= rx + rw)


def test_click_outside_image_returns_none():
    img, _ = make_screenshot()
    assert detector.detect_window_at(img, (-5, -5)) is None
    assert detector.detect_window_at(img, (99999, 99999)) is None


def test_extract_rgba_shapes_and_alpha():
    img, (x, y, ww, wh) = make_screenshot()
    bgra = detector.extract_rgba(img, (x, y, ww, wh))
    assert bgra.shape == (wh, ww, 4)
    # No contour / no radius => fully opaque.
    assert (bgra[:, :, 3] == 255).all()


def test_extract_rounded_corners_makes_corner_transparent():
    img, (x, y, ww, wh) = make_screenshot()
    bgra = detector.extract_rgba(img, (x, y, ww, wh), corner_radius=25)
    # The very corner pixel should be transparent, the centre opaque.
    assert bgra[0, 0, 3] == 0
    assert bgra[wh // 2, ww // 2, 3] == 255


def test_extract_with_contour_masks_outside():
    img, (x, y, ww, wh) = make_screenshot()
    # A diamond contour inside the crop; corners of the bbox must be cut away.
    contour = np.array(
        [
            [x + ww // 2, y],
            [x + ww, y + wh // 2],
            [x + ww // 2, y + wh],
            [x, y + wh // 2],
        ],
        dtype=np.int32,
    ).reshape(-1, 1, 2)
    bgra = detector.extract_rgba(img, (x, y, ww, wh), contour=contour)
    assert bgra[2, 2, 3] == 0  # near bbox corner -> outside diamond
    assert bgra[wh // 2, ww // 2, 3] == 255  # centre -> inside diamond


def test_extract_clamps_out_of_bounds_rect():
    img, _ = make_screenshot()
    h, w = img.shape[:2]
    bgra = detector.extract_rgba(img, (w - 10, h - 10, 500, 500))
    assert bgra.shape[0] <= 10 and bgra.shape[1] <= 10
    assert bgra.shape[2] == 4


if __name__ == "__main__":
    import traceback

    funcs = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in funcs:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception:  # noqa: BLE001
            failed += 1
            print(f"FAIL {fn.__name__}")
            traceback.print_exc()
    print(f"\n{len(funcs) - failed}/{len(funcs)} passed")
    sys.exit(1 if failed else 0)
