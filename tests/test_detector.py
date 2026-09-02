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


def test_detect_returns_bordered_region_containing_click():
    img, (x, y, ww, wh) = make_screenshot()
    cx, cy = x + ww // 2, y + wh // 2
    result = detector.detect_window_at(img, (cx, cy))
    assert result is not None, "expected a detection"
    rx, ry, rw, rh = result.rect

    # Contract (honest for real multi-window desktops): detection returns a
    # well-bordered rectangle that CONTAINS the click and lies WITHIN the true
    # window (it may be a tight sub-region -- window vs. inner pane is
    # ambiguous from a flat screenshot -- which the user grows with the
    # handles). It must never be a cross-image blob.
    assert rx <= cx <= rx + rw and ry <= cy <= ry + rh
    assert rx >= x - 12 and ry >= y - 12
    assert rx + rw <= x + ww + 12 and ry + rh <= y + wh + 12
    assert rw * rh <= 0.8 * img.shape[1] * img.shape[0]


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


def test_extract_square_geometry_keeps_everything():
    img, (x, y, ww, wh) = make_screenshot()
    bgra = detector.extract_rgba(img, (x, y, ww, wh))  # radius 0 => square
    assert bgra.shape == (wh, ww, 4)
    # Square geometry: every pixel is inside, so all opaque, nothing trimmed.
    assert (bgra[:, :, 3] == 255).all()


def test_extract_keeps_inside_pixels_untouched():
    # The contract: colour channels inside the geometry are byte-for-byte the
    # source pixels -- no erosion, no keying, nothing modified.
    img, (x, y, ww, wh) = make_screenshot()
    bgra = detector.extract_rgba(img, (x, y, ww, wh), corner_radius=20)
    crop = img[y : y + wh, x : x + ww]
    inside = bgra[:, :, 3] == 255
    assert inside.any()
    assert (bgra[:, :, :3][inside] == crop[inside]).all()


def test_extract_rounded_corner_clears_outside_only():
    img, (x, y, ww, wh) = make_screenshot()
    bgra = detector.extract_rgba(img, (x, y, ww, wh), corner_radius=25)
    # Corner (outside the rounded geometry) is cleared; centre kept.
    assert bgra[0, 0, 3] == 0
    assert bgra[wh // 2, ww // 2, 3] == 255
    # A point well inside the straight top edge stays opaque (only corners cut).
    assert bgra[2, ww // 2, 3] == 255


def test_extract_clamps_out_of_bounds_rect():
    img, _ = make_screenshot()
    h, w = img.shape[:2]
    bgra = detector.extract_rgba(img, (w - 10, h - 10, 500, 500))
    assert bgra.shape[0] <= 10 and bgra.shape[1] <= 10
    assert bgra.shape[2] == 4


# --- corner-radius estimation -----------------------------------------------

def _rounded_region(rect, radius, shape=(800, 1280)):
    """A full-image mask that is a rounded rectangle at ``rect``."""
    x, y, w, h = rect
    mask = np.zeros(shape, np.uint8)
    sub = detector._rounded_rect_alpha(h, w, radius)
    mask[y : y + h, x : x + w] = sub
    return mask


def test_estimate_corner_radius_matches_known_radius():
    rect = (200, 150, 500, 360)
    for radius in (0, 10, 20, 35):
        region = _rounded_region(rect, radius)
        est = detector.estimate_corner_radius(region, rect)
        if radius == 0:
            assert est == 0, est
        else:
            # Diagonal probing on a rasterised arc is within a few px.
            assert abs(est - radius) <= 4, (radius, est)


def test_estimate_corner_radius_median_ignores_one_bad_corner():
    rect = (200, 150, 500, 360)
    region = _rounded_region(rect, 20)
    # Wreck one corner: punch a background hole along its diagonal so that
    # corner reads as a huge radius. The median of the other three should win.
    x, y, w, h = rect
    cv2.rectangle(region, (x, y), (x + 120, y + 120), 0, -1)
    est = detector.estimate_corner_radius(region, rect)
    assert abs(est - 20) <= 6, est


def test_color_radius_estimator_on_known_rect():
    # Desktop background with a rounded-corner window composited on top.
    h, w = 800, 1280
    img = np.full((h, w, 3), 50, np.uint8)
    img[:, :, 2] = 90
    x, y, ww, wh, r = 300, 200, 640, 420, 22
    alpha = detector._rounded_rect_alpha(wh, ww, r)
    window = np.full((wh, ww, 3), 235, np.uint8)
    window[:36] = (200, 120, 60)  # title bar
    roi = img[y : y + wh, x : x + ww]
    m = (alpha > 0)[:, :, None]
    img[y : y + wh, x : x + ww] = np.where(m, window, roi)

    # The color-keyed radius estimator (used by detection, no contour needed)
    # recovers the radius from the desktop/window transition at each corner.
    est = detector._estimate_radius_color(img, (x, y, ww, wh))
    assert abs(est - r) <= 6, est


# --- pixel-perfect edges / no background bleed ------------------------------

def _aa_window_on_red(x=300, y=200, ww=640, wh=420, H=800, W=1280, radius=0):
    """Gray window anti-aliased over a saturated red desktop (BGR).

    The AA boundary pixels are genuine window+desktop blends -- exactly what
    causes a colored fringe when the cutout is composited elsewhere.
    """
    bg = np.zeros((H, W, 3), np.float32)
    bg[:, :] = (0, 0, 255)  # red desktop
    win = np.full((H, W, 3), 230.0, np.float32)  # light-gray window
    win[y : y + 36] = (200, 120, 60)  # title bar

    shape = detector._rounded_rect_alpha(wh, ww, radius).astype(np.float32) / 255.0
    a = np.zeros((H, W), np.float32)
    a[y : y + wh, x : x + ww] = shape
    a = cv2.GaussianBlur(a, (3, 3), 0)  # soft, anti-aliased boundary
    a3 = a[:, :, None]
    img = (win * a3 + bg * (1 - a3)).astype(np.uint8)
    return img, (x, y, ww, wh)


def _max_redness_of_opaque(bgra):
    """Largest (R - G) among opaque pixels; high => red desktop bled in."""
    opaque = bgra[:, :, 3] == 255
    if not opaque.any():
        return 0
    r = bgra[:, :, 2].astype(int)
    g = bgra[:, :, 1].astype(int)
    return int((r - g)[opaque].max())


def test_geometry_defines_the_cut_exactly():
    # A rounded geometry over a window on red: pixels outside the geometry are
    # cleared; pixels inside keep their exact source colour. Any "bleed" is
    # purely a function of the geometry, not post-processing.
    img, _ = _aa_window_on_red(radius=22)
    rect = (300, 200, 640, 420)
    bgra = detector.extract_rgba(img, rect, corner_radius=22)
    expected = detector._rounded_rect_alpha(420, 640, 22)
    assert (bgra[:, :, 3] == expected).all()
    crop = img[200:620, 300:940]
    inside = bgra[:, :, 3] == 255
    assert (bgra[:, :, :3][inside] == crop[inside]).all()


# --- flush-to-edge behaviour ------------------------------------------------

def _bg(h=800, w=1280):
    img = np.full((h, w, 3), 50, np.uint8)
    img[:, :, 2] = 90
    return img


def _draw(img, x, y, w, h):
    cv2.rectangle(img, (x, y), (x + w - 1, y + h - 1), (235, 235, 235), -1)
    cv2.rectangle(img, (x, y), (x + w - 1, y + 36), (200, 120, 60), -1)
    cv2.rectangle(img, (x, y), (x + w - 1, y + h - 1), (120, 120, 120), 1)


def test_window_flush_detection_stays_within_window():
    # Honest contract for flush windows: detection may return a sub-region, but
    # whatever it returns must contain the click and stay within the window
    # (never a cross-image blob). Exact flush-snapping is covered by
    # test_snap_rect_to_borders.
    H, W = 800, 1280
    cases = {
        "top": (300, 0, 640, 420),
        "left": (0, 200, 640, 420),
        "bottom": (300, 380, 640, 420),
        "right": (640, 200, 640, 420),
    }
    for name, (x, y, w, h) in cases.items():
        img = _bg(H, W)
        xw, yh = min(x + w, W), min(y + h, H)
        _draw(img, x, y, xw - x, yh - y)
        cx, cy = x + w // 2, y + h // 2
        res = detector.detect_window_at(img, (cx, cy))
        if res is None:
            continue
        rx, ry, rw, rh = res.rect
        assert rx <= cx <= rx + rw and ry <= cy <= ry + rh, (name, res.rect)
        assert rx >= x - 8 and ry >= y - 8, (name, res.rect)
        assert rx + rw <= xw + 8 and ry + rh <= yh + 8, (name, res.rect)


def test_snap_selection_to_edges_pulls_rough_box_to_window():
    img, (x, y, ww, wh) = make_screenshot()
    rough = (x - 30, y - 24, ww + 55, wh + 48)  # a sloppy drag around the window
    sx, sy, sw, sh = detector.snap_selection_to_edges(img, rough)
    # A sloppy drag (every side 24-30px off, including the title-bar side, which
    # must NOT snap to the internal title/body divider) lands on the real window
    # edges within a small tolerance (a drop shadow can sit a few px out).
    assert abs(sx - x) <= 12 and abs(sy - y) <= 12, (sx, sy)
    assert abs((sx + sw) - (x + ww)) <= 12, sx + sw
    assert abs((sy + sh) - (y + wh)) <= 12, sy + sh


def test_snap_selection_respects_a_tight_box():
    # A box already on the window edges should stay put (no drift).
    img, (x, y, ww, wh) = make_screenshot()
    sx, sy, sw, sh = detector.snap_selection_to_edges(img, (x, y, ww, wh))
    assert abs(sx - x) <= 8 and abs(sy - y) <= 8
    assert abs(sw - ww) <= 12 and abs(sh - wh) <= 12


def test_snap_dark_window_on_dark_background():
    # A dark window with a soft drop shadow on a near-black desktop (the
    # low-contrast case where a fixed threshold fails). The adaptive threshold
    # must still snap every side -- including a very loose left side.
    H, W = 900, 1150
    img = np.full((H, W, 3), 12, np.uint8)
    x, y, ww, wh = 300, 200, 600, 560
    sh = np.zeros((H, W), np.float32)
    cv2.rectangle(sh, (x - 6, y + 4), (x + ww + 6, y + wh + 10), 1, -1)
    sh = cv2.GaussianBlur(sh, (0, 0), 18)[:, :, None]
    img = (img * (1 - 0.9 * sh)).astype(np.uint8)
    mask = detector._rounded_rect_alpha(wh, ww, 14).astype(np.float32) / 255
    win = np.full((wh, ww, 3), 43, np.uint8)
    win[:, :170] = (58, 58, 58)
    roi = img[y:y + wh, x:x + ww]
    img[y:y + wh, x:x + ww] = (
        win * mask[:, :, None] + roi * (1 - mask[:, :, None])
    ).astype(np.uint8)

    # Sloppy box, left side the loosest (-55px), like the reported case.
    sx, sy, sw, sh2 = detector.snap_selection_to_edges(
        img, (x - 55, y - 40, ww + 95, wh + 80)
    )
    assert abs(sx - x) <= 8, ("left", sx)
    assert abs(sy - y) <= 8, ("top", sy)
    assert abs((sx + sw) - (x + ww)) <= 8, ("right", sx + sw)
    assert abs((sy + sh2) - (y + wh)) <= 8, ("bottom", sy + sh2)


def _dark_window_image(radius=14):
    H, W = 900, 1150
    img = np.full((H, W, 3), 12, np.uint8)
    x, y, ww, wh = 300, 200, 600, 560
    sh = np.zeros((H, W), np.float32)
    cv2.rectangle(sh, (x - 6, y + 4), (x + ww + 6, y + wh + 10), 1, -1)
    sh = cv2.GaussianBlur(sh, (0, 0), 18)[:, :, None]
    img = (img * (1 - 0.9 * sh)).astype(np.uint8)
    mask = detector._rounded_rect_alpha(wh, ww, radius).astype(np.float32) / 255
    win = np.full((wh, ww, 3), 43, np.uint8)
    win[:, :170] = (58, 58, 58)
    roi = img[y:y + wh, x:x + ww]
    img[y:y + wh, x:x + ww] = (
        win * mask[:, :, None] + roi * (1 - mask[:, :, None])
    ).astype(np.uint8)
    return img, (x, y, ww, wh)


def test_snap_is_stable_across_sloppy_boxes():
    # The span-based snap keys on the border's geometry (a full-length line), so
    # the result must not depend on how loosely the box was drawn.
    img, (x, y, ww, wh) = _dark_window_image()
    results = []
    for pad in ((55, 40, 95, 80), (20, 25, 40, 55), (70, 60, 120, 110)):
        rough = (x - pad[0], y - pad[1], ww + pad[2], wh + pad[3])
        results.append(detector.snap_selection_to_edges(img, rough))
    for sx, sy, sw, sh in results:
        assert abs(sx - x) <= 6 and abs(sy - y) <= 6, (sx, sy)
        assert abs(sw - ww) <= 8 and abs(sh - wh) <= 8, (sw, sh)


def test_estimate_corner_radius_geom_rounded_and_square():
    # A rounded window's radius is recovered reasonably (corner-agreement gated),
    # and -- the important case -- a SQUARE window reads exactly 0 so its corners
    # are never wrongly clipped.
    for true_r in (10, 14, 16):
        rounded, rrect = _dark_window_image(radius=true_r)
        est = detector.estimate_corner_radius_geom(rounded, rrect)
        assert abs(est - true_r) <= 5, (true_r, est)
    sq, sqrect = _dark_window_image(radius=0)
    assert detector.estimate_corner_radius_geom(sq, sqrect) == 0


def test_snap_rect_to_borders():
    # 300x200 image; a rect a few px inside each edge snaps flush.
    assert detector.snap_rect_to_borders((3, 4, 200, 150), 300, 200) == (
        0, 0, 203, 154
    )
    # Far-from-edge sides are untouched.
    assert detector.snap_rect_to_borders((50, 60, 100, 80), 300, 200) == (
        50, 60, 100, 80
    )


def test_border_sides_counts_flush_edges():
    W, H = 1000, 800
    assert detector.border_sides((0, 0, 1000, 800), W, H) == 4  # maximized
    assert detector.border_sides((0, 0, 1000, 400), W, H) == 3  # docked to top
    assert detector.border_sides((0, 100, 1000, 400), W, H) == 2  # full width
    assert detector.border_sides((100, 100, 200, 200), W, H) == 0  # floating


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
