"""Window boundary detection from a screenshot.

The core idea: the user clicks a point that is inside an application window.
A screenshot window is (almost always) a large, roughly-rectangular region
separated from the desktop background and from neighbouring windows by strong
straight edges (its border, title bar, or drop shadow). We find candidate
rectangular regions with edge + contour detection and pick the smallest one
that is still "window sized" and contains the clicked point. That heuristic
ignores tiny UI widgets (buttons, icons) while avoiding grabbing the whole
screen, and the result can always be fine-tuned by hand in the GUI.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass
class Detection:
    """Result of a window detection.

    Attributes:
        rect: (x, y, w, h) bounding box in image pixel coordinates. This is the
            window modelled as a plain rectangle (its straight top/bottom/left/
            right edges).
        corner_radius: estimated radius, in pixels, of the window's rounded
            corners -- measured from the four corners and reduced to a single
            median value that is applied symmetrically to all of them. ``0``
            means square corners.
    """

    rect: tuple[int, int, int, int]
    corner_radius: int = 0


def snap_rect_to_borders(rect, img_w, img_h, tol=6):
    """Snap any rect side lying within ``tol`` px of an image edge to the edge.

    A window flush against the screen edge has no gradient there, so its
    detected border can land a few pixels inside the image (or on the shadow),
    leaving a thin sliver of desktop. Snapping cuts the window exactly at the
    image boundary on that side.
    """
    x, y, w, h = rect
    x2, y2 = x + w, y + h
    if x <= tol:
        x = 0
    if y <= tol:
        y = 0
    if img_w - x2 <= tol:
        x2 = img_w
    if img_h - y2 <= tol:
        y2 = img_h
    return (x, y, x2 - x, y2 - y)


def border_sides(rect, img_w, img_h, tol=2):
    """Return how many of the rect's 4 sides sit on the image border.

    Used as the "this looks maximized / full-screen" signal: a window touching
    3+ image edges probably fills the screen, where edge detection cannot find
    an outer boundary and selecting the whole image is the right answer.
    """
    x, y, w, h = rect
    return sum(
        (
            x <= tol,
            y <= tol,
            img_w - (x + w) <= tol,
            img_h - (y + h) <= tol,
        )
    )


def _clean_edge(band_region, bg, interior, first):
    """Locate the boundary between background and window in ``band_region``.

    ``band_region`` is oriented so the scan runs along axis 0 from OUTSIDE
    (background) toward INSIDE (window): rows for a horizontal edge, columns
    (transposed in) for a vertical one, with sampled lines along axis 1.

    A pixel counts as window only once its colour is most of the way from the
    local background to the window interior -- so anti-aliased blend pixels
    (roughly halfway) are excluded and never end up opaque. Returns the median
    boundary offset along axis 0, or ``None`` if the contrast is too low to key
    reliably (e.g. a window edge the same colour as the desktop -- where there
    is no visible fringe to remove anyway).
    """
    contrast = float(np.linalg.norm(bg - interior))
    if contrast < 25:
        return None
    thresh = 0.6 * contrast
    dist = np.linalg.norm(band_region.astype(np.float32) - bg, axis=2)
    fg = dist >= thresh  # True where clearly window, not background/blend
    cols_with_fg = fg.any(axis=0)
    if not cols_with_fg.any():
        return None
    if first:
        # First window pixel scanning inward from the background side.
        idx = np.argmax(fg, axis=0)
    else:
        # Last window pixel (scan reversed, the window is on the outside end).
        idx = (fg.shape[0] - 1) - np.argmax(fg[::-1], axis=0)
    return float(np.median(idx[cols_with_fg]))


def refine_rect_edges(image_bgr, rect, band=7, inner_frac=0.2):
    """Move each straight side of ``rect`` onto the first uncontaminated pixel.

    Detection runs on a *dilated* edge map, so the bounding box lands a few
    pixels off and its exact position varies per side. To make the cut pixel
    accurate (and never sitting on an anti-aliased blend of window + desktop),
    each side is keyed against its own local background: within a small band we
    find the transition from desktop colour to full window colour and put the
    side on the first fully-window pixel. Sides on the image edge, and any side
    whose contrast is too low to key, are left as they are.

    Only the straight middle of each side is sampled (``inner_frac`` trims the
    ends) so rounded corners and corner widgets do not pull an edge inward.
    """
    img = image_bgr[:, :, :3].astype(np.float32)
    H, W = img.shape[:2]
    x, y, w, h = rect
    left, top, right, bottom = x, y, x + w - 1, y + h - 1

    cx0 = min(W - 1, max(0, x + int(w * inner_frac)))
    cx1 = min(W, max(cx0 + 1, x + int(w * (1 - inner_frac))))
    cy0 = min(H - 1, max(0, y + int(h * inner_frac)))
    cy1 = min(H, max(cy0 + 1, y + int(h * (1 - inner_frac))))

    def med(strip):
        return np.median(strip.reshape(-1, 3), axis=0)

    # -- top --
    if top > 0:
        lo, hi = max(0, top - band), min(H, top + band + 1)
        if hi - lo >= 4:
            region = img[lo:hi, cx0:cx1]
            off = _clean_edge(region, med(img[lo:lo + 2, cx0:cx1]),
                              med(img[hi - 2:hi, cx0:cx1]), first=True)
            if off is not None:
                top = lo + int(round(off))
    # -- bottom --
    if bottom < H - 1:
        lo, hi = max(0, bottom - band), min(H, bottom + band + 1)
        if hi - lo >= 4:
            region = img[lo:hi, cx0:cx1]
            off = _clean_edge(region, med(img[hi - 2:hi, cx0:cx1]),
                              med(img[lo:lo + 2, cx0:cx1]), first=False)
            if off is not None:
                bottom = lo + int(round(off))
    # -- left -- (transpose so the scan runs along axis 0)
    if left > 0:
        lo, hi = max(0, left - band), min(W, left + band + 1)
        if hi - lo >= 4:
            region = np.transpose(img[cy0:cy1, lo:hi], (1, 0, 2))
            off = _clean_edge(region, med(img[cy0:cy1, lo:lo + 2]),
                              med(img[cy0:cy1, hi - 2:hi]), first=True)
            if off is not None:
                left = lo + int(round(off))
    # -- right --
    if right < W - 1:
        lo, hi = max(0, right - band), min(W, right + band + 1)
        if hi - lo >= 4:
            region = np.transpose(img[cy0:cy1, lo:hi], (1, 0, 2))
            off = _clean_edge(region, med(img[cy0:cy1, hi - 2:hi]),
                              med(img[cy0:cy1, lo:lo + 2]), first=False)
            if off is not None:
                right = lo + int(round(off))

    if right <= left or bottom <= top:
        return rect  # refinement collapsed; keep the original
    return (left, top, right - left + 1, bottom - top + 1)


def _clamp_rect(rect, w, h):
    x, y, rw, rh = rect
    x = max(0, min(int(x), w - 1))
    y = max(0, min(int(y), h - 1))
    rw = max(1, min(int(rw), w - x))
    rh = max(1, min(int(rh), h - y))
    return (x, y, rw, rh)


def _merge_collinear(lines, coord_tol=6, gap=25):
    """Merge line segments that lie on the same row/column into one.

    Each entry is ``(coord, a0, a1)`` -- the perpendicular position and the
    span. Hough returns a long window border as several pieces; merging them
    lets a broken border read as the single long line it really is.
    """
    merged = []
    for coord, a0, a1 in sorted(lines):
        for m in merged:
            if abs(m[0] - coord) <= coord_tol and not (a1 < m[1] - gap or a0 > m[2] + gap):
                m[0] = (m[0] + coord) / 2.0
                m[1] = min(m[1], a0)
                m[2] = max(m[2], a1)
                break
        else:
            merged.append([float(coord), float(a0), float(a1)])
    return merged


def _side_support(edges, orient, coord, lo, hi, tol=2):
    """Fraction of positions along a segment that have an edge pixel.

    Tolerates gaps: a soft or partly-occluded border still scores high as long
    as edge pixels appear along most of its length (within +/-``tol``).
    """
    H, W = edges.shape[:2]
    coord = int(coord)
    if hi - lo < 2:
        return 0.0
    if orient == "h":
        r0, r1 = max(0, coord - tol), min(H, coord + tol + 1)
        seg = edges[r0:r1, max(0, lo):min(W, hi)]
        present = seg.any(axis=0)
    else:
        c0, c1 = max(0, coord - tol), min(W, coord + tol + 1)
        seg = edges[max(0, lo):min(H, hi), c0:c1]
        present = seg.any(axis=1)
    return float(present.mean()) if present.size else 0.0


def _detect_by_lines(gray, point, min_len_frac=0.08, align_tol=12,
                     min_support=0.32):
    """Find the window as the largest rectangle of long, axis-aligned edges.

    Photographic desktops (a wallpaper photo, a busy background) defeat
    contour-closure detection: they generate dense edges but almost no long
    straight lines. A window, by contrast, is a big axis-aligned rectangle of
    them. We collect candidate horizontal/vertical border positions from long
    Hough segments (after local contrast equalisation, so dark-mode borders on a
    dark background still register), then score every rectangle around the click
    by how much of each of its four sides is actually backed by edge pixels.
    Verifying by edge *support* rather than one continuous segment tolerates
    soft or broken borders; internal panels form smaller rectangles and lose to
    the window's outer border.
    """
    H, W = gray.shape[:2]
    px, py = point
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    eq = clahe.apply(gray)
    edges = cv2.Canny(eq, 30, 100)
    # A little horizontal/vertical closing so a dashed border reads continuous.
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    min_len = max(30, int(min(H, W) * min_len_frac))
    segs = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=40,
        minLineLength=min_len, maxLineGap=20,
    )
    if segs is None:
        return None

    h_coords, v_coords = [], []
    for seg in segs.reshape(-1, 4):
        x1, y1, x2, y2 = (int(v) for v in seg)
        if abs(y1 - y2) <= 3 and abs(x1 - x2) >= min_len:
            h_coords.append((y1 + y2) // 2)
        elif abs(x1 - x2) <= 3 and abs(y1 - y2) >= min_len:
            v_coords.append((x1 + x2) // 2)
    # Deduplicate near-equal coordinates.
    def dedup(vals, tol=6):
        out = []
        for v in sorted(vals):
            if not out or v - out[-1] > tol:
                out.append(v)
            else:
                out[-1] = (out[-1] + v) // 2
        return out
    h_coords, v_coords = dedup(h_coords), dedup(v_coords)

    tops = [c for c in h_coords if c <= py]
    bots = [c for c in h_coords if c >= py]
    lefts = [c for c in v_coords if c <= px]
    rights = [c for c in v_coords if c >= px]
    if not (tops and bots and lefts and rights):
        return None

    # A single non-maximized window rarely spans most of the screen; capping the
    # candidate size stops spurious rectangles that stitch together borders from
    # several different windows. (Truly maximized windows use "Select whole
    # image" instead.)
    max_w, max_h = 0.8 * W, 0.8 * H

    best, best_area = None, 0
    for T in tops:
        for B in bots:
            if B - T < 60 or B - T > max_h:
                continue
            for L in lefts:
                for R in rights:
                    if R - L < 60 or R - L > max_w:
                        continue
                    area = (R - L) * (B - T)
                    if area <= best_area:
                        continue  # can't beat current best; skip the checks
                    if _side_support(edges, "h", T, L, R) < min_support:
                        continue
                    if _side_support(edges, "h", B, L, R) < min_support:
                        continue
                    if _side_support(edges, "v", L, T, B) < min_support:
                        continue
                    if _side_support(edges, "v", R, T, B) < min_support:
                        continue
                    best_area = area
                    best = (int(L), int(T), int(R - L + 1), int(B - T + 1))
    return best


def _detect_by_contours(gray, point, min_area, max_area):
    """Fallback detector: largest bounded contour containing the click.

    Works when the background is fairly plain (a solid-colour desktop), where a
    window's outline closes into a clean contour.
    """
    px, py = point
    h, w = gray.shape[:2]
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blur, 30, 90)
    edges = cv2.bitwise_or(edges, cv2.Canny(blur, 60, 160))
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=2)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    best_rect, best_score = None, None
    for c in contours:
        x, y, rw, rh = cv2.boundingRect(c)
        area = float(rw * rh)
        if area < min_area or area > max_area:
            continue
        if not (x <= px <= x + rw and y <= py <= y + rh):
            continue
        aspect = rw / float(rh) if rh else 999
        if aspect > 12 or aspect < 1 / 12:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        score = area * (1.06 if 4 <= len(approx) <= 8 else 1.0)
        if best_score is None or score > best_score:
            best_score = score
            best_rect = _clamp_rect((x, y, rw, rh), w, h)
    return best_rect


def _area(r):
    return r[2] * r[3] if r else 0


def _contains(outer, inner, tol=10):
    ox, oy, ow, oh = outer
    ix, iy, iw, ih = inner
    return (ox <= ix + tol and oy <= iy + tol
            and ox + ow >= ix + iw - tol and oy + oh >= iy + ih - tol)


def _choose_rect(rect_l, rect_c):
    """Combine the line- and contour-based candidates.

    The line detector is the reliable one on busy/photographic backgrounds; the
    contour detector is better at capturing a whole window (title bar included)
    on plain backgrounds. When they describe the same window (one nested in the
    other) and are within a modest size ratio, take the larger -- so a title bar
    the line detector split off is recovered. When the contour result is much
    bigger (a background blob) or the two disagree, trust the line result.
    """
    if rect_l is None:
        return rect_c
    if rect_c is None:
        return rect_l
    big, small = (rect_c, rect_l) if _area(rect_c) >= _area(rect_l) else (rect_l, rect_c)
    if _contains(big, small) and _area(big) <= 2.0 * _area(small):
        return big
    return rect_l


def detect_window_at(
    image_bgr: np.ndarray,
    point: tuple[int, int],
    min_area_frac: float = 0.01,
    max_area_frac: float = 0.995,
) -> Optional[Detection]:
    """Detect the application window containing ``point``.

    Strategy: first look for the window as the largest rectangle of long,
    axis-aligned edges (robust on busy/photographic desktops). If that finds
    nothing, fall back to contour-closure detection (good on plain desktops).
    The chosen rectangle is then edge-refined, border-snapped, and given a
    color-measured corner radius.

    Args:
        image_bgr: screenshot as an OpenCV BGR (or BGRA) image.
        point: (x, y) click location in image pixel coordinates.
        min_area_frac: for the contour fallback, ignore regions smaller than
            this fraction of the image (the GUI "sensitivity" slider).
        max_area_frac: for the contour fallback, ignore regions larger than
            this fraction of the image.

    Returns:
        A :class:`Detection`, or ``None`` if nothing suitable was found.
    """
    if image_bgr is None or image_bgr.size == 0:
        return None

    h, w = image_bgr.shape[:2]
    px, py = int(point[0]), int(point[1])
    if not (0 <= px < w and 0 <= py < h):
        return None

    if image_bgr.ndim == 3 and image_bgr.shape[2] >= 3:
        gray = cv2.cvtColor(image_bgr[:, :, :3], cv2.COLOR_BGR2GRAY)
    else:
        gray = image_bgr.copy()

    total = float(w * h)
    rect_l = _detect_by_lines(gray, (px, py))
    rect_c = _detect_by_contours(
        gray, (px, py), min_area_frac * total, max_area_frac * total
    )
    rect = _choose_rect(rect_l, rect_c)
    if rect is None:
        return None

    # Key each straight side against its local background and move it onto the
    # first fully-window pixel, so the rectangle is pixel accurate.
    rect = refine_rect_edges(image_bgr, rect)
    # A window flush to the screen edge has no detectable border there; snap.
    rect = snap_rect_to_borders(rect, w, h)

    # Reject a phantom rectangle whose sides are not actually backed by edges
    # (a blob the contour detector stitched out of a busy background). Better to
    # return nothing and let the user drag a selection than to grab the wrong
    # region confidently. Sides on the image border are exempt.
    if not _rect_edges_supported(gray, rect):
        return None

    # Model the corners as one symmetric radius, measured against the desktop.
    radius = _estimate_radius_color(image_bgr, rect)

    return Detection(rect=rect, corner_radius=radius)


def _rect_edges_supported(gray, rect, min_support=0.30):
    """True if each non-border side of ``rect`` lies on real edge pixels."""
    H, W = gray.shape[:2]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    edges = cv2.Canny(clahe.apply(gray), 30, 100)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    x, y, w, h = rect
    L, T, R, B = x, y, x + w - 1, y + h - 1
    checks = []
    if T > 2:
        checks.append(_side_support(edges, "h", T, L, R))
    if B < H - 3:
        checks.append(_side_support(edges, "h", B, L, R))
    if L > 2:
        checks.append(_side_support(edges, "v", L, T, B))
    if R < W - 3:
        checks.append(_side_support(edges, "v", R, T, B))
    return all(c >= min_support for c in checks) if checks else True


# The arc of a quarter circle of radius r inscribed in a right-angle corner
# meets the 45-degree diagonal at a per-axis offset of r * (1 - 1/sqrt(2)).
_DIAG_OFFSET = 1.0 - 1.0 / np.sqrt(2.0)  # ~= 0.2929


def _estimate_radius_color(image_bgr, rect, max_radius=None):
    """Estimate the corner radius by keying each corner against the desktop.

    Walk inward along each corner's diagonal until the pixel stops matching the
    exterior desktop colour; that distance implies the radius (same geometry as
    :func:`estimate_corner_radius`). This needs no contour, so it works with the
    line-based detector. A square corner reads as 0 (its corner pixel is already
    window, not desktop); an occluded or low-contrast corner is skipped and the
    median of the rest is used.
    """
    img = image_bgr[:, :, :3].astype(np.float32)
    H, W = img.shape[:2]
    x, y, w, h = rect
    if max_radius is None:
        max_radius = int(min(min(w, h) // 4, 80))
    if max_radius <= 0:
        return 0

    corners = [
        ((x, y), (1, 1), (y - 3, y, x - 3, x)),
        ((x + w - 1, y), (-1, 1), (y - 3, y, x + w, x + w + 3)),
        ((x, y + h - 1), (1, -1), (y + h, y + h + 3, x - 3, x)),
        ((x + w - 1, y + h - 1), (-1, -1), (y + h, y + h + 3, x + w, x + w + 3)),
    ]
    ests = []
    for (cx, cy), (dx, dy), ext in corners:
        if cx <= 0 or cy <= 0 or cx >= W - 1 or cy >= H - 1:
            continue  # corner on the image border: no desktop to key against
        bg = _median3(img, *ext)
        deep = _median3(
            img, cy + dy * max_radius - 1, cy + dy * max_radius + 2,
            cx + dx * max_radius - 1, cx + dx * max_radius + 2,
        )
        contrast = float(np.linalg.norm(bg - deep))
        if contrast < 25:
            continue  # occluded, or window edge same colour as desktop
        thresh = max(30.0, 0.5 * contrast)
        hit = 0
        for a in range(0, max_radius + 3):
            sx, sy = cx + dx * a, cy + dy * a
            if not (0 <= sx < W and 0 <= sy < H):
                break
            if np.linalg.norm(img[sy, sx] - bg) >= thresh:
                hit = a
                break
        ests.append(hit / _DIAG_OFFSET)
    if not ests:
        return 0
    r = int(round(float(np.median(ests))))
    return 0 if r < 3 else min(r, max_radius)


def estimate_corner_radius(
    region_mask: np.ndarray,
    rect: tuple[int, int, int, int],
    max_radius: Optional[int] = None,
) -> int:
    """Estimate a single rounded-corner radius for a rectangular region.

    For each of the four corners of ``rect`` we walk inward along the corner's
    diagonal until we cross from background into the filled ``region_mask``.
    That crossing distance ``a`` (in per-axis pixels) implies a corner radius
    ``r = a / (1 - 1/sqrt(2))``. The four estimates are reduced with a median
    so one corner spoiled by a shadow, a widget, or a broken edge cannot skew
    the result, then applied symmetrically to every corner.

    Args:
        region_mask: uint8 mask (non-zero = window) covering the image.
        rect: (x, y, w, h) window bounding box.
        max_radius: cap on the returned radius; defaults to a sensible fraction
            of the smaller side (real window radii are small).

    Returns:
        The estimated radius in pixels (0 = square corners).
    """
    x, y, w, h = rect
    mh, mw = region_mask.shape[:2]
    if max_radius is None:
        max_radius = int(min(min(w, h) // 4, 80))
    max_radius = max(0, max_radius)
    if max_radius == 0:
        return 0

    # (corner pixel, inward per-axis direction)
    corners = (
        ((x, y), (1, 1)),                      # top-left
        ((x + w - 1, y), (-1, 1)),             # top-right
        ((x, y + h - 1), (1, -1)),             # bottom-left
        ((x + w - 1, y + h - 1), (-1, -1)),    # bottom-right
    )

    estimates = []
    # Probe a little past max_radius so a genuinely large radius is not clipped
    # to exactly the cap and mistaken for the true value.
    probe = max_radius + 2
    for (cx, cy), (dx, dy) in corners:
        hit = None
        for a in range(0, probe + 1):
            sx, sy = cx + dx * a, cy + dy * a
            if not (0 <= sx < mw and 0 <= sy < mh):
                break
            if region_mask[sy, sx]:
                hit = a
                break
        if hit is None:
            # This corner never entered the region within the probe distance;
            # it is unreliable (shadow/gap), so skip it rather than guess.
            continue
        estimates.append(hit / _DIAG_OFFSET)

    if not estimates:
        return 0

    r = int(round(float(np.median(estimates))))
    # Radii below a couple of pixels are indistinguishable from anti-aliasing
    # on a square corner; treat them as square.
    if r < 3:
        return 0
    return min(r, max_radius)


def _rounded_rect_alpha(h: int, w: int, radius: int) -> np.ndarray:
    """Return an ``h x w`` uint8 alpha mask for a rounded rectangle."""
    alpha = np.zeros((h, w), np.uint8)
    r = int(max(0, min(radius, min(h, w) // 2)))
    if r <= 0:
        alpha[:] = 255
        return alpha
    # Central cross of full-opacity rectangles.
    cv2.rectangle(alpha, (r, 0), (w - r, h), 255, -1)
    cv2.rectangle(alpha, (0, r), (w, h - r), 255, -1)
    # Four corner discs.
    for cx, cy in ((r, r), (w - r, r), (r, h - r), (w - r, h - r)):
        cv2.circle(alpha, (cx, cy), r, 255, -1)
    return alpha


def _clean_matte(alpha, at_top, at_bottom, at_left, at_right, cleanup):
    """Erode ``alpha`` inward by ``cleanup`` px on background-adjacent sides.

    Eliminates the fringe: the outermost ring of a hard cut is an anti-aliased
    blend of window + desktop and would show the old background color when the
    cutout is placed on a new one. Eroding just inside the boundary drops that
    contaminated ring. Sides sitting on the image edge are NOT eroded -- there
    is no background beyond them, so nothing to bleed and no content to lose.
    """
    h, w = alpha.shape[:2]
    k = int(cleanup)
    # Pad every side by k. Background sides are padded with 0 so erosion bites
    # into them; image-border sides are padded with 255 so their real pixels
    # keep no zero neighbour and survive untouched.
    padded = np.zeros((h + 2 * k, w + 2 * k), np.uint8)
    padded[k : k + h, k : k + w] = alpha
    if at_top:
        padded[:k, k : k + w] = 255
    if at_bottom:
        padded[k + h :, k : k + w] = 255
    if at_left:
        padded[k : k + h, :k] = 255
    if at_right:
        padded[k : k + h, k + w :] = 255
    kernel = np.ones((2 * k + 1, 2 * k + 1), np.uint8)
    eroded = cv2.erode(padded, kernel)
    return eroded[k : k + h, k : k + w]


def _median3(img, r0, r1, c0, c1):
    r0, c0 = max(0, r0), max(0, c0)
    r1 = min(img.shape[0], max(r0 + 1, r1))
    c1 = min(img.shape[1], max(c0 + 1, c1))
    return np.median(img[r0:r1, c0:c1].reshape(-1, 3), axis=0)


def _decontaminate_corners(image_bgr, rect, alpha, radius, band=4, size=48):
    """Remove exposed-desktop crescents left inside rounded corners.

    A rounded corner's arc radius is only estimated, so the geometric mask can
    keep a crescent of desktop opaque between the mask arc and the true window
    arc. We key each corner against the desktop colour and drop opaque pixels
    that match -- but ONLY when the corner actually exposes desktop.

    A corner is treated as rounded (desktop-exposing) only if the crop's corner
    pixel matches the exterior desktop just outside the rectangle there. For a
    square window the crop corner *is* the window, does not match the exterior,
    and is skipped -- so window content is never keyed away. Corners on the
    image border are skipped too (no desktop beyond). The region is sized
    generously and independently of the radius estimate, since keying only
    removes true background.
    """
    img = image_bgr[:, :, :3].astype(np.float32)
    H, W = img.shape[:2]
    x, y, w, h = rect
    ah, aw = alpha.shape[:2]
    crop = img[y:y + ah, x:x + aw]
    r = int(min(min(ah, aw) // 2, max(size, radius + band)))
    if r <= 1:
        return alpha

    # Per corner: (alpha corner slice, crop-corner 3x3 box, interior 3x3 box,
    # exterior sample box in full-image coords, this corner's border flags).
    corners = [
        ("tl", (slice(0, r), slice(0, r)), (0, 3, 0, 3), (r - 3, r, r - 3, r),
         (y - 3, y, x - 3, x), (x <= 0 or y <= 0)),
        ("tr", (slice(0, r), slice(aw - r, aw)), (0, 3, aw - 3, aw),
         (r - 3, r, aw - r, aw - r + 3), (y - 3, y, x + w, x + w + 3),
         (x + w >= W or y <= 0)),
        ("bl", (slice(ah - r, ah), slice(0, r)), (ah - 3, ah, 0, 3),
         (ah - r, ah - r + 3, r - 3, r), (y + h, y + h + 3, x - 3, x),
         (x <= 0 or y + h >= H)),
        ("br", (slice(ah - r, ah), slice(aw - r, aw)), (ah - 3, ah, aw - 3, aw),
         (ah - r, ah - r + 3, aw - r, aw - r + 3),
         (y + h, y + h + 3, x + w, x + w + 3), (x + w >= W or y + h >= H)),
    ]
    for _name, csl, cbox, ibox, ext, at_border in corners:
        if at_border:
            continue
        corner_bg = _median3(crop, *cbox)
        exterior = _median3(img, *ext)
        # Only proceed if the crop corner really is exposed desktop.
        if np.linalg.norm(corner_bg - exterior) > 30:
            continue
        interior = _median3(crop, *ibox)
        contrast = float(np.linalg.norm(corner_bg - interior))
        if contrast < 25:
            continue
        thresh = 0.6 * contrast
        sub = crop[csl]
        near_bg = np.linalg.norm(sub - corner_bg, axis=2) < thresh
        a = alpha[csl]
        a[near_bg] = 0
        alpha[csl] = a
    return alpha


def extract_rgba(
    image_bgr: np.ndarray,
    rect: tuple[int, int, int, int],
    contour: Optional[np.ndarray] = None,
    corner_radius: int = 0,
    edge_cleanup: int = 1,
) -> np.ndarray:
    """Crop ``rect`` out of the image and build an clean RGBA cutout.

    Everything outside the window shape is made transparent, and the boundary
    is cleaned so no background color bleeds into the result.

    Args:
        image_bgr: source screenshot (BGR or BGRA).
        rect: (x, y, w, h) region to extract.
        contour: optional outline (image coords) used as the alpha mask; when
            given it takes precedence over ``corner_radius``.
        corner_radius: if > 0 and no contour is supplied, round the crop's
            corners by this many pixels (nice for modern window decorations).
        edge_cleanup: erode the alpha inward by this many pixels on sides that
            border the desktop, removing the anti-aliased fringe so nothing of
            the old background shows when compositing. 0 disables it. Sides on
            the image boundary are never eroded.

    Returns:
        An ``h x w x 4`` BGRA uint8 array.
    """
    h_img, w_img = image_bgr.shape[:2]
    x, y, w, h = _clamp_rect(rect, w_img, h_img)
    crop = image_bgr[y : y + h, x : x + w]

    if crop.ndim == 3 and crop.shape[2] == 4:
        bgra = crop.copy()
    else:
        bgra = cv2.cvtColor(crop[:, :, :3], cv2.COLOR_BGR2BGRA)

    if contour is not None:
        mask = np.zeros((h, w), np.uint8)
        shifted = contour.reshape(-1, 2).astype(np.int32) - np.array([x, y])
        cv2.fillPoly(mask, [shifted], 255)
        alpha = mask
    elif corner_radius > 0:
        alpha = _rounded_rect_alpha(h, w, corner_radius)
    else:
        alpha = np.full((h, w), 255, np.uint8)

    if contour is None and edge_cleanup > 0:
        alpha = _decontaminate_corners(
            image_bgr, (x, y, w, h), alpha, corner_radius
        )

    if edge_cleanup > 0:
        alpha = _clean_matte(
            alpha,
            at_top=(y <= 0),
            at_bottom=(y + h >= h_img),
            at_left=(x <= 0),
            at_right=(x + w >= w_img),
            cleanup=edge_cleanup,
        )

    bgra[:, :, 3] = alpha
    return bgra
