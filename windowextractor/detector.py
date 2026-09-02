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


def _edge_ridges(gray, floor=8):
    """Binary maps of vertical- and horizontal-edge ridges.

    ``rv[r, c]`` marks a vertical-edge ridge (a local maximum of the horizontal
    gradient, above ``floor``) -- the kind of pixel a vertical window border is
    made of; ``rh`` is the horizontal-edge equivalent. Working from ridge
    *presence* (not magnitude) is what lets the geometry test below weigh a long
    continuous line over a short high-contrast one.
    """
    g = gray.astype(np.float32)
    gx = np.abs(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3))
    gy = np.abs(cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3))
    rv = (gx > floor) & (gx >= np.roll(gx, 1, 1)) & (gx >= np.roll(gx, -1, 1))
    rh = (gy > floor) & (gy >= np.roll(gy, 1, 0)) & (gy >= np.roll(gy, -1, 0))
    return rv, rh


def _line_span(ridge, orient, coord, lo, hi, tol=1):
    """Fraction of the ``[lo, hi]`` span covered by an edge at ``coord`` (+/-tol).

    This is the geometry weight: a window border is a straight line that covers
    ~all of its side (span ~1.0), while a button or icon edge covers only a few
    rows/columns (span small), and textured wallpaper never lines up on one
    row/column for its whole length (span stays moderate). So a high span
    uniquely marks a true window border regardless of how strong shorter edges
    are.
    """
    H, W = ridge.shape[:2]
    coord = int(coord)
    if hi - lo < 2:
        return 0.0
    if orient == "v":
        c0, c1 = max(0, coord - tol), min(W, coord + tol + 1)
        seg = ridge[max(0, lo):min(H, hi), c0:c1]
        present = seg.any(axis=1)
    else:
        r0, r1 = max(0, coord - tol), min(H, coord + tol + 1)
        seg = ridge[r0:r1, max(0, lo):min(W, hi)]
        present = seg.any(axis=0)
    return float(present.mean()) if present.size else 0.0


def snap_selection_to_edges(image_bgr, rect, search=90, min_span=0.5):
    """Snap each side of a hand-drawn selection onto the true window border.

    A window border is distinguished from everything else by GEOMETRY, not
    contrast: it is a straight line that runs the (near) full length of its side,
    so almost every row (for a vertical side) has an edge at that exact column.
    A button or icon edge -- however high-contrast -- covers only a few rows; a
    textured wallpaper (a forest photo) has vertical structure but never lines up
    on a single column for its whole length. So for each side we take the line of
    MAXIMUM edge span (``_line_span``) in the search band -- the most complete
    line, which is the border -- provided it clears a ``min_span`` floor, and
    among near-maximal lines we pick the one nearest where the user drew.

    Using the maximum (rather than a fixed span threshold) keeps it robust to a
    loose drag: even when the drawn box is much wider than the window -- so the
    border only covers part of the measured extent -- the border is still the
    most complete line present. Two passes let the tightened perpendicular
    extents sharpen each border's span.
    """
    gray = cv2.cvtColor(image_bgr[:, :, :3], cv2.COLOR_BGR2GRAY)
    H, W = gray.shape[:2]
    rv, rh = _edge_ridges(gray)

    x, y, w, h = rect
    L, T, R, B = x, y, x + w - 1, y + h - 1

    def best(orient, coord, lo, hi, limit):
        ridge = rv if orient == "v" else rh
        lo_c, hi_c = max(1, coord - search), min(limit - 1, coord + search + 1)
        spans = [(c, _line_span(ridge, orient, c, lo, hi))
                 for c in range(lo_c, hi_c)]
        max_s = max((s for _c, s in spans), default=0.0)
        if max_s < min_span:
            return coord  # no line worth snapping to
        near = [c for c, s in spans if s >= max_s - 0.05]
        return min(near, key=lambda c: abs(c - coord))

    # Search from the ORIGINAL drawn line; two passes let each border use the
    # others' tightened extent (which sharpens its span toward 1.0).
    L0, T0, R0, B0 = L, T, R, B
    for _ in range(2):
        T = best("h", T0, L, R, H)
        B = best("h", B0, L, R, H)
        L = best("v", L0, T, B, W)
        R = best("v", R0, T, B, W)
    if R <= L or B <= T:
        return rect
    return (int(L), int(T), int(R - L + 1), int(B - T + 1))


def estimate_corner_radius_geom(image_bgr, rect, floor=8, max_radius=40):
    """Estimate the corner radius from where the straight edges meet the arcs.

    Given the four straight borders, each one runs across the middle of its side
    but is ABSENT in the corner arcs (there the boundary curves away from the
    straight line). The distance from a corner vertex to where the straight edge
    begins is the arc's inset -- the corner radius. We measure that gap at all
    eight edge-ends and take the median. (Returns 0 for square corners, and can
    read low over a textured background whose noise fills the arc region -- the
    live preview lets the user adjust.)
    """
    gray = cv2.cvtColor(image_bgr[:, :, :3], cv2.COLOR_BGR2GRAY)
    H, W = gray.shape[:2]
    rv, rh = _edge_ridges(gray, floor)
    x, y, w, h = _clamp_rect(rect, W, H)
    L, T, R, B = x, y, x + w - 1, y + h - 1
    cap = int(min(max_radius, min(w, h) // 3))

    def inset_v(c, lo, hi, frm):
        c = int(np.clip(c, 1, W - 2))
        col = rv[max(0, lo):min(H, hi), c - 1:c + 2].any(axis=1)
        idx = np.where(col)[0]
        if idx.size == 0:
            return 0
        return int(idx[0]) if frm == "lo" else int(len(col) - 1 - idx[-1])

    def inset_h(r, lo, hi, frm):
        r = int(np.clip(r, 1, H - 2))
        row = rh[r - 1:r + 2, max(0, lo):min(W, hi)].any(axis=0)
        idx = np.where(row)[0]
        if idx.size == 0:
            return 0
        return int(idx[0]) if frm == "lo" else int(len(row) - 1 - idx[-1])

    insets = [
        inset_v(L, T, B, "lo"), inset_v(L, T, B, "hi"),
        inset_v(R, T, B, "lo"), inset_v(R, T, B, "hi"),
        inset_h(T, L, R, "lo"), inset_h(T, L, R, "hi"),
        inset_h(B, L, R, "lo"), inset_h(B, L, R, "hi"),
    ]
    r = int(np.median(insets))
    if r < 3:
        return 0
    return min(r, cap)


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
                     min_support=0.5):
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

    # Pick the SMALLEST well-bordered rectangle enclosing the click, not the
    # largest: on a multi-window desktop the largest supported rectangle just
    # stitches together borders from several different windows. The smallest one
    # whose four sides are each strongly backed by edges is the tightest real
    # frame around the click. (It can still land on an inner pane -- window vs.
    # pane is genuinely ambiguous from a flat screenshot -- which the user then
    # grows with the handles.)
    best, best_area = None, None
    for T in tops:
        for B in bots:
            if not (60 <= B - T <= max_h):
                continue
            for L in lefts:
                for R in rights:
                    if not (60 <= R - L <= max_w):
                        continue
                    area = (R - L) * (B - T)
                    if best_area is not None and area >= best_area:
                        continue
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
    # Prefer the line detector's tightest well-bordered rectangle; fall back to
    # the contour detector only for plain backgrounds. Never combine them into a
    # larger rectangle -- that stitched borders across windows.
    rect = _detect_by_lines(gray, (px, py))
    if rect is None:
        rect = _detect_by_contours(
            gray, (px, py), min_area_frac * total, max_area_frac * total
        )
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
    """Return an ``h x w`` uint8 alpha mask for a rounded rectangle.

    255 inside the rounded-rectangle geometry, 0 outside. The rounded corners
    are anti-aliased so the arc is smooth; the straight sides are pixel-aligned
    hard edges.
    """
    alpha = np.zeros((h, w), np.uint8)
    r = int(max(0, min(radius, min(h, w) // 2)))
    if r <= 0:
        alpha[:] = 255
        return alpha
    # Central cross of full-opacity rectangles.
    cv2.rectangle(alpha, (r, 0), (w - r, h), 255, -1)
    cv2.rectangle(alpha, (0, r), (w, h - r), 255, -1)
    # Four corner discs, anti-aliased for a smooth arc.
    for cx, cy in ((r, r), (w - r - 1, r), (r, h - r - 1), (w - r - 1, h - r - 1)):
        cv2.circle(alpha, (cx, cy), r, 255, -1, lineType=cv2.LINE_AA)
    return alpha


def _median3(img, r0, r1, c0, c1):
    r0, c0 = max(0, r0), max(0, c0)
    r1 = min(img.shape[0], max(r0 + 1, r1))
    c1 = min(img.shape[1], max(c0 + 1, c1))
    return np.median(img[r0:r1, c0:c1].reshape(-1, 3), axis=0)


def extract_rgba(
    image_bgr: np.ndarray,
    rect: tuple[int, int, int, int],
    corner_radius: int = 0,
) -> np.ndarray:
    """Crop ``rect`` out of the image and cut it to the window geometry.

    The geometry is the rectangle plus an optional symmetric ``corner_radius``.
    Every pixel INSIDE the geometry is kept exactly as-is; every pixel OUTSIDE
    it is made fully transparent. Nothing else is touched -- no erosion, no
    colour keying -- so getting a clean cutout is purely a matter of getting the
    geometry right (which the snap + handles do). The rounded corners are
    anti-aliased for a smooth arc.

    Args:
        image_bgr: source screenshot (BGR or BGRA).
        rect: (x, y, w, h) geometry rectangle in image pixels.
        corner_radius: corner radius of the geometry (0 = square corners).

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

    # Alpha = the geometry: 255 inside, 0 outside. Colour channels untouched.
    bgra[:, :, 3] = _rounded_rect_alpha(h, w, corner_radius)
    return bgra
