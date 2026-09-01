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
        rect: (x, y, w, h) bounding box in image pixel coordinates.
        contour: optional Nx1x2 int array tracing the detected window outline,
            in image pixel coordinates. Used to build a tight alpha mask
            (e.g. to cut away rounded corners / drop shadow). May be ``None``.
    """

    rect: tuple[int, int, int, int]
    contour: Optional[np.ndarray] = None


def _clamp_rect(rect, w, h):
    x, y, rw, rh = rect
    x = max(0, min(int(x), w - 1))
    y = max(0, min(int(y), h - 1))
    rw = max(1, min(int(rw), w - x))
    rh = max(1, min(int(rh), h - y))
    return (x, y, rw, rh)


def detect_window_at(
    image_bgr: np.ndarray,
    point: tuple[int, int],
    min_area_frac: float = 0.01,
    max_area_frac: float = 0.995,
) -> Optional[Detection]:
    """Detect the application window containing ``point``.

    Args:
        image_bgr: screenshot as an OpenCV BGR (or BGRA) image.
        point: (x, y) click location in image pixel coordinates.
        min_area_frac: ignore candidate regions smaller than this fraction of
            the whole image. Raising it makes detection prefer larger windows
            (and ignore panels/widgets); lowering it lets it grab smaller
            regions. Exposed as a "sensitivity" slider in the GUI.
        max_area_frac: ignore candidate regions larger than this fraction of
            the image, so a border tracing the entire screenshot does not win
            over the actual window.

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
    min_area = min_area_frac * total
    max_area = max_area_frac * total

    # Build an edge map. Two passes of Canny with different thresholds are
    # merged so we catch both crisp borders and softer shadow edges. The edges
    # are dilated and closed so a window outline forms one continuous contour
    # even where the border is broken by rounded corners or anti-aliasing.
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blur, 30, 90)
    edges2 = cv2.Canny(blur, 60, 160)
    edges = cv2.bitwise_or(edges, edges2)
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=2)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(
        edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )

    best: Optional[Detection] = None
    best_score = None

    for c in contours:
        x, y, rw, rh = cv2.boundingRect(c)
        rect_area = float(rw * rh)
        if rect_area < min_area or rect_area > max_area:
            continue
        # Must contain the clicked point (inside the bounding box).
        if not (x <= px <= x + rw and y <= py <= y + rh):
            continue
        # Skip long thin slivers (toolbars, scrollbars, separators): a window
        # has a sensible aspect ratio.
        aspect = rw / float(rh) if rh else 999
        if aspect > 12 or aspect < 1 / 12:
            continue

        # Approximate the contour to a polygon so we can prefer clean,
        # rectangle-like outlines and reuse the polygon as an alpha mask.
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        # Prefer the LARGEST qualifying region containing the click. A window
        # is the outermost box the user is pointing at -- picking the smallest
        # would grab an interior panel or the region below the title bar, and
        # the max-area guard already keeps us from grabbing the whole desktop.
        # A near-rectangular outline gets a small bonus so a clean window
        # border wins over a ragged content blob of similar size.
        rectangular_bonus = 1.06 if 4 <= len(approx) <= 8 else 1.0
        score = rect_area * rectangular_bonus

        if best_score is None or score > best_score:
            best_score = score
            contour = approx if 4 <= len(approx) <= 8 else None
            best = Detection(
                rect=_clamp_rect((x, y, rw, rh), w, h),
                contour=contour,
            )

    return best


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


def extract_rgba(
    image_bgr: np.ndarray,
    rect: tuple[int, int, int, int],
    contour: Optional[np.ndarray] = None,
    corner_radius: int = 0,
) -> np.ndarray:
    """Crop ``rect`` out of the image and build an RGBA cutout.

    Everything outside the window shape is made transparent.

    Args:
        image_bgr: source screenshot (BGR or BGRA).
        rect: (x, y, w, h) region to extract.
        contour: optional outline (image coords) used as the alpha mask; when
            given it takes precedence over ``corner_radius``.
        corner_radius: if > 0 and no contour is supplied, round the crop's
            corners by this many pixels (nice for modern window decorations).

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
        # Smooth the 1px jaggies from fillPoly a touch.
        mask = cv2.GaussianBlur(mask, (3, 3), 0)
        alpha = mask
    elif corner_radius > 0:
        alpha = _rounded_rect_alpha(h, w, corner_radius)
    else:
        alpha = np.full((h, w), 255, np.uint8)

    bgra[:, :, 3] = alpha
    return bgra
