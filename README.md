# Window Extractor

A small local GTK/GNOME desktop app for Debian/Linux that pulls a single
application window out of a screenshot and gives it a **transparent
background** — ready to **save as a PNG** or **copy straight to the clipboard**.

It solves the everyday annoyance of there being no easy way to capture *only*
one window, or to isolate a window from a busy desktop background.

## How it works

1. **Open** a screenshot (`Ctrl+O`) or **paste** one from the clipboard
   (`Ctrl+V`) — e.g. take a full-screen shot with `PrintScreen`.
2. In **Select** mode (the default), **drag a rough box** around the window.
   On release it **snaps to the window's real edges**. Nudge the 8 handles to
   fine-tune, or press **Snap** (`S`) again after moving it.
3. **Save PNG** (`Ctrl+S`) or **Copy** (`Ctrl+C`). The area outside the window
   is transparent, with pixel-perfect edges (no background bleed).

> **Why drag instead of one click?** Auto-detecting *which* window you mean from
> a flattened screenshot is unreliable on busy or overlapping desktops (a dark
> window on a photo wallpaper, windows stacked on windows — the information that
> says which window is in front is simply gone once the image is flattened). A
> quick rough drag removes that ambiguity; snapping + the pixel-perfect cleanup
> do the precise part. A best-effort **Detect** mode (click-to-guess) is still
> there in the toolbar, but Select is the dependable path.

Turn snapping off (☰ menu) to place the box entirely by hand, and use
**Select whole image** (`Ctrl+A`) for a maximized window.

The menu (☰) has a **detection sensitivity** slider (how large a region the
detector will accept), a **corner radius** control, and zoom controls (also
`Ctrl` + scroll wheel).

### How detection works

A window is modelled as a **rectangle + a single corner radius**:

1. **Rectangle.** Two detectors run and the better result is chosen: a
   **long-straight-line** finder (robust on busy/photographic wallpapers, which
   have almost no long axis-aligned lines) and a **contour** finder (best on
   plain desktops). The result must have its sides backed by real edges — if it
   doesn't, detection reports *nothing found* rather than grabbing a phantom
   region, and you switch to **Select** mode.
2. **Corner radius.** From each corner we walk inward along the 45° diagonal,
   keyed against the desktop colour, until we reach window pixels. For a
   quarter-circle of radius `r`, the arc meets that diagonal at a per-axis
   offset of `r·(1 − 1/√2)`, so each corner gives `r ≈ 3.414 × offset`. The four
   are reduced with a **median** and applied **symmetrically**.

**Limitations & the reliable fallback.** Auto-detection is a heuristic. It does
well on windows with clear borders, but a **dark-mode window on a busy photo
wallpaper** (low-contrast borders, overlapping windows) can defeat it — it will
say *no window found*. When that happens, use **Select** mode: drag a rectangle
and fine-tune with the handles. The selection then goes through the exact same
**pixel-perfect edge cleanup** as an auto-detected one, so a hand-drawn cut is
just as clean.

### Diagnosing / tuning detection on your own screenshot

To see exactly what the detector does on a specific screenshot (and get an
overlay image you can eyeball):

```bash
python3 -m windowextractor.diagnose /path/to/shot.png X Y
# X Y = the pixel you would click inside the window (omit for image centre)
```

It prints the line-based, contour-based, and final rectangles and writes
`shot.detect.png` with them drawn on (green = final, amber = line, blue =
contour, red cross = your click).

### Pixel-perfect edges (no background bleed)

Cutouts are meant to be dropped into other documents, so a 1px halo of the old
desktop bleeding through is unacceptable. Three things prevent it:

1. **Edge refinement.** Detection runs on a *dilated* edge map, which lands the
   box a few pixels off. Each straight side is then keyed against its own local
   background and moved onto the first *fully-window* pixel — so the cut never
   sits on desktop or on an anti-aliased blend.
2. **Corner decontamination.** A rounded corner's radius is only estimated, so
   a crescent of desktop can survive inside the arc. Each corner is keyed
   against the desktop colour and matching pixels are dropped — but only when
   the corner actually exposes desktop (a square window's corner is left
   untouched, so window content is never keyed away).
3. **Edge cleanup.** A final erosion (default **1px**, adjustable 0–5 in the ☰
   menu) trims the anti-aliased boundary ring on desktop-facing sides only —
   never on a side that sits on the image boundary, where there's no background
   to bleed and trimming would lose real content.

The result composites cleanly onto any new background. If you still see a fringe
on an unusual screenshot, raise **Edge cleanup**; to keep every last pixel of a
window's own border, set it to 0.

### Windows at the image edge

A window flush against **one** image edge (or a corner) is still detected: the
edge itself has no gradient to find, but the *perpendicular* sides run to the
image boundary and fix the bounding box. Such sides are then **snapped** exactly
to the image edge so no thin sliver of desktop is left behind.

The one case edge detection *cannot* solve is a **maximized / full-screen
window that fills the whole screenshot** — there is no outer border anywhere in
the image to detect. When a detection hugs 3+ image edges (the tell-tale sign),
the status bar points you at **Select whole image (`Ctrl+A`)**, which is the
correct result: a maximized window *is* the whole screenshot.

## Install (Debian / Ubuntu, GNOME)

```bash
git clone <this-repo> windowsExtractor
cd windowsExtractor
./install.sh
```

`install.sh` installs the dependencies from `apt`, adds a `window-extractor`
launcher to `~/.local/bin`, and registers a GNOME desktop entry (search for
"Window Extractor" in the Activities overview). To remove the launcher and
desktop entry again: `./install.sh --uninstall`.

### Manual dependency install

Everything comes from the Debian archive — no pip or virtualenv needed:

```bash
sudo apt install python3-gi python3-gi-cairo gir1.2-gtk-3.0 \
                 python3-opencv python3-numpy
```

## Run without installing

```bash
./run.sh                 # or:  python3 -m windowextractor
./run.sh some-shot.png   # open a file directly
```

## Notes

- Works under both **X11 and Wayland** (GNOME's default). Clipboard copy/paste
  goes through the standard GTK clipboard, so it behaves like any other app.
- Auto-detection is a heuristic — screenshots vary. If a click grabs too much
  or too little, nudge the **sensitivity** slider, or just use **Select** mode
  and drag the rectangle. Every detection is editable via the handles.
- Detected windows keep their real outline (so a rounded/odd corner is cut to
  transparent). For a hand-drawn selection, set a **rounded corners** radius in
  the menu if you want the corners softened.

## Project layout

| File | Purpose |
|------|---------|
| `windowextractor/detector.py`  | Window detection + RGBA extraction (pure OpenCV/NumPy, no GUI) |
| `windowextractor/imageutils.py`| GdkPixbuf ⇄ NumPy ⇄ PNG conversions |
| `windowextractor/gui.py`       | GTK3 application, canvas, selection handles |
| `tests/test_detector.py`       | Headless tests for the detection/extraction logic |
| `install.sh` / `run.sh`        | Install / run helpers |

## Tests

```bash
python3 tests/test_detector.py      # no display or GTK required
# or, if you have pytest:
pytest tests/
```

## License

MIT — see `LICENSE`.
