# Window Extractor

A small local GTK/GNOME desktop app for Debian/Linux that pulls a single
application window out of a screenshot and gives it a **transparent
background** — ready to **save as a PNG** or **copy straight to the clipboard**.

It solves the everyday annoyance of there being no easy way to capture *only*
one window, or to isolate a window from a busy desktop background.

## How it works

1. **Open** a screenshot (`Ctrl+O`) or **paste** one from the clipboard
   (`Ctrl+V`) — e.g. take a full-screen shot with `PrintScreen`.
2. In **Detect** mode, **click inside the window** you want. The app finds the
   window's rectangular boundary using edge/contour detection.
3. Fine-tune if needed: drag the 8 handles to resize, or drag the middle to
   move. Prefer to do it by hand? Switch to **Select** mode and drag out the
   rectangle yourself.
4. **Save PNG** (`Ctrl+S`) or **Copy** (`Ctrl+C`). The area outside the window
   is transparent.

The menu (☰) has a **detection sensitivity** slider (how large a region the
detector will accept), a **corner radius** control, and zoom controls (also
`Ctrl` + scroll wheel).

### How detection works

A window is modelled as a **rectangle + a single corner radius** — the two
things that fully describe a modern window decoration:

1. **Rectangle.** Edge detection (dual-threshold Canny → dilate/close →
   contours) yields candidate boxes; the largest one containing your click,
   below a full-image guard, is taken as the window's width/height. Robust,
   because a window's straight edges are its strongest lines.
2. **Corner radius.** From each of the four corners we walk inward along the
   45° diagonal until we cross into window pixels. For a quarter-circle of
   radius `r`, the arc meets that diagonal at a per-axis offset of
   `r·(1 − 1/√2)`, so each corner gives `r ≈ 3.414 × offset`. The four
   estimates are reduced with a **median** (so one corner spoiled by a shadow
   or a widget can't skew it) and applied **symmetrically** to all corners.

The measured radius pre-fills the **corner radius** control, where you can
still adjust it. The cutout is then a clean rounded rectangle — no jagged
contour tracing.

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
