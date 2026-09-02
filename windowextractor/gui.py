"""GTK3 GUI for Window Extractor.

Layout: a toolbar of actions on top, and a scrollable image canvas below.
Open or paste a screenshot, then (Select mode, the default) drag a rough box
around a window -- it snaps to the window's real edges -- and nudge the 8
resize handles to fine-tune. (Detect mode is a best-effort click-to-guess that
is unreliable on busy/overlapping desktops.) Export the isolated window -- with
a transparent background -- to a PNG file or the clipboard.
"""

from __future__ import annotations

import math
import os
from typing import Optional

import gi

gi.require_version("Gtk", "3.0")
gi.require_version("Gdk", "3.0")
gi.require_version("GdkPixbuf", "2.0")
from gi.repository import Gdk, GdkPixbuf, Gio, GLib, Gtk  # noqa: E402

try:
    import cairo
except ImportError:  # pragma: no cover - handled at runtime by GTK stack
    cairo = None

from . import __version__, detector, imageutils

APP_ID = "com.veros.WindowExtractor"

# Screen-space size of resize handles and the minimum selection size (px).
HANDLE = 9
MIN_SEL = 8

# Handle identifiers: corners and edge midpoints, plus interior move / new.
_HANDLES = ("nw", "n", "ne", "e", "se", "s", "sw", "w")


class ImageCanvas(Gtk.DrawingArea):
    """Displays the screenshot and manages the selection rectangle."""

    def __init__(self, on_selection_changed):
        super().__init__()
        self._on_selection_changed = on_selection_changed

        self.image_bgr = None  # source screenshot (numpy BGR)
        self.pixbuf = None  # display pixbuf (RGB)
        self.zoom = 1.0
        self.mode = "select"  # "detect" or "select"
        self.sensitivity = 0.01  # min_area_frac for detection
        self.corner_radius = 0
        self.snap_enabled = True  # snap a hand-drawn box to window edges

        # Selection state (image pixel coords). The window is modelled as this
        # rectangle plus a symmetric corner radius (self.corner_radius).
        self.sel = None  # [x, y, w, h]

        # Interaction state.
        self._drag = None  # None | "move" | "new" | handle id
        self._drag_origin = None  # (img_x, img_y) anchor for the drag
        self._sel_at_press = None

        self.set_can_focus(True)
        self.add_events(
            Gdk.EventMask.BUTTON_PRESS_MASK
            | Gdk.EventMask.BUTTON_RELEASE_MASK
            | Gdk.EventMask.POINTER_MOTION_MASK
            | Gdk.EventMask.SCROLL_MASK
        )
        self.connect("draw", self._on_draw)
        self.connect("button-press-event", self._on_press)
        self.connect("button-release-event", self._on_release)
        self.connect("motion-notify-event", self._on_motion)
        self.connect("scroll-event", self._on_scroll)

    # -- image management ---------------------------------------------------

    def set_image_from_pixbuf(self, pixbuf: GdkPixbuf.Pixbuf):
        self.pixbuf = pixbuf
        self.image_bgr = imageutils.pixbuf_to_bgr(pixbuf)
        self.sel = None
        self.zoom = 1.0
        self._update_size()
        self.queue_draw()
        self._emit_changed()

    def has_image(self) -> bool:
        return self.image_bgr is not None

    def has_selection(self) -> bool:
        return self.sel is not None

    def _img_size(self):
        h, w = self.image_bgr.shape[:2]
        return w, h

    def _update_size(self):
        if self.pixbuf is None:
            return
        w, h = self._img_size()
        self.set_size_request(int(w * self.zoom), int(h * self.zoom))

    # -- zoom ---------------------------------------------------------------

    def set_zoom(self, zoom: float, center=None):
        zoom = max(0.05, min(zoom, 20.0))
        self.zoom = zoom
        self._update_size()
        self.queue_draw()

    def fit_to(self, viewport_w: int, viewport_h: int):
        if not self.has_image():
            return
        w, h = self._img_size()
        if w == 0 or h == 0:
            return
        self.zoom = min(viewport_w / w, viewport_h / h, 1.0)
        self._update_size()
        self.queue_draw()

    # -- coordinate helpers -------------------------------------------------

    def _to_image(self, wx, wy):
        return wx / self.zoom, wy / self.zoom

    def _to_widget(self, ix, iy):
        return ix * self.zoom, iy * self.zoom

    def _clamp_point(self, ix, iy):
        w, h = self._img_size()
        return max(0, min(ix, w)), max(0, min(iy, h))

    # -- hit testing --------------------------------------------------------

    def _handle_rects_widget(self):
        """Yield (handle_id, x, y, w, h) rects in widget coords."""
        if self.sel is None:
            return
        x, y, w, h = self.sel
        wx, wy = self._to_widget(x, y)
        ww, wh = w * self.zoom, h * self.zoom
        cx, cy = wx + ww / 2, wy + wh / 2
        pts = {
            "nw": (wx, wy),
            "n": (cx, wy),
            "ne": (wx + ww, wy),
            "e": (wx + ww, cy),
            "se": (wx + ww, wy + wh),
            "s": (cx, wy + wh),
            "sw": (wx, wy + wh),
            "w": (wx, cy),
        }
        half = HANDLE
        for hid, (px, py) in pts.items():
            yield hid, px - half, py - half, half * 2, half * 2

    def _handle_at(self, wx, wy) -> Optional[str]:
        for hid, hx, hy, hw, hh in self._handle_rects_widget():
            if hx <= wx <= hx + hw and hy <= wy <= hy + hh:
                return hid
        return None

    def _inside_selection(self, wx, wy) -> bool:
        if self.sel is None:
            return False
        x, y, w, h = self.sel
        ix, iy = self._to_image(wx, wy)
        return x <= ix <= x + w and y <= iy <= y + h

    # -- events -------------------------------------------------------------

    def _on_press(self, _w, event):
        if not self.has_image() or event.button != 1:
            return False
        self.grab_focus()
        wx, wy = event.x, event.y

        handle = self._handle_at(wx, wy)
        if handle:
            self._drag = handle
            self._sel_at_press = list(self.sel)
            return True

        if self._inside_selection(wx, wy):
            self._drag = "move"
            self._sel_at_press = list(self.sel)
            ix, iy = self._to_image(wx, wy)
            self._drag_origin = (ix, iy)
            return True

        ix, iy = self._to_image(wx, wy)
        ix, iy = self._clamp_point(ix, iy)
        if self.mode == "detect":
            self._detect_at(int(ix), int(iy))
        else:  # select mode: start drawing a new rectangle
            self._drag = "new"
            self._drag_origin = (ix, iy)
            self.sel = [ix, iy, 0, 0]
            self.queue_draw()
        return True

    def _on_release(self, _w, event):
        if event.button != 1:
            return False
        was_new = self._drag == "new"
        if was_new and self.sel is not None:
            # Discard a zero-size accidental click.
            if self.sel[2] < MIN_SEL or self.sel[3] < MIN_SEL:
                self.sel = None
        self._drag = None
        self._drag_origin = None
        self._sel_at_press = None
        self._normalise_selection()
        status = None
        radius = None
        # Snap a freshly-drawn rough box onto the window's real edges, then
        # match the geometry's corner radius to the window automatically.
        if was_new and self.sel is not None:
            if self.snap_enabled:
                self.snap_current()
                status = ("Snapped to edges — nudge the handles to fine-tune, "
                          "then Save or Copy. (Toggle snap in the ☰ menu.)")
            radius = self.estimate_radius()
        self.queue_draw()
        self._emit_changed(status=status, radius=radius)
        return True

    def snap_current(self):
        """Snap the current selection onto nearby window borders."""
        if self.sel is None or not self.has_image():
            return
        x, y, w, h = self.sel
        if w < MIN_SEL or h < MIN_SEL:
            return
        self.sel = list(detector.snap_selection_to_edges(self.image_bgr, (x, y, w, h)))
        self._normalise_selection()
        self.queue_draw()

    def _on_motion(self, _w, event):
        if not self.has_image():
            return False
        wx, wy = event.x, event.y

        if self._drag is None:
            # Update cursor to hint at the interaction under the pointer.
            self._update_cursor(wx, wy)
            return False

        ix, iy = self._to_image(wx, wy)
        ix, iy = self._clamp_point(ix, iy)

        if self._drag == "new":
            ox, oy = self._drag_origin
            self.sel = [min(ox, ix), min(oy, iy), abs(ix - ox), abs(iy - oy)]
        elif self._drag == "move":
            self._move_selection(ix, iy)
        else:  # resize handle
            self._resize_selection(self._drag, ix, iy)

        self.queue_draw()
        return True

    def _on_scroll(self, _w, event):
        # Ctrl+scroll zooms; plain scroll is left to the ScrolledWindow.
        if not (event.state & Gdk.ModifierType.CONTROL_MASK):
            return False
        if event.direction == Gdk.ScrollDirection.UP:
            self.set_zoom(self.zoom * 1.15)
        elif event.direction == Gdk.ScrollDirection.DOWN:
            self.set_zoom(self.zoom / 1.15)
        elif event.direction == Gdk.ScrollDirection.SMOOTH:
            _, dy = event.get_scroll_deltas()[1:]
            if dy:
                self.set_zoom(self.zoom * (1.0 - 0.1 * dy))
        return True

    def _update_cursor(self, wx, wy):
        window = self.get_window()
        if window is None:
            return
        cursor_name = None
        handle = self._handle_at(wx, wy)
        if handle:
            cursor_name = {
                "nw": "nw-resize", "n": "n-resize", "ne": "ne-resize",
                "e": "e-resize", "se": "se-resize", "s": "s-resize",
                "sw": "sw-resize", "w": "w-resize",
            }[handle]
        elif self._inside_selection(wx, wy):
            cursor_name = "move"
        elif self.mode == "detect":
            cursor_name = "crosshair"
        else:
            cursor_name = "cell"
        try:
            cursor = Gdk.Cursor.new_from_name(self.get_display(), cursor_name)
            window.set_cursor(cursor)
        except Exception:
            window.set_cursor(None)

    # -- selection maths ----------------------------------------------------

    def _move_selection(self, ix, iy):
        w, h = self._img_size()
        ox, oy = self._drag_origin
        sx, sy, sw, sh = self._sel_at_press
        nx = sx + (ix - ox)
        ny = sy + (iy - oy)
        nx = max(0, min(nx, w - sw))
        ny = max(0, min(ny, h - sh))
        self.sel = [nx, ny, sw, sh]

    def _resize_selection(self, handle, ix, iy):
        x, y, w, h = self._sel_at_press
        left, top, right, bottom = x, y, x + w, y + h
        if "w" in handle:
            left = ix
        if "e" in handle:
            right = ix
        if "n" in handle:
            top = iy
        if "s" in handle:
            bottom = iy
        self.sel = [
            min(left, right),
            min(top, bottom),
            abs(right - left),
            abs(bottom - top),
        ]

    def _normalise_selection(self):
        if self.sel is None:
            return
        img_w, img_h = self._img_size()
        x, y, w, h = self.sel
        x = max(0, min(int(round(x)), img_w - 1))
        y = max(0, min(int(round(y)), img_h - 1))
        w = max(1, min(int(round(w)), img_w - x))
        h = max(1, min(int(round(h)), img_h - y))
        self.sel = [x, y, w, h]

    # -- detection ----------------------------------------------------------

    def _detect_at(self, ix, iy):
        result = detector.detect_window_at(
            self.image_bgr, (ix, iy), min_area_frac=self.sensitivity
        )
        if result is None:
            self._emit_changed(status="No window boundary found here — "
                               "try Select mode or adjust sensitivity.")
            return
        x, y, w, h = result.rect
        self.sel = [x, y, w, h]
        self.queue_draw()
        radius = result.corner_radius

        img_w, img_h = self._img_size()
        if detector.border_sides((x, y, w, h), img_w, img_h) >= 3:
            # Hugs 3+ image edges: almost certainly a full-screen/maximized
            # window, whose outer border cannot be detected (it is not in the
            # image). Point the user at the whole-image shortcut.
            status = ("Looks like a full-screen window — its outer edge isn't "
                      "in the screenshot. Press Ctrl+A to grab the whole image.")
        elif radius > 0:
            status = (f"Window detected (≈{radius}px rounded corners) — drag "
                      "the handles to adjust, then Save or Copy.")
        else:
            status = ("Window detected — drag the handles to adjust, then "
                      "Save or Copy.")
        # Seed the corner-radius control from the measurement; it remains
        # user-editable and is the single source of truth for extraction.
        self._emit_changed(status=status, radius=radius)

    def select_whole_image(self):
        if not self.has_image():
            return
        w, h = self._img_size()
        self.sel = [0, 0, w, h]
        self.queue_draw()
        self._emit_changed(
            status="Selected the whole image.", radius=0
        )

    # -- extraction ---------------------------------------------------------

    def extract_bgra(self):
        if self.sel is None:
            return None
        x, y, w, h = self.sel
        return detector.extract_rgba(
            self.image_bgr, (x, y, w, h), corner_radius=self.corner_radius
        )

    def estimate_radius(self):
        """Estimate the corner radius of the current selection's window.

        Clamped to a modest maximum: on a loose selection the estimate can run
        large (it measures desktop past the true edge), and over-rounding cuts
        into the window, so we cap it. The user can raise it in the menu.
        """
        if self.sel is None or not self.has_image():
            return 0
        return detector.estimate_corner_radius_geom(self.image_bgr, tuple(self.sel))

    # -- drawing ------------------------------------------------------------

    def _on_draw(self, _w, cr):
        if self.pixbuf is None:
            return False

        cr.save()
        cr.scale(self.zoom, self.zoom)
        Gdk.cairo_set_source_pixbuf(cr, self.pixbuf, 0, 0)
        if cairo is not None:
            # Crisp pixels when zoomed in, smooth when zoomed out.
            flt = cairo.FILTER_NEAREST if self.zoom >= 1.0 else cairo.FILTER_GOOD
            cr.get_source().set_filter(flt)
        cr.paint()
        cr.restore()

        if self.sel is None:
            return False

        x, y, w, h = self.sel
        wx, wy = self._to_widget(x, y)
        ww, wh = w * self.zoom, h * self.zoom
        img_w, img_h = self._img_size()
        full_w, full_h = img_w * self.zoom, img_h * self.zoom
        r = self.corner_radius * self.zoom  # geometry radius in widget px

        # Dim everything OUTSIDE the rounded-rectangle geometry, so what's shown
        # bright is exactly what will be kept (corners included). Uses an
        # even-odd fill of the viewport minus the rounded selection path.
        if cairo is not None:
            cr.set_fill_rule(cairo.FILL_RULE_EVEN_ODD)
            cr.rectangle(0, 0, full_w, full_h)
            self._rounded_path(cr, wx, wy, ww, wh, r)
            cr.set_source_rgba(0, 0, 0, 0.5)
            cr.fill()
            cr.set_fill_rule(cairo.FILL_RULE_WINDING)
        else:  # fallback: plain rectangular dim
            cr.set_source_rgba(0, 0, 0, 0.5)
            cr.rectangle(0, 0, full_w, wy)
            cr.rectangle(0, wy + wh, full_w, full_h - (wy + wh))
            cr.rectangle(0, wy, wx, wh)
            cr.rectangle(wx + ww, wy, full_w - (wx + ww), wh)
            cr.fill()

        # Geometry border (rounded).
        cr.set_line_width(1.5)
        cr.set_source_rgba(0.15, 0.6, 1.0, 1.0)
        self._rounded_path(cr, wx + 0.5, wy + 0.5, ww, wh, r)
        cr.stroke()

        # Resize handles.
        for _hid, hx, hy, hhw, hhh in self._handle_rects_widget():
            cr.set_source_rgba(1, 1, 1, 1)
            cr.rectangle(hx, hy, hhw, hhh)
            cr.fill()
            cr.set_source_rgba(0.15, 0.6, 1.0, 1.0)
            cr.set_line_width(1.0)
            cr.rectangle(hx + 0.5, hy + 0.5, hhw - 1, hhh - 1)
            cr.stroke()
        return False

    @staticmethod
    def _rounded_path(cr, x, y, w, h, r):
        """Add a rounded-rectangle sub-path to the cairo context."""
        r = max(0.0, min(r, w / 2.0, h / 2.0))
        if r <= 0.5:
            cr.rectangle(x, y, w, h)
            return
        cr.new_sub_path()
        cr.arc(x + w - r, y + r, r, -math.pi / 2, 0)
        cr.arc(x + w - r, y + h - r, r, 0, math.pi / 2)
        cr.arc(x + r, y + h - r, r, math.pi / 2, math.pi)
        cr.arc(x + r, y + r, r, math.pi, 1.5 * math.pi)
        cr.close_path()

    # -- signalling ---------------------------------------------------------

    def _emit_changed(self, status=None, radius=None):
        if self._on_selection_changed:
            self._on_selection_changed(status, radius)


class MainWindow(Gtk.ApplicationWindow):
    def __init__(self, app):
        super().__init__(application=app, title="Window Extractor")
        self.set_default_size(1000, 720)
        self.set_icon_name("applets-screenshooter")

        self._build_ui()
        self._connect_accels(app)
        self._update_actions()

    # -- UI construction ----------------------------------------------------

    def _build_ui(self):
        header = Gtk.HeaderBar()
        header.set_show_close_button(True)
        header.props.title = "Window Extractor"
        self.set_titlebar(header)

        open_btn = Gtk.Button.new_from_icon_name(
            "document-open-symbolic", Gtk.IconSize.BUTTON
        )
        open_btn.set_tooltip_text("Open a screenshot… (Ctrl+O)")
        open_btn.connect("clicked", lambda _b: self.on_open())
        header.pack_start(open_btn)

        paste_btn = Gtk.Button.new_from_icon_name(
            "edit-paste-symbolic", Gtk.IconSize.BUTTON
        )
        paste_btn.set_tooltip_text("Paste screenshot from clipboard (Ctrl+V)")
        paste_btn.connect("clicked", lambda _b: self.on_paste())
        header.pack_start(paste_btn)

        # Mode toggle: Detect vs Select.
        mode_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
        mode_box.get_style_context().add_class("linked")
        self.detect_btn = Gtk.ToggleButton(label="Detect")
        self.detect_btn.set_tooltip_text(
            "Best-effort: click inside a window to guess its boundary, then "
            "adjust. Unreliable on busy/overlapping desktops — prefer Select."
        )
        self.select_btn = Gtk.ToggleButton(label="Select")
        self.select_btn.set_tooltip_text(
            "Drag a box around the window — it snaps to the edges. The reliable "
            "way; fine-tune with the handles."
        )
        # Select is the default: assisted-manual is the dependable path.
        self.select_btn.set_active(True)
        self.detect_btn.connect("toggled", self._on_mode_toggled, "detect")
        self.select_btn.connect("toggled", self._on_mode_toggled, "select")
        mode_box.pack_start(self.detect_btn, False, False, 0)
        mode_box.pack_start(self.select_btn, False, False, 0)
        header.pack_start(mode_box)

        # Snap the current selection onto window edges, on demand.
        self.snap_btn = Gtk.Button.new_from_icon_name(
            "zoom-fit-best-symbolic", Gtk.IconSize.BUTTON
        )
        self.snap_btn.set_label("Snap")
        self.snap_btn.set_always_show_image(True)
        self.snap_btn.set_tooltip_text(
            "Snap the current selection onto the window's edges (S)"
        )
        self.snap_btn.connect("clicked", lambda _b: self.on_snap())
        header.pack_start(self.snap_btn)

        # Export actions on the right.
        self.copy_btn = Gtk.Button.new_from_icon_name(
            "edit-copy-symbolic", Gtk.IconSize.BUTTON
        )
        self.copy_btn.set_tooltip_text("Copy the window to the clipboard (Ctrl+C)")
        self.copy_btn.connect("clicked", lambda _b: self.on_copy())
        header.pack_end(self.copy_btn)

        self.save_btn = Gtk.Button.new_from_icon_name(
            "document-save-symbolic", Gtk.IconSize.BUTTON
        )
        self.save_btn.set_tooltip_text("Save the window as a PNG… (Ctrl+S)")
        self.save_btn.connect("clicked", lambda _b: self.on_save())
        header.pack_end(self.save_btn)

        # Options popover (sensitivity + corner radius + zoom).
        menu_btn = Gtk.MenuButton()
        menu_btn.set_image(
            Gtk.Image.new_from_icon_name(
                "open-menu-symbolic", Gtk.IconSize.BUTTON
            )
        )
        menu_btn.set_popover(self._build_options_popover())
        header.pack_end(menu_btn)

        # Main vertical box: canvas + status bar.
        vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        self.add(vbox)

        self.scrolled = Gtk.ScrolledWindow()
        self.scrolled.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
        vbox.pack_start(self.scrolled, True, True, 0)

        self.canvas = ImageCanvas(self._on_selection_changed)
        self.canvas.set_halign(Gtk.Align.CENTER)
        self.canvas.set_valign(Gtk.Align.CENTER)
        self.scrolled.add(self.canvas)

        # Empty-state hint drawn as an overlay label.
        self.statusbar = Gtk.Label()
        self.statusbar.set_xalign(0.0)
        self.statusbar.set_margin_start(8)
        self.statusbar.set_margin_end(8)
        self.statusbar.set_margin_top(4)
        self.statusbar.set_margin_bottom(4)
        self.statusbar.set_line_wrap(True)
        vbox.pack_start(self.statusbar, False, False, 0)

        self._set_status(
            "Open or paste a screenshot (Ctrl+O / Ctrl+V), then drag a box "
            "around a window — it snaps to the edges."
        )

    def _build_options_popover(self):
        pop = Gtk.Popover()
        grid = Gtk.Grid()
        grid.set_row_spacing(10)
        grid.set_column_spacing(10)
        grid.set_border_width(12)

        # Detection sensitivity.
        lbl = Gtk.Label(label="Detection sensitivity")
        lbl.set_xalign(0.0)
        grid.attach(lbl, 0, 0, 2, 1)
        # Slider maps to min_area_frac: left = grabs small regions,
        # right = only large windows. We invert so "more sensitive" is right.
        self.sens_scale = Gtk.Scale.new_with_range(
            Gtk.Orientation.HORIZONTAL, 0.001, 0.15, 0.001
        )
        self.sens_scale.set_value(0.01)
        self.sens_scale.set_draw_value(False)
        self.sens_scale.set_hexpand(True)
        self.sens_scale.set_size_request(220, -1)
        self.sens_scale.set_tooltip_text(
            "Smallest region the detector will accept, as a share of the "
            "image. Lower = grabs smaller regions; higher = only big windows."
        )
        self.sens_scale.connect("value-changed", self._on_sens_changed)
        grid.attach(self.sens_scale, 0, 1, 2, 1)

        # Corner radius.
        lbl2 = Gtk.Label(label="Rounded corners (px)")
        lbl2.set_xalign(0.0)
        grid.attach(lbl2, 0, 2, 1, 1)
        self.radius_spin = Gtk.SpinButton.new_with_range(0, 80, 1)
        self.radius_spin.set_value(0)
        self.radius_spin.set_tooltip_text(
            "Corner radius of the cutout geometry, so a rounded window's corners "
            "are cut cleanly. Measured for you when you snap a selection; adjust "
            "here if a corner looks square or over-rounded."
        )
        self.radius_spin.connect("value-changed", self._on_radius_changed)
        grid.attach(self.radius_spin, 1, 2, 1, 1)

        # Snap toggle: snap a hand-drawn box onto window edges on release.
        self.snap_check = Gtk.CheckButton(label="Snap drawn box to edges")
        self.snap_check.set_active(True)
        self.snap_check.set_tooltip_text(
            "When you drag a selection in Select mode, snap its sides onto the "
            "window's real edges. Turn off to place the box entirely by hand."
        )
        self.snap_check.connect(
            "toggled", lambda b: setattr(self.canvas, "snap_enabled", b.get_active())
        )
        grid.attach(self.snap_check, 0, 3, 2, 1)

        # Select whole image (for full-screen / maximized windows whose outer
        # edge is not present in the screenshot).
        whole_btn = Gtk.Button(label="Select whole image  (Ctrl+A)")
        whole_btn.set_tooltip_text(
            "Select the entire screenshot — use this for a maximized / "
            "full-screen window, which has no detectable outer border."
        )
        whole_btn.connect("clicked", lambda _b: self.on_select_all())
        grid.attach(whole_btn, 0, 4, 2, 1)

        # Zoom controls.
        sep = Gtk.Separator(orientation=Gtk.Orientation.HORIZONTAL)
        grid.attach(sep, 0, 5, 2, 1)
        zoom_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        zoom_box.get_style_context().add_class("linked")
        for label, cb in (
            ("−", lambda _b: self.canvas.set_zoom(self.canvas.zoom / 1.25)),
            ("Fit", lambda _b: self._zoom_fit()),
            ("1:1", lambda _b: self.canvas.set_zoom(1.0)),
            ("+", lambda _b: self.canvas.set_zoom(self.canvas.zoom * 1.25)),
        ):
            b = Gtk.Button(label=label)
            b.connect("clicked", cb)
            zoom_box.pack_start(b, True, True, 0)
        grid.attach(zoom_box, 0, 6, 2, 1)

        ver = Gtk.Label()
        ver.set_markup(
            f"<small>Window Extractor {__version__}</small>"
        )
        ver.set_xalign(0.0)
        grid.attach(ver, 0, 7, 2, 1)

        grid.show_all()
        pop.add(grid)
        return pop

    def _connect_accels(self, app):
        accel = Gtk.AccelGroup()
        self.add_accel_group(accel)
        mappings = [
            ("o", Gdk.ModifierType.CONTROL_MASK, self.on_open),
            ("v", Gdk.ModifierType.CONTROL_MASK, self.on_paste),
            ("s", Gdk.ModifierType.CONTROL_MASK, self.on_save),
            ("c", Gdk.ModifierType.CONTROL_MASK, self.on_copy),
            ("a", Gdk.ModifierType.CONTROL_MASK, self.on_select_all),
            ("s", 0, self.on_snap),
        ]
        for key, mods, cb in mappings:
            keyval = Gdk.keyval_from_name(key)
            accel.connect(
                keyval, mods, Gtk.AccelFlags.VISIBLE,
                lambda *a, cb=cb: (cb(), True)[1],
            )

    # -- option handlers ----------------------------------------------------

    def _on_mode_toggled(self, button, mode):
        if not button.get_active():
            return
        # Keep the two toggles mutually exclusive.
        self.canvas.mode = mode
        if mode == "detect":
            self.select_btn.set_active(False)
        else:
            self.detect_btn.set_active(False)

    def _on_sens_changed(self, scale):
        self.canvas.sensitivity = scale.get_value()

    def _on_radius_changed(self, spin):
        self.canvas.corner_radius = int(spin.get_value())
        self.canvas.queue_draw()  # live-update the rounded preview

    def _zoom_fit(self):
        alloc = self.scrolled.get_allocation()
        self.canvas.fit_to(alloc.width - 24, alloc.height - 24)

    # -- file / clipboard actions ------------------------------------------

    def on_open(self):
        dialog = Gtk.FileChooserDialog(
            title="Open a screenshot",
            parent=self,
            action=Gtk.FileChooserAction.OPEN,
        )
        dialog.add_buttons(
            "_Cancel", Gtk.ResponseType.CANCEL,
            "_Open", Gtk.ResponseType.ACCEPT,
        )
        flt = Gtk.FileFilter()
        flt.set_name("Images")
        for pat in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp", "*.tiff"):
            flt.add_pattern(pat)
        dialog.add_filter(flt)
        allf = Gtk.FileFilter()
        allf.set_name("All files")
        allf.add_pattern("*")
        dialog.add_filter(allf)

        if dialog.run() == Gtk.ResponseType.ACCEPT:
            path = dialog.get_filename()
            dialog.destroy()
            self._load_path(path)
        else:
            dialog.destroy()

    def _load_path(self, path):
        try:
            pixbuf = GdkPixbuf.Pixbuf.new_from_file(path)
        except GLib.Error as exc:
            self._error(f"Could not open image:\n{exc.message}")
            return
        # Drop any alpha/rotation quirks by flattening orientation.
        pixbuf = pixbuf.apply_embedded_orientation() or pixbuf
        self.canvas.set_image_from_pixbuf(pixbuf)
        GLib.idle_add(self._zoom_fit)
        self._set_status(
            f"Loaded {os.path.basename(path)} — drag a box around a window "
            "(it snaps to the edges), then Save or Copy."
        )

    def on_paste(self):
        clipboard = Gtk.Clipboard.get(Gdk.SELECTION_CLIPBOARD)
        pixbuf = clipboard.wait_for_image()
        if pixbuf is None:
            self._error(
                "The clipboard does not contain an image.\n\n"
                "Take a screenshot to the clipboard (in GNOME, press "
                "Shift+PrintScreen and pick an area/window) and try again."
            )
            return
        self.canvas.set_image_from_pixbuf(pixbuf)
        GLib.idle_add(self._zoom_fit)
        self._set_status(
            "Pasted screenshot — drag a box around a window (it snaps to the "
            "edges), then Save or Copy."
        )

    def on_save(self):
        bgra = self.canvas.extract_bgra()
        if bgra is None:
            self._set_status("Nothing selected yet.")
            return
        dialog = Gtk.FileChooserDialog(
            title="Save window as PNG",
            parent=self,
            action=Gtk.FileChooserAction.SAVE,
        )
        dialog.add_buttons(
            "_Cancel", Gtk.ResponseType.CANCEL,
            "_Save", Gtk.ResponseType.ACCEPT,
        )
        dialog.set_do_overwrite_confirmation(True)
        dialog.set_current_name("window.png")
        flt = Gtk.FileFilter()
        flt.set_name("PNG image")
        flt.add_pattern("*.png")
        dialog.add_filter(flt)

        if dialog.run() == Gtk.ResponseType.ACCEPT:
            path = dialog.get_filename()
            dialog.destroy()
            if not path.lower().endswith(".png"):
                path += ".png"
            try:
                imageutils.save_bgra_png(bgra, path)
                self._set_status(f"Saved {os.path.basename(path)}.")
            except Exception as exc:  # noqa: BLE001
                self._error(f"Could not save PNG:\n{exc}")
        else:
            dialog.destroy()

    def on_select_all(self):
        self.canvas.select_whole_image()

    def on_snap(self):
        if not self.canvas.has_selection():
            self._set_status("Draw a selection first, then Snap.")
            return
        self.canvas.snap_current()
        self._on_selection_changed(
            status="Snapped the selection to the window's edges.",
            radius=self.canvas.estimate_radius(),
        )

    def on_copy(self):
        bgra = self.canvas.extract_bgra()
        if bgra is None:
            self._set_status("Nothing selected yet.")
            return
        pixbuf = imageutils.bgra_to_pixbuf(bgra)
        clipboard = Gtk.Clipboard.get(Gdk.SELECTION_CLIPBOARD)
        clipboard.set_image(pixbuf)
        clipboard.store()
        self._set_status("Copied the window to the clipboard.")

    # -- helpers ------------------------------------------------------------

    def _on_selection_changed(self, status=None, radius=None):
        self._update_actions()
        if radius is not None:
            # Reflect the auto-measured corner radius in the control. Setting
            # the spin value fires _on_radius_changed, which updates
            # canvas.corner_radius -- the single source of truth for export.
            self.radius_spin.set_value(radius)
        if status:
            self._set_status(status)

    def _update_actions(self):
        has_sel = self.canvas.has_selection()
        self.save_btn.set_sensitive(has_sel)
        self.copy_btn.set_sensitive(has_sel)
        self.snap_btn.set_sensitive(has_sel)

    def _set_status(self, text):
        self.statusbar.set_text(text)

    def _error(self, message):
        dialog = Gtk.MessageDialog(
            transient_for=self,
            modal=True,
            message_type=Gtk.MessageType.ERROR,
            buttons=Gtk.ButtonsType.OK,
            text="Window Extractor",
        )
        dialog.format_secondary_text(message)
        dialog.run()
        dialog.destroy()


class WindowExtractorApp(Gtk.Application):
    def __init__(self):
        super().__init__(
            application_id=APP_ID,
            flags=Gio.ApplicationFlags.HANDLES_OPEN,
        )
        self._files = []

    def do_activate(self):
        win = self._ensure_window()
        win.show_all()
        win.present()

    def do_open(self, files, n_files, hint):
        win = self._ensure_window()
        win.show_all()
        win.present()
        if files:
            path = files[0].get_path()
            if path:
                GLib.idle_add(win._load_path, path)

    def _ensure_window(self):
        win = self.get_active_window()
        if win is None:
            win = MainWindow(self)
        return win


def main(argv=None):
    import sys

    app = WindowExtractorApp()
    return app.run(sys.argv if argv is None else argv)
