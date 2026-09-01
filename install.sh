#!/usr/bin/env bash
#
# Install Window Extractor on Debian/Ubuntu (GNOME).
#
# It installs the runtime dependencies from apt, drops a launcher into
# ~/.local/bin, and registers a GNOME desktop entry so the app shows up in
# the Activities overview. No pip, no virtualenv required.
#
# Usage:  ./install.sh
#         ./install.sh --uninstall
#
set -euo pipefail

APP_NAME="window-extractor"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN_DIR="${HOME}/.local/bin"
LAUNCHER="${BIN_DIR}/${APP_NAME}"
DESKTOP_DIR="${HOME}/.local/share/applications"
DESKTOP_FILE="${DESKTOP_DIR}/com.veros.WindowExtractor.desktop"

uninstall() {
    echo "Removing launcher and desktop entry…"
    rm -f "$LAUNCHER" "$DESKTOP_FILE"
    command -v update-desktop-database >/dev/null 2>&1 && \
        update-desktop-database "$DESKTOP_DIR" 2>/dev/null || true
    echo "Done. (System packages installed via apt were left in place.)"
    exit 0
}

[[ "${1:-}" == "--uninstall" ]] && uninstall

echo "==> Installing system dependencies (requires sudo)…"
if command -v apt >/dev/null 2>&1; then
    sudo apt update
    sudo apt install -y \
        python3-gi python3-gi-cairo gir1.2-gtk-3.0 \
        python3-opencv python3-numpy
else
    echo "!! apt not found. Install the equivalents of:" >&2
    echo "   python3-gi python3-gi-cairo gir1.2-gtk-3.0 python3-opencv python3-numpy" >&2
    exit 1
fi

echo "==> Installing launcher at ${LAUNCHER}"
mkdir -p "$BIN_DIR"
cat > "$LAUNCHER" <<EOF
#!/usr/bin/env bash
exec python3 -m windowextractor "\$@"
EOF
# Run from the repo so the package is importable without pip-installing.
sed -i "1a cd \"${REPO_DIR}\"" "$LAUNCHER"
chmod +x "$LAUNCHER"

echo "==> Installing desktop entry at ${DESKTOP_FILE}"
mkdir -p "$DESKTOP_DIR"
sed "s|@EXEC@|${LAUNCHER}|g" "${REPO_DIR}/com.veros.WindowExtractor.desktop.in" \
    > "$DESKTOP_FILE"
command -v update-desktop-database >/dev/null 2>&1 && \
    update-desktop-database "$DESKTOP_DIR" 2>/dev/null || true

echo
echo "Installed. Launch it from the Activities overview ('Window Extractor')"
echo "or run '${APP_NAME}' from a terminal."
if ! echo ":$PATH:" | grep -q ":${BIN_DIR}:"; then
    echo
    echo "Note: ${BIN_DIR} is not on your PATH. Add this to ~/.bashrc:"
    echo "  export PATH=\"\$HOME/.local/bin:\$PATH\""
fi
