#!/usr/bin/env bash
# Run Window Extractor from a checkout without installing it.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"
exec python3 -m windowextractor "$@"
