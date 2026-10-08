#!/usr/bin/env bash
# Launch the COSMIC front end; it starts the Python engine from .venv itself.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
binary=ui/target/release/rvc-realtime
if [[ ! -x $binary ]]; then
    echo "Building the COSMIC front end (first run only)…" >&2
    cargo build --release --manifest-path ui/Cargo.toml
fi
RVC_PROJECT_DIR="$PWD" exec "$binary" "$@"
