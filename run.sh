#!/usr/bin/env bash
# Launch the COSMIC front end; it starts the Python engine from .venv itself.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
binary=ui/target/release/rvc-realtime
if [[ ! -x $binary ]]; then
    echo "Building the COSMIC front end (first run only)…" >&2
    cargo build --release --manifest-path ui/Cargo.toml
fi
# Ask PipeWire for a 960-sample period (20 ms at 48 kHz) while RVC runs: a
# 0.10 s chunk is then exactly 5 periods, so the output buffer stays level
# (fewer gaps, slightly less delay; see plan.md).  The graph runs at the
# smallest period any client requests, and returns to normal when RVC exits.
# Set PIPEWIRE_QUANTUM yourself to override.
export PIPEWIRE_QUANTUM="${PIPEWIRE_QUANTUM:-960/48000}"
RVC_PROJECT_DIR="$PWD" exec "$binary" "$@"
