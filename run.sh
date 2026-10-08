#!/usr/bin/env bash
# Launch RVC-Realtime-GUI from the project's virtual environment.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
exec .venv/bin/python -I app/RVC-Realtime-GUI.py "$@"
