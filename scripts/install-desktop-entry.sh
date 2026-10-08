#!/usr/bin/env bash
# Add RVC Realtime to the application launcher for the current user.
set -euo pipefail
project_dir="$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)"
app_id=io.github.rvc_realtime.RvcRealtime
target="${XDG_DATA_HOME:-$HOME/.local/share}/applications/$app_id.desktop"
mkdir -p "$(dirname "$target")"
cat > "$target" <<DESKTOP
[Desktop Entry]
Type=Application
Name=RVC Realtime
Comment=Real-time RVC voice conversion
Exec="$project_dir/run.sh"
Icon=audio-input-microphone
Terminal=false
Categories=AudioVideo;Audio;
StartupWMClass=$app_id
DESKTOP
echo "Installed $target"
