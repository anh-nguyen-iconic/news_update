#!/bin/bash
# Install (or remove) the Mac backup trigger: a launchd job at 09:00 Mon-Fri (Mac local time) that runs
# mac_backup_trigger.sh. Usage:  scripts/install_mac_backup.sh [--uninstall]
#
# The script is copied to ~/Library/Application Support/news-agent/ because macOS privacy protection
# blocks background jobs from reading files under ~/Documents. Re-run this installer after editing it.
# If the Mac is asleep at 09:00, launchd runs the job when it wakes; if it is shut down, the run is skipped.
set -euo pipefail

LABEL="com.news-agent.daily-backup"
HOUR="${NEWS_AGENT_BACKUP_HOUR:-9}"
MINUTE="${NEWS_AGENT_BACKUP_MINUTE:-0}"
SRC="$(cd "$(dirname "$0")" && pwd)/mac_backup_trigger.sh"
DEST_DIR="$HOME/Library/Application Support/news-agent"
DEST="$DEST_DIR/daily_backup.sh"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
LOG="$HOME/Library/Logs/news-agent-backup.log"

launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true

if [ "${1:-}" = "--uninstall" ]; then
  rm -f "$PLIST" "$DEST"
  echo "Removed $LABEL"; exit 0
fi

mkdir -p "$DEST_DIR" "$(dirname "$PLIST")" "$(dirname "$LOG")"
install -m 755 "$SRC" "$DEST"

intervals=""
for day in 1 2 3 4 5; do
  intervals+="    <dict><key>Weekday</key><integer>$day</integer><key>Hour</key><integer>$HOUR</integer><key>Minute</key><integer>$MINUTE</integer></dict>
"
done

cat > "$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key>
  <array><string>/bin/bash</string><string>$DEST</string></array>
  <key>StartCalendarInterval</key>
  <array>
$intervals  </array>
  <key>StandardOutPath</key><string>$LOG</string>
  <key>StandardErrorPath</key><string>$LOG</string>
</dict>
</plist>
EOF
plutil -lint "$PLIST" >/dev/null
launchctl bootstrap "gui/$(id -u)" "$PLIST"
printf 'Installed %s: Mon-Fri %02d:%02d (Mac local time). Log: %s\n' "$LABEL" "$HOUR" "$MINUTE" "$LOG"
echo "Test now with: launchctl kickstart gui/$(id -u)/$LABEL"
