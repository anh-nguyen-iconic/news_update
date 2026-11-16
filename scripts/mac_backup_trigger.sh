#!/bin/bash
# Backup trigger for the GitHub schedule (GitHub often starts scheduled runs hours late).
# Runs on the Mac via launchd (see install_mac_backup.sh). On weekdays, if today's daily digest is not on
# GitHub yet and no run is in progress, it starts the "Daily digest" workflow with force=false, so it can
# never cause a second email (the workflow also skips itself when today's digest already exists).
set -uo pipefail

REPO="${NEWS_AGENT_REPO:-anh-nguyen-iconic/news_update}"
TZ_LOCAL="Asia/Ho_Chi_Minh"   # must match the digest's timezone (NEWS_AGENT_TZ default)
export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin"

log() { echo "$(date '+%F %T') $*"; }

today=$(TZ=$TZ_LOCAL date +%F)
if [ "$(TZ=$TZ_LOCAL date +%u)" -gt 5 ]; then
  log "$today is a weekend - nothing to do"; exit 0
fi

# Reuse the GitHub login git already stores in the macOS keychain (no extra token on disk).
GH_TOKEN=$(printf 'protocol=https\nhost=github.com\n\n' | GIT_TERMINAL_PROMPT=0 git credential fill 2>/dev/null \
  | sed -n 's/^password=//p')
if [ -z "$GH_TOKEN" ]; then
  log "ERROR: no GitHub credential in the keychain (push to the repo once to store it)"; exit 1
fi
export GH_TOKEN

# After waking from sleep the network can take a moment: retry for up to ~5 minutes.
for attempt in 1 2 3 4 5 6; do
  err=$(gh api "repos/$REPO/contents/digests/daily/$today.html" --jq .name 2>&1 >/dev/null)
  if [ $? -eq 0 ]; then
    log "today's digest ($today) is already sent - nothing to do"; exit 0
  fi
  if echo "$err" | grep -q "HTTP 404"; then
    break  # not sent yet
  fi
  log "GitHub not reachable yet (attempt $attempt): ${err:0:120}"
  sleep 50
done
if [ "${attempt:-0}" -eq 6 ] && ! echo "${err:-}" | grep -q "HTTP 404"; then
  log "ERROR: could not reach GitHub"; exit 1
fi

active=$(gh run list -R "$REPO" --workflow daily.yml --limit 5 --json status \
  -q '[.[] | select(.status != "completed")] | length' 2>/dev/null || echo 0)
if [ "${active:-0}" -gt 0 ]; then
  log "a Daily digest run is already queued/running - nothing to do"; exit 0
fi

if gh workflow run daily.yml -R "$REPO" -f force=false -f dry_run=false; then
  log "today's digest ($today) was not sent yet - started the Daily digest workflow"
else
  log "ERROR: failed to start the workflow"; exit 1
fi
