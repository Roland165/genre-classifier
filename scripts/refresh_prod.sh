#!/usr/bin/env bash
# Make production load the current champion after the registry alias moved
# (used by the CT and rollback workflows).
#
# - If ADMIN_TOKEN is set: POST /admin/reload (instant, same container).
# - Otherwise: Render deploy hook with the image that is already live
#   (its SHA is read from /model-info), so only the model changes, not the code.
#
# Needs: PROD_URL, and ADMIN_TOKEN or RENDER_DEPLOY_HOOK_URL.
set -euo pipefail

: "${PROD_URL:?PROD_URL is not set}"
BASE="${PROD_URL%/}"
# Render free tier may be asleep: retry while it wakes up
CURL=(curl -fsS --retry 12 --retry-delay 10 --retry-all-errors --max-time 120)

if [ -n "${ADMIN_TOKEN:-}" ]; then
  echo "Refreshing via /admin/reload"
  "${CURL[@]}" -X POST -H "X-Admin-Token: ${ADMIN_TOKEN}" "${BASE}/admin/reload"
  echo
else
  : "${RENDER_DEPLOY_HOOK_URL:?set ADMIN_TOKEN or RENDER_DEPLOY_HOOK_URL}"
  SHA=$("${CURL[@]}" "${BASE}/model-info" | python3 -c "import json, sys; print(json.load(sys.stdin)['git_sha'])")
  IMAGE="ghcr.io/${GITHUB_REPOSITORY,,}:${SHA}"
  echo "Refreshing via Render deploy hook, redeploying live image ${IMAGE}"
  ENC=$(python3 -c "import sys, urllib.parse; print(urllib.parse.quote(sys.argv[1], safe=''))" "$IMAGE")
  "${CURL[@]}" -X POST "${RENDER_DEPLOY_HOOK_URL}&imgURL=${ENC}" > /dev/null
fi
