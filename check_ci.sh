#!/usr/bin/env bash
# =============================================================================
# check_ci.sh
# ProbOS -- permanent CI watcher script
#
# USAGE:
#   bash check_ci.sh              -- watch the run for the CURRENT commit (HEAD)
#   bash check_ci.sh <run-id>     -- watch a specific run by ID
#
# WHY THE SHA-MATCHING (not just "most recent"):
#   Found via direct observation this session: immediately after a push,
#   `gh run list --limit 1` frequently returns the PREVIOUS commit's
#   already-completed run, since GitHub hasn't registered the new run
#   yet. Matching against the exact commit SHA (git rev-parse HEAD)
#   instead of "most recent by time" makes this structurally impossible
#   to get wrong, regardless of timing. No more need to manually sleep
#   before calling this script.
# =============================================================================

set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR" || exit 1

if ! gh auth status > /dev/null 2>&1; then
    echo "============================================================"
    echo "  gh CLI is not authenticated (token may have expired)"
    echo "============================================================"
    echo "  Run: gh auth login"
    echo "  Choose: GitHub.com -> HTTPS or SSH -> Paste an auth token"
    echo "  Generate a token at: https://github.com/settings/tokens"
    echo "  Minimum scopes needed: repo, read:org"
    echo "============================================================"
    exit 1
fi

RUN_ID="${1:-}"

if [ -z "$RUN_ID" ]; then
    HEAD_SHA="$(git rev-parse HEAD)"
    echo "============================================================"
    echo "  Looking up CI run for commit ${HEAD_SHA:0:7} (current HEAD)..."
    echo "============================================================"

    MAX_ATTEMPTS=30
    ATTEMPT=0
    RUN_ID=""

    while [ "$ATTEMPT" -lt "$MAX_ATTEMPTS" ]; do
        RUN_ID=$(gh run list --limit 10 --json databaseId,headSha \
            --jq ".[] | select(.headSha == \"$HEAD_SHA\") | .databaseId" \
            | head -1)

        if [ -n "$RUN_ID" ] && [ "$RUN_ID" != "null" ]; then
            break
        fi

        ATTEMPT=$((ATTEMPT + 1))
        if [ "$ATTEMPT" -eq 1 ]; then
            echo "  (waiting for GitHub to register the run for this commit...)"
        fi
        sleep 2
    done

    if [ -z "$RUN_ID" ] || [ "$RUN_ID" == "null" ]; then
        echo "ERROR: No CI run found for commit ${HEAD_SHA:0:7} after"
        echo "${MAX_ATTEMPTS} attempts (60 seconds). Has this commit been"
        echo "pushed? Does this workflow trigger on this branch/event?"
        exit 1
    fi
fi

echo "Watching run ID: $RUN_ID"
echo ""

gh run watch "$RUN_ID" --exit-status
WATCH_EXIT=$?

echo ""
echo "============================================================"

if [ "$WATCH_EXIT" -eq 0 ]; then
    echo "  CI RESULT: PASS -- all jobs succeeded"
    echo "============================================================"
    exit 0
else
    echo "  CI RESULT: FAIL -- at least one job failed"
    echo "============================================================"
    echo ""
    echo "  Showing failed job logs below:"
    echo "------------------------------------------------------------"
    gh run view "$RUN_ID" --log-failed
    echo "------------------------------------------------------------"
    echo ""
    echo "  Full run details: gh run view $RUN_ID"
    echo "  Open in browser:  gh run view $RUN_ID --web"
    exit 1
fi
