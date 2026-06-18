#!/usr/bin/env bash
# Runs from cron on the VPS every 15 min. Supervises the *current* SPRT only —
# it never authors new search code. When a test concludes (or times out) it
# promotes/rejects mechanically and hands off to a human (STATE=NEEDS_HUMAN)
# rather than guessing what to test next.
set -uo pipefail
cd "$(dirname "$0")/.."   # -> /root/littleindian

STATUS=STATUS.txt
LOCKDIR=training/sprt_bloco6   # holds out.log / sprt.pgn for the active test
MAX_HOURS=48                   # backstop if cutechess's own SPRT bound never trips

log() { echo "[$(date -Is)] $*" >> training/autoloop.log; }

read_status() { grep -m1 "^STATE=" "$STATUS" 2>/dev/null | cut -d= -f2; }

state="$(read_status)"
[ -z "$state" ] && { log "no $STATUS, nothing to do"; exit 0; }

if [ "$state" = "NEEDS_HUMAN" ] || [ "$state" = "IDLE" ]; then
    exit 0   # waiting on a person; don't spam the log every 15 min
fi

if [ "$state" != "TESTING" ]; then
    log "unknown STATE=$state, leaving alone"
    exit 0
fi

candidate="$(grep -m1 '^CANDIDATE=' "$STATUS" | cut -d= -f2)"
base="$(grep -m1 '^BASE=' "$STATUS" | cut -d= -f2)"
started="$(grep -m1 '^STARTED=' "$STATUS" | cut -d= -f2)"
outlog="$LOCKDIR/out.log"

running=$(pgrep -f "cutechess-cli.*candidate_$candidate" || true)

if [ -n "$running" ]; then
    # still going — only intervene if it's run past the wall-clock backstop
    if [ -n "$started" ]; then
        elapsed_h=$(( ( $(date +%s) - $(date -d "$started" +%s) ) / 3600 ))
        if [ "$elapsed_h" -ge "$MAX_HOURS" ]; then
            log "candidate_$candidate vs $base: ${elapsed_h}h elapsed, no SPRT bound hit — stopping, flagging for human"
            kill $running 2>/dev/null
            sleep 5
            kill -9 $running 2>/dev/null
            {
                echo "STATE=NEEDS_HUMAN"
                echo "CANDIDATE=$candidate"
                echo "BASE=$base"
                echo "RESULT=INCONCLUSIVE_TIMEOUT"
                echo "NOTE=Ran ${elapsed_h}h without crossing SPRT bound. Check $outlog by hand — could be a genuinely flat result (see feedback_universal_techniques memory: some universal techniques are kept despite flat/negative isolated SPRT, but that is the owner's call, not this script's)."
            } > "$STATUS"
        fi
    fi
    exit 0
fi

# process has exited — see how
verdict=$(grep -E "H0 was accepted|H1 was accepted" "$outlog" | tail -1)
crash=$(grep -E "Cannot start engine|Cannot execute command" "$outlog" | tail -1)

if [ -n "$crash" ]; then
    log "candidate_$candidate vs $base: crashed before a verdict ($crash) — flagging"
    {
        echo "STATE=NEEDS_HUMAN"
        echo "CANDIDATE=$candidate"
        echo "BASE=$base"
        echo "RESULT=CRASHED"
        echo "NOTE=$crash"
    } > "$STATUS"
    exit 0
fi

if echo "$verdict" | grep -q "H1 was accepted"; then
    log "candidate_$candidate vs $base: H1 accepted ($verdict) — promoting"
    mkdir -p archive
    mv "littleindian.$base" "archive/littleindian.${base}_$(date +%Y%m%d_%H%M%S)" 2>/dev/null
    cp "littleindian.candidate_$candidate" "littleindian.$base"
    git add -A
    git commit -q -m "autoloop: promote candidate_$candidate to $base ($verdict)"
    {
        echo "STATE=NEEDS_HUMAN"
        echo "CANDIDATE=$candidate"
        echo "BASE=$base"
        echo "RESULT=PROMOTED"
        echo "NOTE=$verdict — littleindian.$base updated. No further pre-built candidate queued; next step is authoring a new feature with a human driving."
    } > "$STATUS"
elif echo "$verdict" | grep -q "H0 was accepted"; then
    log "candidate_$candidate vs $base: H0 accepted ($verdict) — rejecting"
    mkdir -p archive
    mv "littleindian.candidate_$candidate" "archive/littleindian.candidate_${candidate}_rejected_$(date +%Y%m%d_%H%M%S)" 2>/dev/null
    {
        echo "STATE=NEEDS_HUMAN"
        echo "CANDIDATE=$candidate"
        echo "BASE=$base"
        echo "RESULT=REJECTED"
        echo "NOTE=$verdict — $base unchanged. candidate_$candidate archived (not deleted) in case some piece of it is worth re-testing in isolation."
    } > "$STATUS"
else
    log "candidate_$candidate vs $base: process exited, no SPRT verdict line and no crash match — flagging for manual look"
    {
        echo "STATE=NEEDS_HUMAN"
        echo "CANDIDATE=$candidate"
        echo "BASE=$base"
        echo "RESULT=UNKNOWN"
        echo "NOTE=cutechess-cli exited without a recognizable verdict in $outlog. Inspect by hand."
    } > "$STATUS"
fi
