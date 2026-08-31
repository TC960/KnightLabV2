#!/usr/bin/env bash
# Bulletproof guardrail for a Brev GPU extraction run.
#
# Runs LOCALLY, independent of any Claude session. Every 60s it:
#   1. copies the remote checkpoint down (incremental — partial results always survive)
#   2. checks for a DONE / FAILED marker
#   3. DELETES the instance when the job finishes, fails, or the deadline passes
#
# Deletion, not stop: massedcompute instances do not support stop, so delete is the
# only way to halt billing. That is why results are copied down BEFORE the check
# and on every tick, not just at the end.
#
# Three independent stop conditions, so no single failure leaves the GPU billing:
#   - job wrote DONE            -> copy, delete
#   - job wrote FAILED / died   -> copy, delete
#   - HARD_DEADLINE reached     -> copy, delete regardless of state
#
# Usage: watchdog.sh <instance-name> <max-hours>
set -uo pipefail
INST="${1:?instance name required}"
MAX_HOURS="${2:-6}"
BASE="$HOME/.brev-watchdog"
LOG="$BASE/$INST.log"
DEST="/Users/mohak/Desktop/Lab Work/proj_2_attempt3/kg/gpu_results"
REMOTE_DIR="~/kg/repo/proj_2_attempt3/kg"
mkdir -p "$BASE" "$DEST"

DEADLINE=$(( $(date +%s) + MAX_HOURS * 3600 ))
say(){ echo "[$(date '+%H:%M:%S')] $*" >>"$LOG"; }
ssh_q(){ ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no "$INST" "$1" 2>/dev/null; }

say "=== watchdog start: $INST, max ${MAX_HOURS}h, deadline $(date -r $DEADLINE '+%H:%M') ==="

kill_instance(){
  local why="$1"
  say "STOPPING INSTANCE ($why)"
  # final copy before the box goes away
  scp -o ConnectTimeout=15 -o BatchMode=yes -r "$INST:$REMOTE_DIR/extract_out" "$DEST/" 2>/dev/null
  scp -o ConnectTimeout=15 -o BatchMode=yes "$INST:~/kg/run.log" "$DEST/gpu_run.log" 2>/dev/null
  # stop if the provider supports it, else delete (massedcompute is delete-only)
  if brev stop "$INST" 2>&1 | grep -qi "does not support stop"; then
    say "provider has no stop; deleting"
    brev delete "$INST" >>"$LOG" 2>&1
  else
    say "stop issued"
  fi
  sleep 20
  local st
  st=$(brev ls 2>/dev/null | awk -v n="$INST" '$1==n {print $2}')
  say "final state: ${st:-GONE}"
  if [ -n "$st" ] && [ "$st" != "STOPPED" ] && [ "$st" != "DELETING" ]; then
    say "still up — forcing delete"
    brev delete "$INST" >>"$LOG" 2>&1
  fi
  say "=== watchdog done ==="
  exit 0
}

MISSES=0
while true; do
  NOW=$(date +%s)
  [ "$NOW" -ge "$DEADLINE" ] && kill_instance "hard deadline ${MAX_HOURS}h"

  # incremental pull — partial results survive any failure
  scp -o ConnectTimeout=15 -o BatchMode=yes -r "$INST:$REMOTE_DIR/extract_out" "$DEST/" 2>/dev/null

  STATE=$(brev ls 2>/dev/null | awk -v n="$INST" '$1==n {print $2}')
  if [ -z "$STATE" ]; then
    say "instance not in brev ls — already gone"; exit 0
  fi

  # Liveness must be probed with something that ALWAYS answers on a healthy box.
  # Probing for STATUS/checkpoint instead would count the whole provisioning phase
  # (~15 min of apt + pip + a 16GB model download) as "unreachable" and kill a
  # perfectly healthy instance before the job ever starts.
  ALIVE=$(ssh_q "echo alive")
  MARK=$(ssh_q "cat $REMOTE_DIR/STATUS 2>/dev/null")
  N=$(ssh_q "wc -l < $REMOTE_DIR/extract_out/checkpoint.jsonl 2>/dev/null" | tr -dc '0-9')
  if [ -z "$ALIVE" ]; then
    MISSES=$((MISSES+1))
    say "unreachable ($MISSES/20) state=$STATE"
    # 20 consecutive misses ~ 20 min of silence: assume dead, do not keep paying
    [ "$MISSES" -ge 20 ] && kill_instance "unreachable for 20 consecutive checks"
  else
    MISSES=0
    say "state=$STATE papers=${N:-0} marker=${MARK:-provisioning/running}"
    case "$MARK" in
      DONE*)   kill_instance "job reported DONE" ;;
      FAILED*) kill_instance "job reported FAILED" ;;
    esac
  fi
  sleep 60
done
