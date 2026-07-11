#!/usr/bin/env bash
# Sequential eval across models (single GPU -> one at a time, no parallel).
# Each model: download GGUF to /tmp, run samgated-v1 prompt on test_set_v2,
# score, append to results/leaderboard.csv, then delete the GGUF from cache.
set -u
cd "$(dirname "$0")"
export HF_HOME=/tmp/hf TMPDIR=/tmp
PY=/tmp/venv/bin/python
mkdir -p results
LOG=results/_runlog.txt

MODELS=(
  qwen2.5-32b-instruct      # clean baseline (Apache-2.0)
  qwopus3.5-27b-v3          # the previous distill, now on the gated prompt
  qwen3.6-35b-a3b           # clean Qwen3.6 MoE (35B total / 3B active)
  qwythos-9b                # small distill
  qwopus3.6-35b-a3b-mtp     # newest Qwopus MoE (MTP)
)

echo "=== run_all started $(date) ===" | tee -a "$LOG"
for m in "${MODELS[@]}"; do
  echo ""            | tee -a "$LOG"
  echo ">>> $m"      | tee -a "$LOG"
  $PY run_eval.py --model "$m" 2>&1 | tee -a "$LOG"
done
echo ""                                 | tee -a "$LOG"
echo "=== run_all done $(date) ==="     | tee -a "$LOG"
echo ""
echo "================ LEADERBOARD ================"
cat results/leaderboard.csv
