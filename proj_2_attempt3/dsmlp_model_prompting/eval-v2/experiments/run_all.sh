#!/usr/bin/env bash
# Sequential experiment suite. Each run_*.py reads its LLM stage from cache/ (pilot: Fable-
# generated; production: swap in a GGUF/Qwopus batch), scores with the taxonomy-aware metric,
# and appends to results/experiments_leaderboard.csv.
set -e
cd "$(dirname "$0")"
export PYTHONPATH=/tmp/pylibs:$PYTHONPATH
PY=${PY:-/opt/conda/bin/python3}
rm -f results/experiments_leaderboard.csv
echo "=== building LLM-stage cache ==="; $PY build_cache.py
for r in run_0_original run_1_relate run_2_grounded run_3_normalize run_4_judge; do
  echo; echo "=== $r ==="; $PY $r.py
done
echo; echo "=== leaderboard ==="
$PY -c "import csv;rows=list(csv.reader(open('results/experiments_leaderboard.csv')));w=[max(len(r[i]) for r in rows) for i in range(len(rows[0]))];[print('  '.join(c.ljust(w[i]) for i,c in enumerate(r))) for r in rows]"
