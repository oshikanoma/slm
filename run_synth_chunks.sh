#!/usr/bin/env bash
# Generate synthetic data in independent chunks so a mid-run rate-limit (Groq free
# tier has a daily token cap) never wipes accepted progress. Each chunk writes its
# own file; we concatenate survivors at the end into data/train.synth.jsonl.
set -u
source /tmp/.slm_teacher_env
cd "$(dirname "$0")"

CHUNK=${CHUNK:-120}       # examples per chunk
N_CHUNKS=${N_CHUNKS:-6}   # up to CHUNK*N_CHUNKS total
mkdir -p data/synth_chunks

for i in $(seq 1 "$N_CHUNKS"); do
  out="data/synth_chunks/chunk_${i}.jsonl"
  if [ -s "$out" ]; then
    echo "[chunk $i] already exists, skipping"
    continue
  fi
  echo "[chunk $i/$N_CHUNKS] generating $CHUNK -> $out (seed $((100+i)))"
  python3 datagen.py --n "$CHUNK" --out "$out" --seed $((100+i)) --max-attempts $((CHUNK*3)) 2>&1 \
    | grep -E "accepted [0-9]+/|summary|bucket mix|WARNING" | tail -4
  # if a chunk produced nothing, the daily limit is likely hit — stop cleanly
  if [ ! -s "$out" ]; then
    echo "[chunk $i] produced 0 rows — likely daily token limit. Stopping."
    break
  fi
done

# concatenate all non-empty chunks
cat data/synth_chunks/chunk_*.jsonl > data/train.synth.jsonl 2>/dev/null
total=$(wc -l < data/train.synth.jsonl 2>/dev/null || echo 0)
echo "=== TOTAL SYNTHETIC: $total rows -> data/train.synth.jsonl ==="
