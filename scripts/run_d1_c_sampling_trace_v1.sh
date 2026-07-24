#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "$0")/.."

ROOT="$PWD"

DSFORMER_PY="$ROOT/.venv/bin/python"

ANNO="$ROOT/data/floodpsg/annotations/floodpsg_canonical_trainval_coarse8_groupstrict_v1.json"

HN_INDEX="$ROOT/data/floodpsg/stats/dsformer_floodhn_sampling_v1/train_hardneg_index_v1.json"

HOOK_DIR="$ROOT/audit_tools/d1_c_sampling_trace_v1"

TRACE_ROOT="$ROOT/data/floodpsg/stats/dsformer_fibe_d1_floodhn_r50_v1/sampling_trace_v1"

LOG_ROOT="$ROOT/data/floodpsg/logs/d1_c_sampling_trace_v1"

MODE="${1:-}"

case "$MODE" in
  C)
    RUN_ID="C_floodhn_target50"

    CONFIG="$ROOT/outputs/flood_groupstrict_coarse8_floodhn_r50_seed3407_40epoch_v1/config.json"

    OUTPUT="$ROOT/outputs/audit_c_floodhn_r50_seed3407_3epoch_sampling_trace_v1"
    ;;

  D1)
    RUN_ID="D1_floodhn_target50_fibe"

    CONFIG="$ROOT/configs/floodpsg/masks-loc-sem-flood-groupstrict-coarse8-fibe-d1-v1.json"

    OUTPUT="$ROOT/outputs/audit_d1_floodhn_r50_fibe_seed3407_3epoch_sampling_trace_v1"
    ;;

  *)
    echo "Usage:"
    echo "  bash scripts/run_d1_c_sampling_trace_v1.sh C"
    echo "  bash scripts/run_d1_c_sampling_trace_v1.sh D1"
    exit 2
    ;;
esac

TRACE_DIR="$TRACE_ROOT/$RUN_ID"
LOG="$LOG_ROOT/${RUN_ID}.log"

for path in \
  "$DSFORMER_PY" \
  "$ANNO" \
  "$HN_INDEX" \
  "$HOOK_DIR/sitecustomize.py" \
  "$CONFIG"
do
  test -e "$path" || {
    echo "ERROR: missing required path: $path"
    exit 1
  }
done

test ! -e "$OUTPUT" || {
  echo "ERROR: output already exists: $OUTPUT"
  exit 1
}

test ! -e "$TRACE_DIR" || {
  echo "ERROR: trace directory already exists: $TRACE_DIR"
  exit 1
}

mkdir -p \
  "$TRACE_DIR" \
  "$LOG_ROOT"

CURRENT_COMMIT="$(git rev-parse HEAD)"

echo "================================================================"
echo "D1/C SAMPLING TRACE"
echo "================================================================"
echo "mode=$MODE"
echo "run_id=$RUN_ID"
echo "commit=$CURRENT_COMMIT"
echo "config=$CONFIG"
echo "annotation=$ANNO"
echo "hardneg_index=$HN_INDEX"
echo "trace_dir=$TRACE_DIR"
echo "output=$OUTPUT"
echo "epochs=3"
echo "workers=4"
echo "seed=3407"
echo "================================================================"

set -o pipefail

/usr/bin/time -v \
env \
  PYTHONPATH="$HOOK_DIR:$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
  PYTHONHASHSEED=3407 \
  FLOODPSG_TRAIN_SEED=3407 \
  FLOODPSG_NEG_MODE=flood_hn \
  FLOODPSG_ZERO_NEG_PER_IMAGE=2 \
  FLOODPSG_HARDNEG_FRACTION=0.50 \
  FLOODPSG_HARDNEG_INDEX="$HN_INDEX" \
  FLOODPSG_SAMPLING_TRACE_DIR="$TRACE_DIR" \
  FLOODPSG_TRACE_RUN_ID="$RUN_ID" \
  FLOODPSG_TRACE_EXPECTED_TRAIN_IMAGES=1500 \
  FLOODPSG_TRACE_CODE_COMMIT="$CURRENT_COMMIT" \
  OMP_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 \
"$DSFORMER_PY" -u \
  scripts/run_fair_psgg_seeded.py \
  "$CONFIG" \
  "$OUTPUT" \
  --anno "$ANNO" \
  --img data/floodpsg \
  --seg data/floodpsg \
  --epochs 3 \
  --workers 4 \
  2>&1 | tee "$LOG"

STATUS=${PIPESTATUS[0]}

if [ "$STATUS" -ne 0 ]; then
  echo "ERROR: $RUN_ID trace run failed: status=$STATUS"
  exit "$STATUS"
fi

test -f "$OUTPUT/done.txt" || {
  echo "ERROR: missing done marker: $OUTPUT/done.txt"
  exit 1
}

grep -qx "done" "$OUTPUT/done.txt" || {
  echo "ERROR: invalid done marker"
  exit 1
}

TRACE_LINES="$(
  find "$TRACE_DIR" \
    -maxdepth 1 \
    -type f \
    -name '*.jsonl' \
    -print0 \
  | xargs -0 cat \
  | wc -l
)"

echo
echo "trace JSONL records: $TRACE_LINES"
echo "$RUN_ID SAMPLING TRACE: PASS"
