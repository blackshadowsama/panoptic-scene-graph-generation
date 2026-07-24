#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "$0")/.."

ROOT="$PWD"

PY="$ROOT/.venv/bin/python"

ANNO="$ROOT/data/floodpsg/annotations/floodpsg_canonical_trainval_coarse8_groupstrict_v1.json"

HN_INDEX="$ROOT/data/floodpsg/stats/dsformer_floodhn_sampling_v1/train_hardneg_index_v1.json"

AUDIT_ROOT="$ROOT/data/floodpsg/stats/dsformer_fibe_d1_floodhn_r50_v1"

TRACE_ROOT="$AUDIT_ROOT/raw_batch_trace_v2"

LOG_ROOT="$ROOT/data/floodpsg/logs/d1_c_det_raw_batch_trace_v2"

MODE="${1:-}"

case "$MODE" in
  C)
    RUN_ID="C_Det_FloodHN_R50"

    CONFIG="$ROOT/outputs/flood_groupstrict_coarse8_floodhn_r50_seed3407_40epoch_v1/config.json"

    OUTPUT="$ROOT/outputs/audit_c_det_floodhn_r50_seed3407_3epoch_rawbatch_v2"
    ;;

  D1)
    RUN_ID="D1_Det_FloodHN_R50_FIBE"

    CONFIG="$ROOT/configs/floodpsg/masks-loc-sem-flood-groupstrict-coarse8-fibe-d1-v1.json"

    OUTPUT="$ROOT/outputs/audit_d1_det_floodhn_r50_fibe_seed3407_3epoch_rawbatch_v2"
    ;;

  *)
    echo "Usage:"
    echo "  bash scripts/run_d1_c_det_raw_batch_trace_v2.sh C"
    echo "  bash scripts/run_d1_c_det_raw_batch_trace_v2.sh D1"
    exit 2
    ;;
esac

TRACE_DIR="$TRACE_ROOT/$RUN_ID"
LOG="$LOG_ROOT/${RUN_ID}.log"

for path in \
  "$PY" \
  "$ANNO" \
  "$HN_INDEX" \
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
  "$TRACE_ROOT" \
  "$LOG_ROOT"

COMMIT="$(git rev-parse HEAD)"

echo "================================================================"
echo "FLOODPSG C-DET / D1-DET RAW BATCH TRACE V2"
echo "================================================================"
echo "mode=$MODE"
echo "run_id=$RUN_ID"
echo "branch=$(git branch --show-current)"
echo "commit=$COMMIT"
echo "config=$CONFIG"
echo "annotation=$ANNO"
echo "hardneg_index=$HN_INDEX"
echo "trace_dir=$TRACE_DIR"
echo "output=$OUTPUT"
echo "epochs=3"
echo "workers=4"
echo "seed=3407"
echo "keep_zero_rel=1"
echo "deterministic_loader=1"
echo "deterministic_sampling=1"
echo "================================================================"

set -o pipefail

/usr/bin/time -v \
env \
  PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
  PYTHONHASHSEED=3407 \
  FLOODPSG_TRAIN_SEED=3407 \
  FLOODPSG_KEEP_ZERO_REL=1 \
  FLOODPSG_NEG_MODE=flood_hn \
  FLOODPSG_ZERO_NEG_PER_IMAGE=2 \
  FLOODPSG_HARDNEG_FRACTION=0.50 \
  FLOODPSG_HARDNEG_INDEX="$HN_INDEX" \
  FLOODPSG_DETERMINISTIC_LOADER=1 \
  FLOODPSG_LOADER_SEED=3407 \
  FLOODPSG_DETERMINISTIC_SAMPLING=1 \
  FLOODPSG_SAMPLING_SEED=3407 \
  FLOODPSG_RAW_BATCH_TRACE_DIR="$TRACE_DIR" \
  FLOODPSG_TRACE_RUN_ID="$RUN_ID" \
  FLOODPSG_TRACE_CODE_COMMIT="$COMMIT" \
  OMP_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 \
"$PY" -u \
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
  echo "ERROR: training exited with status=$STATUS"
  exit "$STATUS"
fi

test -f "$OUTPUT/done.txt" || {
  echo "ERROR: done.txt missing"
  exit 1
}

grep -qx "done" "$OUTPUT/done.txt" || {
  echo "ERROR: invalid done.txt"
  exit 1
}

TRACE_FILE_COUNT="$(
  find "$TRACE_DIR" \
    -maxdepth 1 \
    -type f \
    -name '*.jsonl' \
    | wc -l
)"

if [ "$TRACE_FILE_COUNT" -ne 3 ]; then
  echo "ERROR: expected 3 epoch trace files, got $TRACE_FILE_COUNT"
  exit 1
fi

echo
echo "trace files=$TRACE_FILE_COUNT"
echo "$RUN_ID RAW BATCH TRACE V2: PASS"
