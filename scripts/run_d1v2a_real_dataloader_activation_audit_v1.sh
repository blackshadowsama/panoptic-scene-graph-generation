#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/projects/panoptic-scene-graph-generation}"
cd "$PROJECT_ROOT"

EXPECTED_BRANCH="exp/dsformer-fibe-d1v2a-geometry-existence-gate"
BRANCH="$(git branch --show-current)"

if [[ "$BRANCH" != "$EXPECTED_BRANCH" ]]; then
  echo "STOP: expected branch $EXPECTED_BRANCH, got $BRANCH" >&2
  exit 1
fi

if [[ -n "$(git status --short --untracked-files=no)" ]]; then
  echo "STOP: tracked worktree is not clean" >&2
  git status --short --untracked-files=no >&2
  exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"
SCRIPT="$PROJECT_ROOT/scripts/audit_d1v2a_real_dataloader_activation_v1.py"

CONFIG="$PROJECT_ROOT/configs/floodpsg/masks-loc-sem-flood-groupstrict-coarse8-fibe-d1v2a-geometry-existence-gate-v1.json"
ANNOTATION="$PROJECT_ROOT/data/floodpsg/annotations/floodpsg_canonical_trainval_coarse8_groupstrict_v1.json"
DATA_ROOT="$PROJECT_ROOT/data/floodpsg"
STAGE_ROOT="$PROJECT_ROOT/data/floodpsg/stats/dsformer_fibe_d1v2a_geometry_existence_gate_v1"
OUTPUT="$STAGE_ROOT/audits/D1V2A_REAL_DATALOADER_ACTIVATION_AUDIT_V1.json"

for path in "$PYTHON_BIN" "$SCRIPT" "$CONFIG" "$ANNOTATION"; do
  if [[ ! -f "$path" ]]; then
    echo "STOP: missing file $path" >&2
    exit 1
  fi
done

if [[ ! -d "$DATA_ROOT" ]]; then
  echo "STOP: missing data root $DATA_ROOT" >&2
  exit 1
fi

if [[ -e "$OUTPUT" ]]; then
  echo "STOP: audit output already exists: $OUTPUT" >&2
  exit 1
fi

export PYTHONPATH="$PROJECT_ROOT"
export PYTHONHASHSEED=3407
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

export FLOODPSG_NEG_MODE=flood_hn
export FLOODPSG_HARDNEG_FRACTION=0.50
export FLOODPSG_ZERO_NEG_PER_IMAGE=2
export FLOODPSG_KEEP_ZERO_REL=1
export FLOODPSG_HARDNEG_INDEX="$PROJECT_ROOT/data/floodpsg/stats/dsformer_floodhn_sampling_v1/train_hardneg_index_v1.json"

export FLOODPSG_DETERMINISTIC_SAMPLING=1
export FLOODPSG_TRAIN_SEED=3407
export FLOODPSG_SAMPLING_SEED=3407
export FLOODPSG_CURRENT_EPOCH=0

echo "================================================================"
echo "FLOODPSG D1-v2a REAL DATALOADER ACTIVATION AUDIT V1"
echo "branch=$BRANCH"
echo "commit=$(git rev-parse HEAD)"
echo "seed=3407"
echo "epoch=0"
echo "workers=0"
echo "final_test_allowed=false"
echo "================================================================"

/usr/bin/time -v \
  "$PYTHON_BIN" -u "$SCRIPT" \
  --config "$CONFIG" \
  --annotation "$ANNOTATION" \
  --data-root "$DATA_ROOT" \
  --output "$OUTPUT" \
  --seed 3407 \
  --epoch 0 \
  --workers 0 \
  --max-raw-batches 64 \
  --expected-gate-parameters 10241

"$PYTHON_BIN" - "$OUTPUT" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
report = json.loads(path.read_text(encoding="utf-8"))
failed = [key for key, value in report["checks"].items() if not value]

print("status:", report["status"])
print("final_test_allowed:", report["final_test_allowed"])
print("selected_batch:", report["selected_batch"])
print("step1:", report["step1"])
print("step2:", report["step2"])
print("failed_checks:", failed)

assert report["status"] == "PASS"
assert report["final_test_allowed"] is False
assert not failed
PY

sha256sum "$OUTPUT"

echo "================================================================"
echo "D1-v2a REAL DATALOADER ACTIVATION AUDIT PIPELINE: PASS"
echo "Result: $OUTPUT"
echo "Formal 3-epoch training remains NOT ALLOWED until this audit is reviewed."
echo "Final test remains LOCKED."
echo "================================================================"
