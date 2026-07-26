#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/projects/panoptic-scene-graph-generation}"
cd "$PROJECT_ROOT"

BRANCH_EXPECTED="exp/dsformer-fibe-d1v2a-geometry-existence-gate"
BRANCH_ACTUAL="$(git branch --show-current)"
if [[ "$BRANCH_ACTUAL" != "$BRANCH_EXPECTED" ]]; then
  echo "STOP: expected branch $BRANCH_EXPECTED, got $BRANCH_ACTUAL" >&2
  exit 1
fi

if [[ -n "$(git status --short --untracked-files=no)" ]]; then
  echo "STOP: tracked worktree is not clean" >&2
  git status --short --untracked-files=no >&2
  exit 1
fi

if ! git merge-base --is-ancestor \
  23eac3db9b0a3029a1aecfb4d962275d5d69ad33 HEAD
then
  echo "STOP: current branch does not contain reviewed real-DataLoader audit commit" >&2
  exit 1
fi

if pgrep -af 'run_fair_psgg_seeded.py|fair_psgg.*--epochs' >/dev/null; then
  echo "STOP: another training process appears to be running" >&2
  pgrep -af 'run_fair_psgg_seeded.py|fair_psgg.*--epochs' >&2 || true
  exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"
SEEDED_RUNNER="$PROJECT_ROOT/scripts/run_fair_psgg_seeded.py"
SAMPLING_AUDITOR="$PROJECT_ROOT/scripts/audit_d1_c_det_raw_batch_equivalence_v2.py"
CHECKPOINT_AUDITOR="$PROJECT_ROOT/scripts/audit_d1v2a_checkpoint_selection_protocol_v1.py"

C_CONFIG="$PROJECT_ROOT/outputs/flood_groupstrict_coarse8_floodhn_r50_detv2_seed3407_40epoch_v1/config.json"
D1V2A_CONFIG="$PROJECT_ROOT/configs/floodpsg/masks-loc-sem-flood-groupstrict-coarse8-fibe-d1v2a-geometry-existence-gate-v1.json"
ANNOTATION="$PROJECT_ROOT/data/floodpsg/annotations/floodpsg_canonical_trainval_coarse8_groupstrict_v1.json"
DATA_ROOT="$PROJECT_ROOT/data/floodpsg"
HN_INDEX="$PROJECT_ROOT/data/floodpsg/stats/dsformer_floodhn_sampling_v1/train_hardneg_index_v1.json"
TRAINER_SOURCE="$PROJECT_ROOT/fair_psgg/trainer.py"
RAW_TRACE_SOURCE="$PROJECT_ROOT/fair_psgg/data/raw_batch_trace.py"

STATIC_AUDIT="$PROJECT_ROOT/data/floodpsg/stats/dsformer_fibe_d1v2a_geometry_existence_gate_v1/audits/D1V2A_GEOMETRY_EXISTENCE_GATE_STATIC_AUDIT_V1.json"
REAL_AUDIT="$PROJECT_ROOT/data/floodpsg/stats/dsformer_fibe_d1v2a_geometry_existence_gate_v1/audits/D1V2A_REAL_DATALOADER_ACTIVATION_AUDIT_V1.json"
CONTRACT="$PROJECT_ROOT/data/floodpsg/stats/dsformer_fibe_d1v2a_geometry_existence_gate_v1/D1V2A_GEOMETRY_EXISTENCE_GATE_CONTRACT_V1.txt"

STAGE_ROOT="$PROJECT_ROOT/data/floodpsg/stats/dsformer_fibe_d1v2a_geometry_existence_gate_v1"
AUDIT_ROOT="$STAGE_ROOT/sampling_equivalence"
TRACE_ROOT="$AUDIT_ROOT/raw_batch_trace_v2"
C_TRACE="$TRACE_ROOT/C_DetV2_CurrentCode"
D1V2A_TRACE="$TRACE_ROOT/D1V2A_GeometryGate"

C_OUTPUT="$PROJECT_ROOT/outputs/audit_c_detv2_currentcode_seed3407_3epoch_d1v2a_v1"
D1V2A_OUTPUT="$PROJECT_ROOT/outputs/audit_d1v2a_geometry_gate_seed3407_3epoch_v1"

C_LOG="$AUDIT_ROOT/C_DetV2_CurrentCode_3epoch.log"
D1V2A_LOG="$AUDIT_ROOT/D1V2A_GeometryGate_3epoch.log"
SAMPLING_JSON="$AUDIT_ROOT/D1V2A_C_SAMPLING_EQUIVALENCE_AUDIT_V1.json"
SAMPLING_CSV="$AUDIT_ROOT/D1V2A_C_SAMPLING_EQUIVALENCE_MISMATCHES_V1.csv"
CHECKPOINT_JSON="$AUDIT_ROOT/D1V2A_3EPOCH_CHECKPOINT_SELECTION_PROTOCOL_AUDIT_V1.json"
PIPELINE_LOCK="$AUDIT_ROOT/D1V2A_3EPOCH_SAMPLING_AND_CHECKPOINT_PIPELINE_LOCK_V1.txt"

declare -A EXPECTED_SHA
EXPECTED_SHA["$C_CONFIG"]="4c005b4424e9d6fa60c356e18b19d2d92482e2d9f069dde3fdc658c9c13b3c8a"
EXPECTED_SHA["$D1V2A_CONFIG"]="b0ed00d180878072c63f5c0972f75ac670c05644f9f1af3b6e0c8eec9d13f397"
EXPECTED_SHA["$ANNOTATION"]="28f4dfe34bc14b389a3cffb63e6d463888668743df67e78022287e36fe3f9354"
EXPECTED_SHA["$HN_INDEX"]="b893ba565fd8cfdb1137f59bed343e6fb54d746e46b20382c655c07be0c6691f"
EXPECTED_SHA["$SEEDED_RUNNER"]="9368022c236165cc628de67f45d3dcc462645784db97cfd9d3e943fc23973b6c"
EXPECTED_SHA["$SAMPLING_AUDITOR"]="6e1316eaa19e96f47bf35101a071737e1f081cdb8458fb3015bdf99ba26b1e46"
EXPECTED_SHA["$RAW_TRACE_SOURCE"]="f7af648ca196c417222da964ad62af4c7a4b895712c0401c13a975ce7887c179"
EXPECTED_SHA["$TRAINER_SOURCE"]="9a0a4934b2567dc417f890641f352a82c19cdf2b3a42dd9f82157a795fee7355"
EXPECTED_SHA["$STATIC_AUDIT"]="bc67a9adad6b9cef4deed6137fbc60ec1e4406d8cb649a48f0ed20866109dc29"
EXPECTED_SHA["$REAL_AUDIT"]="def4775a7a658dcf01717f9365aa8cf99b2578ceecbd2beba353cd9f0d0d01c2"
EXPECTED_SHA["$CONTRACT"]="d5644eb7005d345c505fa85729b6fb6492cf0c47923e03f127bccbd3e664b931"

check_sha() {
  local path="$1"
  local expected="$2"
  if [[ ! -f "$path" ]]; then
    echo "STOP: missing file $path" >&2
    exit 1
  fi
  local actual
  actual="$(sha256sum "$path" | awk '{print $1}')"
  if [[ "$actual" != "$expected" ]]; then
    echo "STOP: SHA mismatch for $path" >&2
    echo "expected=$expected" >&2
    echo "actual=$actual" >&2
    exit 1
  fi
  echo "PASS SHA256: $path"
}

for path in "${!EXPECTED_SHA[@]}"; do
  check_sha "$path" "${EXPECTED_SHA[$path]}"
done

for executable in "$PYTHON_BIN" "$CHECKPOINT_AUDITOR"; do
  if [[ ! -f "$executable" ]]; then
    echo "STOP: missing executable/script $executable" >&2
    exit 1
  fi
done

for path in \
  "$C_OUTPUT" "$D1V2A_OUTPUT" \
  "$C_TRACE" "$D1V2A_TRACE" \
  "$C_LOG" "$D1V2A_LOG" \
  "$SAMPLING_JSON" "$SAMPLING_CSV" \
  "$CHECKPOINT_JSON" "$PIPELINE_LOCK"
do
  if [[ -e "$path" ]]; then
    echo "STOP: refusing to overwrite $path" >&2
    exit 1
  fi
done

mkdir -p "$AUDIT_ROOT" "$TRACE_ROOT"

CURRENT_COMMIT="$(git rev-parse HEAD)"

export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=3407
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

export FLOODPSG_TRAIN_SEED=3407
export FLOODPSG_KEEP_ZERO_REL=1
export FLOODPSG_NEG_MODE=flood_hn
export FLOODPSG_ZERO_NEG_PER_IMAGE=2
export FLOODPSG_HARDNEG_FRACTION=0.50
export FLOODPSG_HARDNEG_INDEX="$HN_INDEX"

export FLOODPSG_DETERMINISTIC_LOADER=1
export FLOODPSG_LOADER_SEED=3407
export FLOODPSG_DETERMINISTIC_SAMPLING=1
export FLOODPSG_SAMPLING_SEED=3407
export FLOODPSG_TRACE_CODE_COMMIT="$CURRENT_COMMIT"

run_training() {
  local label="$1"
  local run_id="$2"
  local config="$3"
  local output="$4"
  local trace_dir="$5"
  local log="$6"

  export FLOODPSG_RAW_BATCH_TRACE_DIR="$trace_dir"
  export FLOODPSG_TRACE_RUN_ID="$run_id"

  echo "================================================================"
  echo "D1-v2a 3-EPOCH SAMPLING TRACE RUN"
  echo "label=$label"
  echo "run_id=$run_id"
  echo "branch=$BRANCH_ACTUAL"
  echo "commit=$CURRENT_COMMIT"
  echo "config=$config"
  echo "output=$output"
  echo "trace_dir=$trace_dir"
  echo "epochs=3"
  echo "workers=4"
  echo "seed=3407"
  echo "final_test_allowed=false"
  echo "================================================================"

  (
    /usr/bin/time -v \
      "$PYTHON_BIN" -u "$SEEDED_RUNNER" \
      "$config" \
      "$output" \
      --anno "$ANNOTATION" \
      --img "$DATA_ROOT" \
      --seg "$DATA_ROOT" \
      --epochs 3 \
      --workers 4 \
      --no-bpbar
  ) 2>&1 | tee "$log"

  local trace_count
  trace_count="$(find "$trace_dir" -maxdepth 1 -type f -name '*.jsonl' | wc -l)"
  if [[ "$trace_count" -ne 3 ]]; then
    echo "STOP: $label produced $trace_count trace files, expected 3" >&2
    exit 1
  fi

  if [[ ! -f "$output/done.txt" ]]; then
    echo "STOP: $label did not produce done.txt" >&2
    exit 1
  fi

  echo "$label: PASS"
}

run_training \
  "C-DetV2 current-code control" \
  "C_DetV2_CurrentCode" \
  "$C_CONFIG" \
  "$C_OUTPUT" \
  "$C_TRACE" \
  "$C_LOG"

run_training \
  "D1-v2a GeometryGate" \
  "D1V2A_GeometryGate" \
  "$D1V2A_CONFIG" \
  "$D1V2A_OUTPUT" \
  "$D1V2A_TRACE" \
  "$D1V2A_LOG"

env FLOODPSG_KEEP_ZERO_REL=1 \
  "$PYTHON_BIN" -u "$SAMPLING_AUDITOR" \
  --c-trace "$C_TRACE" \
  --d1-trace "$D1V2A_TRACE" \
  --annotation "$ANNOTATION" \
  --hardneg-index "$HN_INDEX" \
  --c-config "$C_CONFIG" \
  --d1-config "$D1V2A_CONFIG" \
  --expected-epochs 3 \
  --expected-batches 187 \
  --expected-raw-images 1496 \
  --expected-train-images 1500 \
  --output-json "$SAMPLING_JSON" \
  --output-csv "$SAMPLING_CSV"

"$PYTHON_BIN" -u "$CHECKPOINT_AUDITOR" \
  --c-output "$C_OUTPUT" \
  --d1v2a-output "$D1V2A_OUTPUT" \
  --trainer-source "$TRAINER_SOURCE" \
  --annotation "$ANNOTATION" \
  --output "$CHECKPOINT_JSON" \
  --expected-commit "$CURRENT_COMMIT" \
  --expected-epochs 3 \
  --expected-workers 4

"$PYTHON_BIN" - \
  "$SAMPLING_JSON" \
  "$CHECKPOINT_JSON" <<'PY'
import json
import sys
from pathlib import Path

sampling_path = Path(sys.argv[1])
checkpoint_path = Path(sys.argv[2])

sampling = json.loads(sampling_path.read_text(encoding="utf-8"))
checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))

counter_keys = [
    key
    for key in sampling
    if key.endswith("_mismatches") or key.endswith("_failures")
]
nonzero = {
    key: sampling[key]
    for key in counter_keys
    if int(sampling[key]) != 0
}

print("sampling status:", sampling["status"])
print("sampling nonzero counters:", nonzero)
print("checkpoint status:", checkpoint["status"])
print("checkpoint checks:", checkpoint["checks"])

assert sampling["status"] == "PASS"
assert not nonzero
assert checkpoint["status"] == "PASS"
assert checkpoint["final_test_allowed"] is False
PY

cat > "$PIPELINE_LOCK" <<EOF
FLOODPSG D1-v2a 3-EPOCH SAMPLING AND CHECKPOINT PIPELINE LOCK V1

date=$(date --iso-8601=seconds)
branch=$BRANCH_ACTUAL
commit=$CURRENT_COMMIT

control=C-DetV2 current code
candidate=D1-v2a Geometry-only Relation-Existence Gate
epochs=3
workers=4
seed=3407

negative_sampling_mode=flood_hn
hard_negative_fraction=0.50
zero_negatives_per_image=2
keep_zero_relation_images=true
deterministic_loader=true
deterministic_sampling=true

sampling_equivalence_status=PASS
checkpoint_selection_protocol_status=PASS
checkpoint_selection_metric=validation_rel_mean_recall_at_50
water_hn_used_for_checkpoint_selection=false

formal_40epoch_training_allowed=false
review_required_before_formal_training=true
final_test_allowed=false
EOF

{
  echo "===== SOURCE SHA256 ====="
  sha256sum \
    "$C_CONFIG" "$D1V2A_CONFIG" "$ANNOTATION" "$HN_INDEX" \
    "$SEEDED_RUNNER" "$SAMPLING_AUDITOR" "$CHECKPOINT_AUDITOR" \
    "$RAW_TRACE_SOURCE" "$TRAINER_SOURCE" \
    "$STATIC_AUDIT" "$REAL_AUDIT" "$CONTRACT"
  echo
  echo "===== OUTPUT SHA256 ====="
  sha256sum \
    "$SAMPLING_JSON" "$SAMPLING_CSV" "$CHECKPOINT_JSON" \
    "$C_OUTPUT/config.json" "$C_OUTPUT/best_state.pth" "$C_OUTPUT/best_metrics.pth" \
    "$D1V2A_OUTPUT/config.json" "$D1V2A_OUTPUT/best_state.pth" "$D1V2A_OUTPUT/best_metrics.pth"
} >> "$PIPELINE_LOCK"

sha256sum "$PIPELINE_LOCK"

echo "================================================================"
echo "D1-v2a 3-EPOCH SAMPLING EQUIVALENCE PIPELINE: PASS"
echo "Sampling audit: $SAMPLING_JSON"
echo "Checkpoint audit: $CHECKPOINT_JSON"
echo "Pipeline lock: $PIPELINE_LOCK"
echo "Formal 40-epoch training remains NOT ALLOWED until reviewed."
echo "Final test remains LOCKED."
echo "================================================================"
