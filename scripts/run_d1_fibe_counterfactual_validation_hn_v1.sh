#!/usr/bin/env bash
set -euo pipefail

PROTOCOL="FLOODPSG D1 FIBE VALIDATION WATER-HN COUNTERFACTUAL PIPELINE V1"
EXPECTED_BRANCH="exp/dsformer-fibe-d1-diagnostics-v1"
EXPECTED_SCORER_SHA="34dc2b7b397fcbdf081cf58312f29f58a7472940d407d80fc51aa874089ff5ba"
EXPECTED_SUMMARIZER_SHA="52d2f90c4a3e1ff474d9b01e01d8981a92d69dce26071a8a70d62664f1a6aec5"
EXPECTED_EVALUATOR_SHA="8ab797bb4b95dac7cd0afe9b7e4365fc23753a9bfa811409701d00dd3be55ae5"
EXPECTED_D1_CONFIG_SHA="bd095e9c4b04d7b7a39d6ddc73e55424d90da3c926ff7031167fd562ef94ad80"
EXPECTED_D1_CKPT_SHA="e6d158e41ac2a3cb933941a457cde19d3512a08a7263eeeab7cf521dfa80f492"
EXPECTED_ANNO_SHA="28f4dfe34bc14b389a3cffb63e6d463888668743df67e78022287e36fe3f9354"
EXPECTED_HN_SHA="1ef449f61d3e3bc0f6a56cbab038ec4264d43e775bd20bbf6006e3458b5ae47c"
EXPECTED_FORMAL_C_PKL_SHA="3e339f190ed5092177c2619dea64ed2e1e6b32ab7ebcc1203c8258cc43bb7cc5"
EXPECTED_FORMAL_D1_PKL_SHA="5637b8a2dd4b06922503883290e64c2c8a3063becdebec30e0930bff4a3ed1dd"
EXPECTED_BASELINE_LOCK_SHA="b540d1cd07cde6153fc380c96f847d54e329bf3062ac17b1c4c56ae0cf43b1e4"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="$PROJECT_ROOT/.venv/bin/python"
SCORER="$PROJECT_ROOT/scripts/score_d1_fibe_counterfactual_validation_hn_v1.py"
SUMMARIZER="$PROJECT_ROOT/scripts/summarize_d1_fibe_counterfactual_v1.py"
EVALUATOR="$PROJECT_ROOT/scripts/evaluate_dsformer_hardneg_strict_v1.py"

D1_DIR="$PROJECT_ROOT/outputs/flood_groupstrict_coarse8_floodhn_r50_fibe_d1_detv2_seed3407_40epoch_v1"
ANNO="$PROJECT_ROOT/data/floodpsg/annotations/floodpsg_canonical_trainval_coarse8_groupstrict_v1.json"
HN_CSV="$PROJECT_ROOT/data/floodpsg/stats/frozen_hardneg_eval_validation_v1/02_validation_hardneg_water.csv"

DIAG_ROOT="$PROJECT_ROOT/data/floodpsg/stats/dsformer_fibe_d1_floodhn_r50_v1/d1_diagnostics_v1"
BASELINE_LOCK="$DIAG_ROOT/D1_DIAGNOSTICS_BASELINE_LOCK_V1.txt"
FORMAL_ROOT="$PROJECT_ROOT/data/floodpsg/stats/dsformer_fibe_d1_floodhn_r50_v1/validation_hn_gtpair_scoring_v1"
ROOT="$DIAG_ROOT/counterfactual_v1"
LOCK="$ROOT/D1_FIBE_COUNTERFACTUAL_PIPELINE_LOCK_V1.txt"

EXPECTED_IMAGES=75
EXPECTED_PAIRS=322
SEED=3407

sha256_value() {
  sha256sum "$1" | awk '{print $1}'
}

require_sha() {
  local path="$1"
  local expected="$2"
  local actual
  actual="$(sha256_value "$path")"
  if [[ "$actual" != "$expected" ]]; then
    echo "FAIL SHA256: $path" >&2
    echo "expected=$expected" >&2
    echo "actual=$actual" >&2
    exit 1
  fi
  echo "PASS SHA256: $path"
}

run_logged() {
  local label="$1"
  local log="$2"
  shift 2
  echo
  echo "===== $label ====="
  set +e
  /usr/bin/time -v "$@" > >(tee "$log") 2> >(tee -a "$log" >&2)
  local status=$?
  set -e
  if [[ $status -ne 0 ]]; then
    echo "ERROR: command failed with status $status; log=$log" >&2
    exit "$status"
  fi
}

printf '%s\n' "================================================================"
printf '%s\n' "$PROTOCOL"
printf '%s\n' "================================================================"
echo "date=$(date --iso-8601=seconds)"
echo "project=$PROJECT_ROOT"
echo "branch=$(git branch --show-current)"
echo "commit=$(git rev-parse HEAD)"
echo "output_root=$ROOT"
echo "development_split=validation"
echo "expected_images=$EXPECTED_IMAGES"
echo "expected_pairs=$EXPECTED_PAIRS"
echo "final_test_allowed=false"
printf '%s\n' "================================================================"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "FAIL: Python environment missing: $PYTHON_BIN" >&2
  exit 1
fi

if [[ "$(git branch --show-current)" != "$EXPECTED_BRANCH" ]]; then
  echo "FAIL: expected branch $EXPECTED_BRANCH" >&2
  exit 1
fi

TRACKED_STATUS="$(git status --short --untracked-files=no)"
if [[ -n "$TRACKED_STATUS" ]]; then
  echo "FAIL: tracked worktree is not clean:" >&2
  printf '%s\n' "$TRACKED_STATUS" >&2
  exit 1
fi

for path in \
  "$SCORER" \
  "$SUMMARIZER" \
  "$EVALUATOR" \
  "$D1_DIR/config.json" \
  "$D1_DIR/best_state.pth" \
  "$ANNO" \
  "$HN_CSV" \
  "$BASELINE_LOCK" \
  "$FORMAL_ROOT/C_DetV2_validation_hn_gtpairs_v1.pkl" \
  "$FORMAL_ROOT/D1_DetV2_validation_hn_gtpairs_v1.pkl" \
  "$FORMAL_ROOT/C_D1_VALIDATION_HN_POINT_ESTIMATE_GATE_V1.json"
do
  if [[ ! -f "$path" ]]; then
    echo "FAIL: required file missing: $path" >&2
    exit 1
  fi
done

if [[ -e "$ROOT" ]]; then
  echo "FAIL: refusing to overwrite diagnostic output root: $ROOT" >&2
  exit 1
fi

require_sha "$SCORER" "$EXPECTED_SCORER_SHA"
require_sha "$SUMMARIZER" "$EXPECTED_SUMMARIZER_SHA"
require_sha "$EVALUATOR" "$EXPECTED_EVALUATOR_SHA"
require_sha "$D1_DIR/config.json" "$EXPECTED_D1_CONFIG_SHA"
require_sha "$D1_DIR/best_state.pth" "$EXPECTED_D1_CKPT_SHA"
require_sha "$ANNO" "$EXPECTED_ANNO_SHA"
require_sha "$HN_CSV" "$EXPECTED_HN_SHA"
require_sha "$FORMAL_ROOT/C_DetV2_validation_hn_gtpairs_v1.pkl" "$EXPECTED_FORMAL_C_PKL_SHA"
require_sha "$FORMAL_ROOT/D1_DetV2_validation_hn_gtpairs_v1.pkl" "$EXPECTED_FORMAL_D1_PKL_SHA"
require_sha "$BASELINE_LOCK" "$EXPECTED_BASELINE_LOCK_SHA"

"$PYTHON_BIN" -m py_compile "$SCORER" "$SUMMARIZER" "$EVALUATOR"
bash -n "$0"

mkdir -p "$ROOT"

COMMON_ENV=(
  env
  "PYTHONPATH=$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
  "PYTHONHASHSEED=$SEED"
  "OMP_NUM_THREADS=1"
  "MKL_NUM_THREADS=1"
)

MODES=(normal residual_bypass raw_feature_zero family_shuffle)

for mode in "${MODES[@]}"; do
  PKL="$ROOT/D1_${mode}_validation_hn_gtpairs_v1.pkl"
  REPORT="$ROOT/D1_${mode}_validation_hn_gtpairs_v1.json"
  PAIR_AUDIT="$ROOT/D1_${mode}_pair_audit_v1.csv"
  SCORE_LOG="$ROOT/D1_${mode}_score_v1.log"
  EVAL_ROOT="$ROOT/D1_${mode}_strict_eval_v1"
  EVAL_LOG="$ROOT/D1_${mode}_strict_eval_v1.log"

  run_logged \
    "SCORE D1 MODE=$mode" \
    "$SCORE_LOG" \
    "${COMMON_ENV[@]}" \
    "$PYTHON_BIN" -u "$SCORER" \
      --model-dir "$D1_DIR" \
      --annotation "$ANNO" \
      --hardneg "$HN_CSV" \
      --data-root "$PROJECT_ROOT/data/floodpsg" \
      --output "$PKL" \
      --report "$REPORT" \
      --pair-audit "$PAIR_AUDIT" \
      --mode "$mode" \
      --split validation \
      --expected-images "$EXPECTED_IMAGES" \
      --expected-pairs "$EXPECTED_PAIRS" \
      --batch-size 1 \
      --workers 0 \
      --max-relations 512 \
      --seed "$SEED" \
      --device cuda \
      --require-all-fibe-valid

  run_logged \
    "STRICT EVALUATE D1 MODE=$mode" \
    "$EVAL_LOG" \
    "${COMMON_ENV[@]}" \
    "$PYTHON_BIN" -u "$EVALUATOR" \
      --results "$PKL" \
      --annotation "$ANNO" \
      --hardneg "$HN_CSV" \
      --output "$EVAL_ROOT" \
      --data-root "$PROJECT_ROOT/data/floodpsg" \
      --split validation \
      --expected-pairs "$EXPECTED_PAIRS" \
      --expected-images "$EXPECTED_IMAGES" \
      --strict-complete
done

AUDIT_JSON="$ROOT/D1_FIBE_COUNTERFACTUAL_AUDIT_V1.json"
FAMILY_CSV="$ROOT/D1_FIBE_COUNTERFACTUAL_PER_FAMILY_V1.csv"
PAIR_DELTA_CSV="$ROOT/D1_C_PAIRWISE_LOGIT_DELTA_V1.csv"
SUMMARY_LOG="$ROOT/D1_FIBE_COUNTERFACTUAL_SUMMARY_V1.log"

run_logged \
  "SUMMARIZE D1 FIBE COUNTERFACTUALS" \
  "$SUMMARY_LOG" \
  "${COMMON_ENV[@]}" \
  "$PYTHON_BIN" -u "$SUMMARIZER" \
    --root "$ROOT" \
    --formal-root "$FORMAL_ROOT" \
    --output "$AUDIT_JSON" \
    --family-output "$FAMILY_CSV" \
    --pair-delta-output "$PAIR_DELTA_CSV"

{
  echo "FLOODPSG D1 FIBE COUNTERFACTUAL PIPELINE LOCK V1"
  echo
  echo "date=$(date --iso-8601=seconds)"
  echo "branch=$(git branch --show-current)"
  echo "commit=$(git rev-parse HEAD)"
  echo "development_split=validation"
  echo "images=$EXPECTED_IMAGES"
  echo "pairs=$EXPECTED_PAIRS"
  echo "seed=$SEED"
  echo "new_model_training=false"
  echo "formal_D1_gate_remains=FAIL"
  echo "final_test_allowed=false"
  echo
  echo "===== SOURCE SHA256 ====="
  sha256sum "$SCORER" "$SUMMARIZER" "$EVALUATOR"
  echo
  echo "===== LOCKED INPUT SHA256 ====="
  sha256sum \
    "$D1_DIR/config.json" \
    "$D1_DIR/best_state.pth" \
    "$ANNO" \
    "$HN_CSV" \
    "$BASELINE_LOCK" \
    "$FORMAL_ROOT/C_DetV2_validation_hn_gtpairs_v1.pkl" \
    "$FORMAL_ROOT/D1_DetV2_validation_hn_gtpairs_v1.pkl" \
    "$FORMAL_ROOT/C_D1_VALIDATION_HN_POINT_ESTIMATE_GATE_V1.json"
  echo
  echo "===== COUNTERFACTUAL OUTPUT SHA256 ====="
  find "$ROOT" -type f ! -name "$(basename "$LOCK")" -print0 \
    | sort -z \
    | xargs -0 sha256sum
  echo
  echo "===== COUNTERFACTUAL AUDIT ====="
  cat "$AUDIT_JSON"
} > "$LOCK"

cat "$AUDIT_JSON"

echo
echo "================================================================"
echo "D1 FIBE VALIDATION WATER-HN COUNTERFACTUAL PIPELINE: PASS"
echo "Diagnostic result: $AUDIT_JSON"
echo "Formal D1 Water-HN gate remains FAIL."
echo "Final test remains LOCKED."
echo "================================================================"
