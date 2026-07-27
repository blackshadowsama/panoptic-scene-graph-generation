#!/usr/bin/env bash
set -euo pipefail

PROTOCOL="FLOODPSG D1-v2a VALIDATION WATER-HN SCIENTIFIC GATE PIPELINE V1"
EXPECTED_BRANCH="exp/dsformer-fibe-d1v2a-geometry-existence-gate"
FORMAL_MODEL_COMMIT="fe3a136ef18600aba7780ab5816316a4df0d7a95"

EXPECTED_SCORER_SHA="ceec8773fefbaf99579c804eb15bf2f75c236ba76cf38d8dc301b1be801d36fd"
EXPECTED_SUMMARIZER_SHA="35e2328dea6f4c62bfe3ad8bdab454263be15eed7d475705ed0de87bead51f13"
EXPECTED_EVALUATOR_SHA="8ab797bb4b95dac7cd0afe9b7e4365fc23753a9bfa811409701d00dd3be55ae5"

EXPECTED_MODEL_CONFIG_SHA="c6192314591b99976e4d8d10f54006971e8a90994bd0bebc97c7d04fb744e2e4"
EXPECTED_MODEL_CKPT_SHA="a3fa5cb575b34005ecdce024bb8d50a331a7ad1f062a4e56d82694f3dd2f2fe8"
EXPECTED_MODEL_COMMIT_FILE_SHA="31f3b5fcced7254b346f77f82576f567edd06fcabadd8cd56f5104b2c2f8d07b"
EXPECTED_ANNO_SHA="28f4dfe34bc14b389a3cffb63e6d463888668743df67e78022287e36fe3f9354"
EXPECTED_HN_SHA="1ef449f61d3e3bc0f6a56cbab038ec4264d43e775bd20bbf6006e3458b5ae47c"
EXPECTED_VALIDATION_CACHE_SHA="8bb894f50ce3cd767b788cfd5e361d35dff011a66c57b753db4c61a333e246be"
EXPECTED_FORMAL_C_PKL_SHA="3e339f190ed5092177c2619dea64ed2e1e6b32ab7ebcc1203c8258cc43bb7cc5"
EXPECTED_COMPLETION_AUDIT_SHA="5ce3766c627336cbbf9214ce8967d4020249fb1c427e834a1b5417e050d2c882"

EXPECTED_CONFIG_SOURCE_SHA="ee7a960bf13c511532fe3957aa78fb865ac3064ec35715928e4a9b5011281d13"
EXPECTED_FROM_CONFIG_SOURCE_SHA="bce135ac113917bc2466ab92482f0689ab3228a51738e02cbc151975862b1eb1"
EXPECTED_TRAINER_SOURCE_SHA="9a0a4934b2567dc417f890641f352a82c19cdf2b3a42dd9f82157a795fee7355"
EXPECTED_DANIFORMER_SOURCE_SHA="8d9f120df80fbed932ff5469fccff79cd4ef842d989219a9f55dd6ba635253a6"
EXPECTED_GATE_SOURCE_SHA="77841bcdd69576fa5b18931c483e150954b9159846e2906a029d4596321c94eb"
EXPECTED_CACHE_SOURCE_SHA="e700cf76ce2f74996ee73344b8dad10799d5f20578fa2fb4e597eb8bf90eab71"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="$PROJECT_ROOT/.venv/bin/python"
SCORER="$PROJECT_ROOT/scripts/score_d1v2a_validation_hn_counterfactual_v1.py"
SUMMARIZER="$PROJECT_ROOT/scripts/summarize_d1v2a_validation_hn_scientific_gate_v1.py"
EVALUATOR="$PROJECT_ROOT/scripts/evaluate_dsformer_hardneg_strict_v1.py"

MODEL_DIR="$PROJECT_ROOT/outputs/flood_groupstrict_coarse8_floodhn_r50_fibe_d1v2a_geometry_gate_detv2_seed3407_40epoch_v1"
ANNO="$PROJECT_ROOT/data/floodpsg/annotations/floodpsg_canonical_trainval_coarse8_groupstrict_v1.json"
HN_CSV="$PROJECT_ROOT/data/floodpsg/stats/frozen_hardneg_eval_validation_v1/02_validation_hardneg_water.csv"
VALIDATION_CACHE="$PROJECT_ROOT/data/floodpsg/features/fibe_scalar_v1/validation_features_metadata_v1.pt"
FORMAL_C_PKL="$PROJECT_ROOT/data/floodpsg/stats/dsformer_fibe_d1_floodhn_r50_v1/validation_hn_gtpair_scoring_v1/C_DetV2_validation_hn_gtpairs_v1.pkl"

STAGE_ROOT="$PROJECT_ROOT/data/floodpsg/stats/dsformer_fibe_d1v2a_geometry_existence_gate_v1"
COMPLETION_AUDIT="$STAGE_ROOT/audits/D1V2A_FORMAL_40EPOCH_COMPLETION_AUDIT_V1.json"
ROOT="$STAGE_ROOT/validation_hn_scientific_gate_v1"
LOCK="$ROOT/D1V2A_VALIDATION_HN_SCIENTIFIC_GATE_PIPELINE_LOCK_V1.txt"

EXPECTED_IMAGES=75
EXPECTED_PAIRS=322
SEED=3407
BOOTSTRAP_ITERATIONS=2000

sha256_value() {
  sha256sum "$1" | awk '{print $1}'
}

require_sha() {
  local path="$1"
  local expected="$2"
  local actual

  if [[ ! -f "$path" ]]; then
    echo "FAIL: required file missing: $path" >&2
    exit 1
  fi

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
echo "current_commit=$(git rev-parse HEAD)"
echo "formal_model_commit=$FORMAL_MODEL_COMMIT"
echo "output_root=$ROOT"
echo "development_split=validation"
echo "expected_images=$EXPECTED_IMAGES"
echo "expected_pairs=$EXPECTED_PAIRS"
echo "bootstrap_iterations=$BOOTSTRAP_ITERATIONS"
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

if ! git merge-base --is-ancestor "$FORMAL_MODEL_COMMIT" HEAD; then
  echo "FAIL: current branch does not contain formal model commit" >&2
  exit 1
fi

TRACKED_STATUS="$(git status --short --untracked-files=no)"
if [[ -n "$TRACKED_STATUS" ]]; then
  echo "FAIL: tracked worktree is not clean:" >&2
  printf '%s\n' "$TRACKED_STATUS" >&2
  exit 1
fi

if [[ -e "$ROOT" ]]; then
  echo "FAIL: refusing to overwrite output root: $ROOT" >&2
  exit 1
fi

require_sha "$SCORER" "$EXPECTED_SCORER_SHA"
require_sha "$SUMMARIZER" "$EXPECTED_SUMMARIZER_SHA"
require_sha "$EVALUATOR" "$EXPECTED_EVALUATOR_SHA"

require_sha "$MODEL_DIR/config.json" "$EXPECTED_MODEL_CONFIG_SHA"
require_sha "$MODEL_DIR/best_state.pth" "$EXPECTED_MODEL_CKPT_SHA"
require_sha "$MODEL_DIR/commit.txt" "$EXPECTED_MODEL_COMMIT_FILE_SHA"
require_sha "$ANNO" "$EXPECTED_ANNO_SHA"
require_sha "$HN_CSV" "$EXPECTED_HN_SHA"
require_sha "$VALIDATION_CACHE" "$EXPECTED_VALIDATION_CACHE_SHA"
require_sha "$FORMAL_C_PKL" "$EXPECTED_FORMAL_C_PKL_SHA"
require_sha "$COMPLETION_AUDIT" "$EXPECTED_COMPLETION_AUDIT_SHA"

require_sha "$PROJECT_ROOT/fair_psgg/config.py" "$EXPECTED_CONFIG_SOURCE_SHA"
require_sha "$PROJECT_ROOT/fair_psgg/from_config.py" "$EXPECTED_FROM_CONFIG_SOURCE_SHA"
require_sha "$PROJECT_ROOT/fair_psgg/trainer.py" "$EXPECTED_TRAINER_SOURCE_SHA"
require_sha "$PROJECT_ROOT/fair_psgg/models/daniformer.py" "$EXPECTED_DANIFORMER_SOURCE_SHA"
require_sha "$PROJECT_ROOT/fair_psgg/models/fibe_existence_gate.py" "$EXPECTED_GATE_SOURCE_SHA"
require_sha "$PROJECT_ROOT/fair_psgg/data/fibe_cache.py" "$EXPECTED_CACHE_SOURCE_SHA"

if [[ "$(cat "$MODEL_DIR/commit.txt")" != "$FORMAL_MODEL_COMMIT" ]]; then
  echo "FAIL: formal model commit text mismatch" >&2
  exit 1
fi

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

MODES=(normal gate_bypass raw_feature_zero family_shuffle)

for mode in "${MODES[@]}"; do
  PKL="$ROOT/D1V2A_${mode}_validation_hn_gtpairs_v1.pkl"
  REPORT="$ROOT/D1V2A_${mode}_validation_hn_gtpairs_v1.json"
  PAIR_AUDIT="$ROOT/D1V2A_${mode}_pair_audit_v1.csv"
  SCORE_LOG="$ROOT/D1V2A_${mode}_score_v1.log"
  EVAL_ROOT="$ROOT/D1V2A_${mode}_strict_eval_v1"
  EVAL_LOG="$ROOT/D1V2A_${mode}_strict_eval_v1.log"

  run_logged \
    "SCORE D1-v2a MODE=$mode" \
    "$SCORE_LOG" \
    "${COMMON_ENV[@]}" \
    "$PYTHON_BIN" -u "$SCORER" \
      --model-dir "$MODEL_DIR" \
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
    "STRICT EVALUATE D1-v2a MODE=$mode" \
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

AUDIT_JSON="$ROOT/D1V2A_VALIDATION_HN_SCIENTIFIC_GATE_V1.json"
FAMILY_CSV="$ROOT/D1V2A_VALIDATION_HN_PER_FAMILY_V1.csv"
PAIR_CSV="$ROOT/D1V2A_VALIDATION_HN_PAIRWISE_C_COMPARISON_V1.csv"
BOOTSTRAP_JSON="$ROOT/D1V2A_VALIDATION_HN_IMAGE_CLUSTER_BOOTSTRAP_V1.json"
SUMMARY_LOG="$ROOT/D1V2A_VALIDATION_HN_SCIENTIFIC_GATE_SUMMARY_V1.log"

run_logged \
  "SUMMARIZE D1-v2a VALIDATION WATER-HN SCIENTIFIC GATE" \
  "$SUMMARY_LOG" \
  "${COMMON_ENV[@]}" \
  "$PYTHON_BIN" -u "$SUMMARIZER" \
    --root "$ROOT" \
    --formal-c-pickle "$FORMAL_C_PKL" \
    --completion-audit "$COMPLETION_AUDIT" \
    --output "$AUDIT_JSON" \
    --family-output "$FAMILY_CSV" \
    --pair-output "$PAIR_CSV" \
    --bootstrap-output "$BOOTSTRAP_JSON" \
    --bootstrap-iterations "$BOOTSTRAP_ITERATIONS" \
    --seed "$SEED"

GATE_RESULT="$(
  "$PYTHON_BIN" - "$AUDIT_JSON" <<'PY'
import json
import sys
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
assert payload["technical_status"] == "PASS"
assert payload["final_test_allowed"] is False
print(payload["scientific_point_estimate_gate"])
PY
)"

{
  echo "FLOODPSG D1-v2a VALIDATION WATER-HN SCIENTIFIC GATE PIPELINE LOCK V1"
  echo
  echo "date=$(date --iso-8601=seconds)"
  echo "branch=$(git branch --show-current)"
  echo "current_commit=$(git rev-parse HEAD)"
  echo "formal_model_commit=$FORMAL_MODEL_COMMIT"
  echo "development_split=validation"
  echo "images=$EXPECTED_IMAGES"
  echo "pairs=$EXPECTED_PAIRS"
  echo "seed=$SEED"
  echo "bootstrap_iterations=$BOOTSTRAP_ITERATIONS"
  echo "technical_pipeline=PASS"
  echo "scientific_point_estimate_gate=$GATE_RESULT"
  echo "final_test_allowed=false"
  echo
  echo "===== SOURCE SHA256 ====="
  sha256sum "$SCORER" "$SUMMARIZER" "$EVALUATOR"
  echo
  echo "===== LOCKED MODEL/DATA SHA256 ====="
  sha256sum \
    "$MODEL_DIR/config.json" \
    "$MODEL_DIR/best_state.pth" \
    "$MODEL_DIR/commit.txt" \
    "$ANNO" \
    "$HN_CSV" \
    "$VALIDATION_CACHE" \
    "$FORMAL_C_PKL" \
    "$COMPLETION_AUDIT"
  echo
  echo "===== LOCKED PROJECT SOURCE SHA256 ====="
  sha256sum \
    "$PROJECT_ROOT/fair_psgg/config.py" \
    "$PROJECT_ROOT/fair_psgg/from_config.py" \
    "$PROJECT_ROOT/fair_psgg/trainer.py" \
    "$PROJECT_ROOT/fair_psgg/models/daniformer.py" \
    "$PROJECT_ROOT/fair_psgg/models/fibe_existence_gate.py" \
    "$PROJECT_ROOT/fair_psgg/data/fibe_cache.py"
  echo
  echo "===== OUTPUT SHA256 ====="
  find "$ROOT" -type f ! -name "$(basename "$LOCK")" -print0 \
    | sort -z \
    | xargs -0 sha256sum
  echo
  echo "===== SCIENTIFIC GATE REPORT ====="
  cat "$AUDIT_JSON"
} > "$LOCK"

sha256sum "$LOCK"

cat "$AUDIT_JSON"

echo
echo "================================================================"
echo "D1-v2a VALIDATION WATER-HN SCIENTIFIC GATE PIPELINE: TECHNICAL PASS"
echo "Scientific point-estimate gate: $GATE_RESULT"
echo "Result: $AUDIT_JSON"
echo "Final test remains LOCKED pending explicit review."
echo "================================================================"
