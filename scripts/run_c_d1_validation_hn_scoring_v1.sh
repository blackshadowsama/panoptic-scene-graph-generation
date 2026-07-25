#!/usr/bin/env bash

set -euo pipefail

PROJECT="${PROJECT:-$HOME/projects/panoptic-scene-graph-generation}"
cd "$PROJECT"

PY="$PROJECT/.venv/bin/python"
SCORER="$PROJECT/scripts/score_frozen_validation_hn_gtpairs_v1.py"
EVALUATOR="$PROJECT/scripts/evaluate_dsformer_hardneg_strict_v1.py"

EXPECTED_BRANCH="exp/dsformer-fibe-d1-det-v2"

C_MODEL="$PROJECT/outputs/flood_groupstrict_coarse8_floodhn_r50_detv2_seed3407_40epoch_v1"
D1_MODEL="$PROJECT/outputs/flood_groupstrict_coarse8_floodhn_r50_fibe_d1_detv2_seed3407_40epoch_v1"

ANNOTATION="$PROJECT/data/floodpsg/annotations/floodpsg_canonical_trainval_coarse8_groupstrict_v1.json"
HARDNEG="$PROJECT/data/floodpsg/stats/frozen_hardneg_eval_validation_v1/02_validation_hardneg_water.csv"
DATA_ROOT="$PROJECT/data/floodpsg"

OUTPUT_ROOT="$PROJECT/data/floodpsg/stats/dsformer_fibe_d1_floodhn_r50_v1/validation_hn_gtpair_scoring_v1"

C_RESULTS="$OUTPUT_ROOT/C_DetV2_validation_hn_gtpairs_v1.pkl"
D1_RESULTS="$OUTPUT_ROOT/D1_DetV2_validation_hn_gtpairs_v1.pkl"
C_REPORT="$OUTPUT_ROOT/C_DetV2_validation_hn_gtpairs_v1.json"
D1_REPORT="$OUTPUT_ROOT/D1_DetV2_validation_hn_gtpairs_v1.json"
C_SCORE_LOG="$OUTPUT_ROOT/C_DetV2_validation_hn_gtpairs_v1.log"
D1_SCORE_LOG="$OUTPUT_ROOT/D1_DetV2_validation_hn_gtpairs_v1.log"

C_EVAL="$OUTPUT_ROOT/C_DetV2_strict_eval_v1"
D1_EVAL="$OUTPUT_ROOT/D1_DetV2_strict_eval_v1"
C_EVAL_LOG="$OUTPUT_ROOT/C_DetV2_strict_eval_v1.log"
D1_EVAL_LOG="$OUTPUT_ROOT/D1_DetV2_strict_eval_v1.log"

POINT_GATE="$OUTPUT_ROOT/C_D1_VALIDATION_HN_POINT_ESTIMATE_GATE_V1.json"
LOCK_FILE="$OUTPUT_ROOT/C_D1_VALIDATION_HN_GTPAIR_SCORING_LOCK_V1.txt"

EXPECTED_SCORER_SHA="6682526c267c5f02531d3a3055e4f937dcf58d4aab2e1b3e07f4ce8f42ffd5bc"
EXPECTED_EVALUATOR_SHA="8ab797bb4b95dac7cd0afe9b7e4365fc23753a9bfa811409701d00dd3be55ae5"
EXPECTED_ANNOTATION_SHA="28f4dfe34bc14b389a3cffb63e6d463888668743df67e78022287e36fe3f9354"
EXPECTED_HARDNEG_SHA="1ef449f61d3e3bc0f6a56cbab038ec4264d43e775bd20bbf6006e3458b5ae47c"
EXPECTED_C_CONFIG_SHA="4c005b4424e9d6fa60c356e18b19d2d92482e2d9f069dde3fdc658c9c13b3c8a"
EXPECTED_D1_CONFIG_SHA="bd095e9c4b04d7b7a39d6ddc73e55424d90da3c926ff7031167fd562ef94ad80"
EXPECTED_C_CHECKPOINT_SHA="ceba5a6feb7487d0c929efc68fff1c1427350111320c2be5dba7c4f4ccb6f480"
EXPECTED_D1_CHECKPOINT_SHA="e6d158e41ac2a3cb933941a457cde19d3512a08a7263eeeab7cf521dfa80f492"

timestamp() {
  date --iso-8601=seconds
}

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

require_file() {
  local path="$1"
  [ -f "$path" ] || fail "missing required file: $path"
}

require_sha256() {
  local path="$1"
  local expected="$2"
  require_file "$path"
  local actual
  actual="$(sha256sum "$path" | awk '{print $1}')"
  if [ "$actual" != "$expected" ]; then
    fail "SHA256 mismatch: $path expected=$expected actual=$actual"
  fi
  echo "PASS SHA256: $path"
}

run_logged() {
  local log_path="$1"
  shift

  set +e
  /usr/bin/time -v "$@" 2>&1 | tee "$log_path"
  local status=${PIPESTATUS[0]}
  set -e

  if [ "$status" -ne 0 ]; then
    fail "command failed with status $status; log=$log_path"
  fi
}

echo "================================================================"
echo "FLOODPSG C/D1 VALIDATION WATER-HN GT-PAIR SCORING V1"
echo "================================================================"
echo "date=$(timestamp)"
echo "project=$PROJECT"
echo "branch=$(git branch --show-current)"
echo "commit=$(git rev-parse HEAD)"
echo "output_root=$OUTPUT_ROOT"
echo "split=validation"
echo "expected_images=75"
echo "expected_pairs=322"
echo "final_test_allowed=false"
echo "================================================================"

[ -x "$PY" ] || fail "Python interpreter is unavailable: $PY"

current_branch="$(git branch --show-current)"
[ "$current_branch" = "$EXPECTED_BRANCH" ] ||   fail "unexpected branch: $current_branch"

if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  git status --short --untracked-files=no
  fail "tracked working tree is not clean"
fi

[ ! -e "$OUTPUT_ROOT" ] ||   fail "formal output root already exists: $OUTPUT_ROOT"

require_sha256 "$SCORER" "$EXPECTED_SCORER_SHA"
require_sha256 "$EVALUATOR" "$EXPECTED_EVALUATOR_SHA"
require_sha256 "$ANNOTATION" "$EXPECTED_ANNOTATION_SHA"
require_sha256 "$HARDNEG" "$EXPECTED_HARDNEG_SHA"
require_sha256 "$C_MODEL/config.json" "$EXPECTED_C_CONFIG_SHA"
require_sha256 "$D1_MODEL/config.json" "$EXPECTED_D1_CONFIG_SHA"
require_sha256 "$C_MODEL/best_state.pth" "$EXPECTED_C_CHECKPOINT_SHA"
require_sha256 "$D1_MODEL/best_state.pth" "$EXPECTED_D1_CHECKPOINT_SHA"

"$PY" -m py_compile "$SCORER" "$EVALUATOR"

"$PY" - "$C_MODEL/config.json" "$D1_MODEL/config.json" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

c = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
d1 = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))

assert c["fibe"]["enabled"] is False
assert d1["fibe"]["enabled"] is True
assert d1["fibe"]["validation_cache"]
assert Path(d1["fibe"]["validation_cache"]).is_file()
assert c["architecture"] == d1["architecture"]
assert c["extractor"] == d1["extractor"]
assert c["data"] == d1["data"]

print("C/D1 CONFIG IDENTITY AND FIBE DELTA CHECK: PASS")
PY

mkdir -p "$OUTPUT_ROOT"

COMMON_ENV=(
  env
  PYTHONPATH="$PROJECT${PYTHONPATH:+:$PYTHONPATH}"
  PYTHONHASHSEED=3407
  OMP_NUM_THREADS=1
  MKL_NUM_THREADS=1
)

echo
echo "===== SCORE C-DET ====="

run_logged "$C_SCORE_LOG"   "${COMMON_ENV[@]}"   "$PY" -u "$SCORER"   --model-dir "$C_MODEL"   --annotation "$ANNOTATION"   --hardneg "$HARDNEG"   --data-root "$DATA_ROOT"   --output "$C_RESULTS"   --report "$C_REPORT"   --split validation   --expected-images 75   --expected-pairs 322   --batch-size 1   --workers 0   --max-relations 512   --seed 3407   --device cuda   --expect-fibe disabled

echo
echo "===== STRICT EVALUATE C-DET ====="

run_logged "$C_EVAL_LOG"   "${COMMON_ENV[@]}"   "$PY" -u "$EVALUATOR"   --results "$C_RESULTS"   --annotation "$ANNOTATION"   --hardneg "$HARDNEG"   --output "$C_EVAL"   --data-root "$DATA_ROOT"   --split validation   --expected-pairs 322   --expected-images 75   --strict-complete

echo
echo "===== SCORE D1-DET ====="

run_logged "$D1_SCORE_LOG"   "${COMMON_ENV[@]}"   "$PY" -u "$SCORER"   --model-dir "$D1_MODEL"   --annotation "$ANNOTATION"   --hardneg "$HARDNEG"   --data-root "$DATA_ROOT"   --output "$D1_RESULTS"   --report "$D1_REPORT"   --split validation   --expected-images 75   --expected-pairs 322   --batch-size 1   --workers 0   --max-relations 512   --seed 3407   --device cuda   --expect-fibe enabled   --require-all-fibe-valid

echo
echo "===== STRICT EVALUATE D1-DET ====="

run_logged "$D1_EVAL_LOG"   "${COMMON_ENV[@]}"   "$PY" -u "$EVALUATOR"   --results "$D1_RESULTS"   --annotation "$ANNOTATION"   --hardneg "$HARDNEG"   --output "$D1_EVAL"   --data-root "$DATA_ROOT"   --split validation   --expected-pairs 322   --expected-images 75   --strict-complete

echo
echo "===== C/D1 POINT-ESTIMATE GATE ====="

"$PY" -   "$C_REPORT"   "$D1_REPORT"   "$C_EVAL/summary.json"   "$D1_EVAL/summary.json"   "$POINT_GATE" <<'PY'
from __future__ import annotations

import json
import math
import sys
from pathlib import Path


c_report_path = Path(sys.argv[1])
d1_report_path = Path(sys.argv[2])
c_summary_path = Path(sys.argv[3])
d1_summary_path = Path(sys.argv[4])
output_path = Path(sys.argv[5])

c_report = json.loads(c_report_path.read_text(encoding="utf-8"))
d1_report = json.loads(d1_report_path.read_text(encoding="utf-8"))
c_summary = json.loads(c_summary_path.read_text(encoding="utf-8"))
d1_summary = json.loads(d1_summary_path.read_text(encoding="utf-8"))

for label, report in (("C", c_report), ("D1", d1_report)):
    assert report["status"] == "PASS", (label, report["status"])
    assert report["split"] == "validation"
    assert report["counts"]["output_images"] == 75
    assert report["counts"]["output_pairs"] == 322
    assert report["counts"]["nonfinite_score_values"] == 0

assert c_report["fibe"]["enabled"] is False
assert d1_report["fibe"]["enabled"] is True
assert d1_report["fibe"]["selected_valid_pairs"] == 322
assert d1_report["fibe"]["selected_invalid_pairs"] == 0
assert (
    c_report["input_semantic_sha256"]
    == d1_report["input_semantic_sha256"]
), (
    c_report["input_semantic_sha256"],
    d1_report["input_semantic_sha256"],
)

for label, summary in (("C", c_summary), ("D1", d1_summary)):
    assert summary["status"] == "PASS", (label, summary["status"])
    assert summary["split"] == "validation"
    assert summary["strict_complete"] is True
    assert summary["total_frozen_water_hard_negative_images"] == 75
    assert summary["total_frozen_water_hard_negative_pairs"] == 322
    assert summary["scored_pairs"] == 322
    assert summary["unscored_pairs"] == 0
    assert summary["coverage_gate"]["status"] == "PASS"
    assert all(summary["coverage_gate"]["checks"].values())

metric_key = {
    "0.5": "formal_max_positive_fpr_at_0.5",
    "0.7": "formal_max_positive_fpr_at_0.7",
    "0.9": "formal_max_positive_fpr_at_0.9",
}

c_metrics = c_summary["pair_level_metrics_on_scored_pairs"]
d1_metrics = d1_summary["pair_level_metrics_on_scored_pairs"]

values = {}
for threshold, key in metric_key.items():
    c_value = float(c_metrics[key])
    d1_value = float(d1_metrics[key])
    if not (math.isfinite(c_value) and math.isfinite(d1_value)):
        raise RuntimeError(f"Nonfinite FPR at threshold {threshold}")
    values[threshold] = {
        "c_fpr": c_value,
        "d1_fpr": d1_value,
        "improvement_c_minus_d1": c_value - d1_value,
        "improvement_percentage_points": 100.0 * (c_value - d1_value),
    }

checks = {
    "technical_scoring_and_coverage_pass": True,
    "base_input_semantic_sha256_equal": (
        c_report["input_semantic_sha256"]
        == d1_report["input_semantic_sha256"]
    ),
    "d1_all_322_fibe_pairs_valid": (
        d1_report["fibe"]["selected_valid_pairs"] == 322
        and d1_report["fibe"]["selected_invalid_pairs"] == 0
    ),
    "fpr_at_0_5_improves_at_least_5pp": (
        values["0.5"]["improvement_c_minus_d1"] >= 0.05
    ),
    "fpr_at_0_7_same_direction": (
        values["0.7"]["d1_fpr"] <= values["0.7"]["c_fpr"]
    ),
    "fpr_at_0_9_same_direction": (
        values["0.9"]["d1_fpr"] <= values["0.9"]["c_fpr"]
    ),
}

point_estimate_pass = all(checks.values())
report = {
    "protocol": (
        "FloodPSG C-Det vs D1-Det validation Water-HN "
        "point-estimate gate v1"
    ),
    "scope": "validation only; final test remains locked",
    "images": 75,
    "pairs": 322,
    "threshold_metrics": values,
    "input_semantic_sha256": c_report["input_semantic_sha256"],
    "checks": checks,
    "point_estimate_gate_status": (
        "PASS" if point_estimate_pass else "FAIL"
    ),
    "overall_d2_gate_status": "PENDING",
    "pending": [
        "paired image-cluster bootstrap CI",
        "FIBE zeroing counterfactual",
        "FIBE shuffle counterfactual",
        "image/family concentration analysis",
        "D1 versus C-Adapter",
    ],
    "final_test_allowed": False,
}

output_path.write_text(
    json.dumps(
        report,
        ensure_ascii=False,
        indent=2,
        allow_nan=False,
    )
    + "\n",
    encoding="utf-8",
)

print(json.dumps(report, ensure_ascii=False, indent=2))
print()
print(
    "VALIDATION WATER-HN POINT-ESTIMATE GATE: "
    + report["point_estimate_gate_status"]
)
PY

{
  echo "FLOODPSG C/D1 VALIDATION WATER-HN GT-PAIR SCORING LOCK V1"
  echo
  echo "date=$(timestamp)"
  echo "branch=$(git branch --show-current)"
  echo "commit=$(git rev-parse HEAD)"
  echo "development_split=validation"
  echo "images=75"
  echo "pairs=322"
  echo "final_test_allowed=false"
  echo
  echo "===== SOURCE SHA256 ====="
  sha256sum "$SCORER" "$EVALUATOR"
  echo
  echo "===== MODEL AND INPUT SHA256 ====="
  sha256sum     "$C_MODEL/config.json"     "$C_MODEL/best_state.pth"     "$D1_MODEL/config.json"     "$D1_MODEL/best_state.pth"     "$ANNOTATION"     "$HARDNEG"
  echo
  echo "===== SCORING OUTPUT SHA256 ====="
  sha256sum     "$C_RESULTS"     "$C_REPORT"     "$D1_RESULTS"     "$D1_REPORT"
  echo
  echo "===== STRICT SUMMARY SHA256 ====="
  sha256sum     "$C_EVAL/summary.json"     "$D1_EVAL/summary.json"     "$POINT_GATE"
  echo
  echo "===== POINT-ESTIMATE GATE ====="
  cat "$POINT_GATE"
} | tee "$LOCK_FILE"

echo
echo "================================================================"
echo "C/D1 VALIDATION WATER-HN GT-PAIR SCORING PIPELINE: PASS"
echo "Point-estimate gate is recorded in:"
echo "$POINT_GATE"
echo "Final test remains LOCKED."
echo "================================================================"
