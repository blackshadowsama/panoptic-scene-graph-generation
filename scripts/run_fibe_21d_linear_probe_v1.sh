#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${HOME}/projects/panoptic-scene-graph-generation"
PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
PROBE_SCRIPT="${REPO_ROOT}/scripts/run_fibe_21d_linear_probe_v1.py"

TRAIN_CACHE="${REPO_ROOT}/data/floodpsg/features/fibe_scalar_v1/train_features_metadata_v1.pt"
VALIDATION_CACHE="${REPO_ROOT}/data/floodpsg/features/fibe_scalar_v1/validation_features_metadata_v1.pt"
ANNOTATION="${REPO_ROOT}/data/floodpsg/annotations/floodpsg_canonical_trainval_coarse8_groupstrict_v1.json"
HN_SOURCE="${REPO_ROOT}/data/floodpsg/stats/hard_negative_audit_v4/01_all_rows_canonical_v4.csv"
FROZEN_VALIDATION_HN="${REPO_ROOT}/data/floodpsg/stats/frozen_hardneg_eval_validation_v1/02_validation_hardneg_water.csv"

PROBE_ROOT="${REPO_ROOT}/data/floodpsg/stats/dsformer_fibe_d1_floodhn_r50_v1/d1_diagnostics_v1/linear_probe_v1"
LOCK_PATH="${PROBE_ROOT}/FIBE_21D_LINEAR_PROBE_PIPELINE_LOCK_V1.txt"

EXPECTED_PROBE_SCRIPT_SHA="50dbc77b57a0d71e633dc2ece0ed925fd1023db58278a8eaf9ea9e8b94e2e593"
EXPECTED_TRAIN_CACHE_SHA="8dea40e299d5f12163374fcc5e72ef97a3dda52a09eaa2d2cc3cc78d2d75f926"
EXPECTED_VALIDATION_CACHE_SHA="8bb894f50ce3cd767b788cfd5e361d35dff011a66c57b753db4c61a333e246be"
EXPECTED_ANNOTATION_SHA="28f4dfe34bc14b389a3cffb63e6d463888668743df67e78022287e36fe3f9354"
EXPECTED_HN_SOURCE_SHA="2bcaf823bb91fabe48a50911298ae6f9571e8ccd47cae4fbd16043ff5e163569"
EXPECTED_FROZEN_VALIDATION_HN_SHA="1ef449f61d3e3bc0f6a56cbab038ec4264d43e775bd20bbf6006e3458b5ae47c"

check_sha() {
  local expected="$1"
  local path="$2"
  test -f "$path" || {
    echo "FAIL missing file: $path" >&2
    exit 1
  }
  local actual
  actual="$(sha256sum "$path" | awk '{print $1}')"
  test "$actual" = "$expected" || {
    echo "FAIL SHA256: $path" >&2
    echo "expected=$expected" >&2
    echo "actual=$actual" >&2
    exit 1
  }
  echo "PASS SHA256: $path"
}

cd "$REPO_ROOT"

test -x "$PYTHON_BIN" || {
  echo "FAIL Python interpreter: $PYTHON_BIN" >&2
  exit 1
}

test ! -e "$PROBE_ROOT" || {
  echo "STOP: formal linear-probe output already exists:" >&2
  echo "$PROBE_ROOT" >&2
  exit 1
}

check_sha "$EXPECTED_PROBE_SCRIPT_SHA" "$PROBE_SCRIPT"
check_sha "$EXPECTED_TRAIN_CACHE_SHA" "$TRAIN_CACHE"
check_sha "$EXPECTED_VALIDATION_CACHE_SHA" "$VALIDATION_CACHE"
check_sha "$EXPECTED_ANNOTATION_SHA" "$ANNOTATION"
check_sha "$EXPECTED_HN_SOURCE_SHA" "$HN_SOURCE"
check_sha "$EXPECTED_FROZEN_VALIDATION_HN_SHA" "$FROZEN_VALIDATION_HN"

"$PYTHON_BIN" - <<'PY'
import sklearn
import numpy
import pandas
import torch
print("PASS dependencies")
print("sklearn=", sklearn.__version__)
print("numpy=", numpy.__version__)
print("pandas=", pandas.__version__)
print("torch=", torch.__version__)
PY

echo "================================================================"
echo "FLOODPSG FIBE 21D MATCHED LINEAR PROBE V1"
echo "branch=$(git branch --show-current)"
echo "commit=$(git rev-parse HEAD)"
echo "fit_split=train"
echo "evaluation_split=validation"
echo "frozen_validation_hn_pairs=322"
echo "bootstrap_cluster=global_image_key"
echo "bootstrap_iterations=2000"
echo "seed=3407"
echo "final_test_allowed=false"
echo "================================================================"

/usr/bin/time -v \
  env \
    PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    PYTHONHASHSEED=3407 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    "$PYTHON_BIN" -u "$PROBE_SCRIPT" \
      --train-cache "$TRAIN_CACHE" \
      --validation-cache "$VALIDATION_CACHE" \
      --annotation "$ANNOTATION" \
      --hardneg-source "$HN_SOURCE" \
      --frozen-validation-hn "$FROZEN_VALIDATION_HN" \
      --output-dir "$PROBE_ROOT" \
      --seed 3407 \
      --bootstrap-iterations 2000 \
      --expected-train-images 1500 \
      --expected-validation-images 173 \
      --expected-validation-hn-pairs 322

"$PYTHON_BIN" - "$PROBE_ROOT" <<'PY'
from __future__ import annotations
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
report = json.loads((root / "FIBE_21D_LINEAR_PROBE_V1.json").read_text(encoding="utf-8"))
audit = json.loads((root / "FIBE_21D_LINEAR_PROBE_DATASET_AUDIT_V1.json").read_text(encoding="utf-8"))
bootstrap = json.loads((root / "FIBE_21D_IMAGE_CLUSTER_BOOTSTRAP_V1.json").read_text(encoding="utf-8"))

checks = {
    "main_report_pass": report.get("status") == "PASS",
    "dataset_audit_pass": audit.get("status") == "PASS",
    "bootstrap_pass": bootstrap.get("status") == "PASS",
    "final_test_locked_main": report.get("final_test_allowed") is False,
    "final_test_locked_audit": audit.get("final_test_allowed") is False,
    "validation_hn_exact_322": audit["final_rows"]["validation_negative"] == 322,
    "train_validation_image_overlap_zero": audit["leakage_checks"]["train_validation_image_overlap"] == 0,
    "train_validation_pair_overlap_zero": audit["leakage_checks"]["train_validation_pair_overlap"] == 0,
    "all_three_probes_present": set(report["probes"]) == {"G", "F", "GF"},
    "bootstrap_2000": all(
        item["requested_iterations"] == 2000
        for item in bootstrap["probes"].values()
    ),
}
status = "PASS" if all(checks.values()) else "FAIL"
print(json.dumps({"checks": checks, "status": status}, indent=2))
if status != "PASS":
    raise SystemExit(1)
PY

{
  echo "FLOODPSG FIBE 21D MATCHED LINEAR PROBE PIPELINE LOCK V1"
  echo
  echo "date=$(date --iso-8601=seconds)"
  echo "branch=$(git branch --show-current)"
  echo "commit=$(git rev-parse HEAD)"
  echo
  echo "fit_split=train"
  echo "evaluation_split=validation"
  echo "seed=3407"
  echo "bootstrap_cluster=global_image_key"
  echo "bootstrap_iterations=2000"
  echo "frozen_validation_hn_pairs=322"
  echo "final_test_allowed=false"
  echo
  echo "===== SOURCE SHA256 ====="
  sha256sum \
    "$PROBE_SCRIPT" \
    "$TRAIN_CACHE" \
    "$VALIDATION_CACHE" \
    "$ANNOTATION" \
    "$HN_SOURCE" \
    "$FROZEN_VALIDATION_HN"
  echo
  echo "===== OUTPUT SHA256 ====="
  find "$PROBE_ROOT" -maxdepth 1 -type f ! -name "$(basename "$LOCK_PATH")" -print0 \
    | sort -z \
    | xargs -0 sha256sum
} > "$LOCK_PATH"

cat "$LOCK_PATH"
sha256sum "$LOCK_PATH"

echo "================================================================"
echo "FIBE 21D MATCHED LINEAR PROBE PIPELINE: PASS"
echo "Result: $PROBE_ROOT/FIBE_21D_LINEAR_PROBE_V1.json"
echo "Final test remains LOCKED."
echo "================================================================"
