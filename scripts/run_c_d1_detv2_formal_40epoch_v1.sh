#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "$0")/.."

ROOT="$PWD"
PY="$ROOT/.venv/bin/python"

EXPECTED_BRANCH="exp/dsformer-fibe-d1-det-v2"

ANNO="$ROOT/data/floodpsg/annotations/floodpsg_canonical_trainval_coarse8_groupstrict_v1.json"

HN_INDEX="$ROOT/data/floodpsg/stats/dsformer_floodhn_sampling_v1/train_hardneg_index_v1.json"

AUDIT_DIR="$ROOT/data/floodpsg/stats/dsformer_fibe_d1_floodhn_r50_v1"

AUDIT_JSON="$AUDIT_DIR/D1_C_SAMPLING_EQUIVALENCE_AUDIT_V1.json"

FORMAL_DIR="$AUDIT_DIR/formal_detv2"

LOG_DIR="$ROOT/data/floodpsg/logs/formal_detv2"

MODE="${1:-}"

case "$MODE" in
  C)
    RUN_ID="C_DetV2_FloodHN_R50_Seed3407_40Epoch"

    CONFIG="$ROOT/outputs/flood_groupstrict_coarse8_floodhn_r50_seed3407_40epoch_v1/config.json"

    EXPECTED_CONFIG_SHA="31592941e501b59d80d7bed6ebe45e8dd1846a230a788cd57e77b1fda1f4de22"

    OUTPUT="$ROOT/outputs/flood_groupstrict_coarse8_floodhn_r50_detv2_seed3407_40epoch_v1"
    ;;

  D1)
    RUN_ID="D1_DetV2_FloodHN_R50_FIBE_Seed3407_40Epoch"

    CONFIG="$ROOT/configs/floodpsg/masks-loc-sem-flood-groupstrict-coarse8-fibe-d1-v1.json"

    EXPECTED_CONFIG_SHA="9da5cf81356e6e23f5fd2b7d03e7d07065ae2523359b08daca7a8b38cebf8c86"

    OUTPUT="$ROOT/outputs/flood_groupstrict_coarse8_floodhn_r50_fibe_d1_detv2_seed3407_40epoch_v1"
    ;;

  *)
    echo "Usage:"
    echo "  bash scripts/run_c_d1_detv2_formal_40epoch_v1.sh C"
    echo "  bash scripts/run_c_d1_detv2_formal_40epoch_v1.sh D1"
    exit 2
    ;;
esac

LOG="$LOG_DIR/${RUN_ID}.log"

PREFLIGHT="$FORMAL_DIR/${RUN_ID}_PREFLIGHT_V1.json"

POSTFLIGHT="$FORMAL_DIR/${RUN_ID}_POSTFLIGHT_LOCK_V1.txt"


check_sha256() {
  local expected="$1"
  local path="$2"

  local actual

  actual="$(
    sha256sum "$path" \
    | awk '{print $1}'
  )"

  if [ "$actual" != "$expected" ]; then
    echo "ERROR: SHA256 mismatch"
    echo "path=$path"
    echo "expected=$expected"
    echo "actual=$actual"
    exit 1
  fi

  echo "PASS SHA256: $path"
}


echo "================================================================"
echo "FLOODPSG FORMAL DETERMINISTIC V2 TRAINING"
echo "================================================================"
echo "mode=$MODE"
echo "run_id=$RUN_ID"
echo "branch=$(git branch --show-current)"
echo "commit=$(git rev-parse HEAD)"
echo "config=$CONFIG"
echo "annotation=$ANNO"
echo "hardneg_index=$HN_INDEX"
echo "sampling_audit=$AUDIT_JSON"
echo "output=$OUTPUT"
echo "log=$LOG"
echo "epochs=40"
echo "workers=4"
echo "seed=3407"
echo "from_scratch=true"
echo "raw_batch_trace=false"
echo "================================================================"


# ------------------------------------------------------------
# Branch and source-state checks
# ------------------------------------------------------------

CURRENT_BRANCH="$(
  git branch --show-current
)"

if [ "$CURRENT_BRANCH" != "$EXPECTED_BRANCH" ]; then
  echo "ERROR: wrong branch"
  echo "expected=$EXPECTED_BRANCH"
  echo "actual=$CURRENT_BRANCH"
  exit 1
fi

if [ -n "$(
  git status \
    --porcelain \
    --untracked-files=no
)" ]; then
  echo "ERROR: tracked working tree is not clean"
  git status --short
  exit 1
fi


# ------------------------------------------------------------
# Input existence checks
# ------------------------------------------------------------

for path in \
  "$PY" \
  "$ANNO" \
  "$HN_INDEX" \
  "$AUDIT_JSON" \
  "$CONFIG"
do
  test -e "$path" || {
    echo "ERROR: missing required path: $path"
    exit 1
  }
done


# ------------------------------------------------------------
# Refuse all overwrites
# ------------------------------------------------------------

test ! -e "$OUTPUT" || {
  echo "ERROR: formal output already exists:"
  echo "$OUTPUT"
  exit 1
}

test ! -e "$LOG" || {
  echo "ERROR: formal log already exists:"
  echo "$LOG"
  exit 1
}

test ! -e "$PREFLIGHT" || {
  echo "ERROR: preflight manifest already exists:"
  echo "$PREFLIGHT"
  exit 1
}

test ! -e "$POSTFLIGHT" || {
  echo "ERROR: postflight lock already exists:"
  echo "$POSTFLIGHT"
  exit 1
}

mkdir -p \
  "$FORMAL_DIR" \
  "$LOG_DIR"


# ------------------------------------------------------------
# Frozen input hashes
# ------------------------------------------------------------

check_sha256 \
  "28f4dfe34bc14b389a3cffb63e6d463888668743df67e78022287e36fe3f9354" \
  "$ANNO"

check_sha256 \
  "b893ba565fd8cfdb1137f59bed343e6fb54d746e46b20382c655c07be0c6691f" \
  "$HN_INDEX"

check_sha256 \
  "$EXPECTED_CONFIG_SHA" \
  "$CONFIG"

check_sha256 \
  "5f17a56221790a6bb450c79bfeee94cdf0ce7a96d8c5af228f97e1394dc594c1" \
  "$AUDIT_JSON"


# ------------------------------------------------------------
# Formal sampling-gate check
# ------------------------------------------------------------

"$PY" - "$AUDIT_JSON" "$CONFIG" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path


audit_path = Path(sys.argv[1])
config_path = Path(sys.argv[2])

audit = json.loads(
    audit_path.read_text(
        encoding="utf-8"
    )
)

if audit.get("status") != "PASS":
    raise SystemExit(
        "ERROR: formal sampling-equivalence "
        "audit is not PASS."
    )

required_zero_counters = [
    "raw_batch_count_mismatches",
    "raw_image_count_mismatches",
    "image_order_mismatches",
    "dataset_index_order_mismatches",
    "pair_order_mismatches",
    "positive_pair_mismatches",
    "negative_pair_mismatches",
    "hard_negative_flag_mismatches",
    "positive_budget_mismatches",
    "negative_budget_mismatches",
    "zero_relation_treatment_mismatches",
    "omitted_image_set_mismatches",
    "trace_config_mismatches",
    "internal_budget_failures",
    "duplicate_image_failures",
    "model_eligible_count_mismatches",
    "model_eligibility_partition_failures",
    "semantic_trace_hash_mismatches",
]

for key in required_zero_counters:
    value = int(audit.get(key, -1))

    if value != 0:
        raise SystemExit(
            f"ERROR: audit counter {key}={value}"
        )

config = json.loads(
    config_path.read_text(
        encoding="utf-8"
    )
)

model_state = config.get(
    "model_state"
)

if model_state not in {
    None,
    "",
}:
    raise SystemExit(
        "ERROR: config contains a non-empty "
        f"model_state: {model_state!r}"
    )

print(
    "FORMAL SAMPLING GATE: PASS"
)

print(
    "FROM-SCRATCH CONFIG CHECK: PASS"
)
PY


# ------------------------------------------------------------
# Write immutable preflight manifest
# ------------------------------------------------------------

export RUN_ID
export MODE
export CONFIG
export OUTPUT
export LOG
export PREFLIGHT
export ANNO
export HN_INDEX
export AUDIT_JSON

"$PY" - <<'PY'
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path


def sha256(path: str) -> str:
    digest = hashlib.sha256()

    with open(path, "rb") as handle:
        for block in iter(
            lambda: handle.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args],
        text=True,
    ).strip()


payload = {
    "protocol": (
        "FloodPSG formal deterministic "
        "training v2"
    ),
    "created_at": (
        datetime.now()
        .astimezone()
        .isoformat()
    ),
    "run_id": os.environ["RUN_ID"],
    "mode": os.environ["MODE"],
    "branch": git(
        "branch",
        "--show-current",
    ),
    "commit": git(
        "rev-parse",
        "HEAD",
    ),
    "from_scratch": True,
    "epochs": 40,
    "workers": 4,
    "seed": 3407,
    "output": os.environ["OUTPUT"],
    "log": os.environ["LOG"],
    "environment": {
        "PYTHONHASHSEED": "3407",
        "FLOODPSG_TRAIN_SEED": "3407",
        "FLOODPSG_KEEP_ZERO_REL": "1",
        "FLOODPSG_NEG_MODE": "flood_hn",
        "FLOODPSG_ZERO_NEG_PER_IMAGE": "2",
        "FLOODPSG_HARDNEG_FRACTION": "0.50",
        "FLOODPSG_DETERMINISTIC_LOADER": "1",
        "FLOODPSG_LOADER_SEED": "3407",
        "FLOODPSG_DETERMINISTIC_SAMPLING": "1",
        "FLOODPSG_SAMPLING_SEED": "3407",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    },
    "raw_batch_trace_enabled": False,
    "input_sha256": {
        "config": sha256(
            os.environ["CONFIG"]
        ),
        "annotation": sha256(
            os.environ["ANNO"]
        ),
        "hard_negative_index": sha256(
            os.environ["HN_INDEX"]
        ),
        "sampling_equivalence_audit": sha256(
            os.environ["AUDIT_JSON"]
        ),
    },
}

path = Path(
    os.environ["PREFLIGHT"]
)

path.write_text(
    json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)

print(
    "preflight manifest:",
    path,
)
PY


# ------------------------------------------------------------
# Ensure trace-only variables cannot leak into formal run
# ------------------------------------------------------------

unset FLOODPSG_RAW_BATCH_TRACE_DIR || true
unset FLOODPSG_TRACE_RUN_ID || true
unset FLOODPSG_TRACE_CODE_COMMIT || true
unset FLOODPSG_CURRENT_EPOCH || true


# ------------------------------------------------------------
# Formal 40-epoch run
# ------------------------------------------------------------

set -o pipefail

/usr/bin/time -v \
env \
  -u FLOODPSG_RAW_BATCH_TRACE_DIR \
  -u FLOODPSG_TRACE_RUN_ID \
  -u FLOODPSG_TRACE_CODE_COMMIT \
  -u FLOODPSG_CURRENT_EPOCH \
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
  OMP_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 \
"$PY" -u \
  scripts/run_fair_psgg_seeded.py \
  "$CONFIG" \
  "$OUTPUT" \
  --anno "$ANNO" \
  --img data/floodpsg \
  --seg data/floodpsg \
  --epochs 40 \
  --workers 4 \
  2>&1 | tee "$LOG"

STATUS=${PIPESTATUS[0]}

if [ "$STATUS" -ne 0 ]; then
  echo "ERROR: formal training failed"
  echo "status=$STATUS"
  exit "$STATUS"
fi


# ------------------------------------------------------------
# Required completion artifacts
# ------------------------------------------------------------

test -f "$OUTPUT/done.txt" || {
  echo "ERROR: done.txt missing"
  exit 1
}

grep -qx "done" "$OUTPUT/done.txt" || {
  echo "ERROR: invalid done.txt"
  exit 1
}

test -f "$OUTPUT/best_state.pth" || {
  echo "ERROR: best_state.pth missing"
  exit 1
}

test -f "$OUTPUT/config.json" || {
  echo "ERROR: saved config.json missing"
  exit 1
}


# ------------------------------------------------------------
# Postflight lock
# ------------------------------------------------------------

{
  echo "FLOODPSG FORMAL DETV2 POSTFLIGHT LOCK"
  echo
  echo "date=$(date --iso-8601=seconds)"
  echo "mode=$MODE"
  echo "run_id=$RUN_ID"
  echo "branch=$(git branch --show-current)"
  echo "commit=$(git rev-parse HEAD)"
  echo "output=$OUTPUT"
  echo "log=$LOG"
  echo

  echo "===== COMPLETION ====="
  cat "$OUTPUT/done.txt"
  echo

  echo "===== BEST VALIDATION LINE ====="
  grep -F \
    "Best value for rel_mean_recall/50:" \
    "$LOG" \
    | tail -n 1 \
    || true

  echo
  echo "===== REQUIRED FILE SHA256 ====="

  sha256sum \
    "$OUTPUT/done.txt" \
    "$OUTPUT/config.json" \
    "$OUTPUT/best_state.pth" \
    "$PREFLIGHT" \
    "$LOG"

  echo
  echo "===== OUTPUT INVENTORY ====="

  find "$OUTPUT" \
    -maxdepth 1 \
    -type f \
    -printf '%f\t%s bytes\n' \
    | sort

} | tee "$POSTFLIGHT"

echo
echo "================================================================"
echo "$RUN_ID FORMAL 40-EPOCH TRAINING: PASS"
echo "================================================================"
