#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path.home() / "projects/panoptic-scene-graph-generation"

ANNOTATION = ROOT / (
    "data/floodpsg/annotations/"
    "floodpsg_canonical_trainval_coarse8_groupstrict_v1.json"
)

DATA_ROOT = ROOT / "data/floodpsg"

CONFIG = ROOT / (
    "configs/floodpsg/"
    "masks-loc-sem-flood.json"
)

INDEX = ROOT / (
    "data/floodpsg/stats/"
    "dsformer_floodhn_sampling_v1/"
    "train_hardneg_index_v1.json"
)

SEEDED_RUNNER = ROOT / (
    "scripts/run_fair_psgg_seeded.py"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


parser = argparse.ArgumentParser()

parser.add_argument(
    "mode",
    choices=(
        "uniform_all",
        "flood_hn",
    ),
)

parser.add_argument(
    "stage",
    choices=(
        "smoke",
        "train",
    ),
)

parser.add_argument(
    "--hard-fraction",
    type=float,
    default=0.25,
)

parser.add_argument(
    "--zero-negatives-per-image",
    type=int,
    default=2,
)

parser.add_argument(
    "--seed",
    type=int,
    default=3407,
)

parser.add_argument(
    "--workers",
    type=int,
    default=4,
)

args = parser.parse_args()

if args.mode == "uniform_all":
    hard_fraction = 0.0
    tag = "uniform_all"
else:
    hard_fraction = args.hard_fraction
    percent = int(round(hard_fraction * 100))
    tag = f"floodhn_r{percent:02d}"

if not 0 <= hard_fraction <= 1:
    raise SystemExit(
        "--hard-fraction must be in [0, 1]"
    )

epochs = 1 if args.stage == "smoke" else 40

stage_tag = (
    "smoke"
    if args.stage == "smoke"
    else "40epoch"
)

output = ROOT / (
    f"outputs/flood_groupstrict_coarse8_"
    f"{tag}_seed{args.seed}_{stage_tag}_v1"
)

log = ROOT / (
    f"data/floodpsg/logs/"
    f"flood_groupstrict_coarse8_"
    f"{tag}_seed{args.seed}_{stage_tag}_v1.log"
)

manifest = ROOT / (
    f"data/floodpsg/stats/"
    f"dsformer_floodhn_sampling_v1/"
    f"{tag}_seed{args.seed}_{stage_tag}_manifest.json"
)

required = [
    ANNOTATION,
    DATA_ROOT,
    CONFIG,
    SEEDED_RUNNER,
]

if args.mode == "flood_hn":
    required.append(INDEX)

missing = [
    str(path)
    for path in required
    if not path.exists()
]

if missing:
    raise SystemExit(
        "Missing required paths:\n  "
        + "\n  ".join(missing)
    )

if output.exists() and any(
    output.iterdir()
):
    raise SystemExit(
        f"Output directory is not empty:\n"
        f"{output}\n"
        f"Use a new version instead of overwriting."
    )

output.parent.mkdir(
    parents=True,
    exist_ok=True,
)

log.parent.mkdir(
    parents=True,
    exist_ok=True,
)

manifest.parent.mkdir(
    parents=True,
    exist_ok=True,
)

environment = os.environ.copy()

# FLOODPSG_LAUNCHER_PYTHONPATH_V1
existing_pythonpath = environment.get(
    "PYTHONPATH",
    "",
)

environment["PYTHONPATH"] = str(ROOT)

if existing_pythonpath:
    environment["PYTHONPATH"] += (
        os.pathsep + existing_pythonpath
    )

environment.update(
    {
        "PYTHONUNBUFFERED": "1",
        "PYTHONHASHSEED": str(args.seed),
        "FLOODPSG_TRAIN_SEED": str(args.seed),
        "FLOODPSG_KEEP_ZERO_REL": "1",
        "FLOODPSG_NEG_MODE": args.mode,
        "FLOODPSG_ZERO_NEG_PER_IMAGE":
            str(args.zero_negatives_per_image),
        "FLOODPSG_HARDNEG_FRACTION":
            str(hard_fraction),
        "FLOODPSG_HARDNEG_INDEX":
            str(INDEX),
    }
)

command = [
    sys.executable,
    "-u",
    str(SEEDED_RUNNER),
    "--anno",
    str(ANNOTATION.relative_to(ROOT)),
    "--img",
    str(DATA_ROOT.relative_to(ROOT)),
    "--seg",
    str(DATA_ROOT.relative_to(ROOT)),
    "--epochs",
    str(epochs),
    "--workers",
    str(args.workers),
    str(CONFIG.relative_to(ROOT)),
    str(output.relative_to(ROOT)),
]

run_manifest = {
    "created_at_utc": utc_now(),
    "protocol": (
        "FloodPSG DSFormer negative-sampling "
        "experiment v1"
    ),
    "mode": args.mode,
    "stage": args.stage,
    "epochs": epochs,
    "seed": args.seed,
    "zero_negatives_per_image":
        args.zero_negatives_per_image,
    "hardneg_fraction": hard_fraction,
    "training_annotation": str(
        ANNOTATION.relative_to(ROOT)
    ),
    "config": str(
        CONFIG.relative_to(ROOT)
    ),
    "hard_negative_index": (
        str(INDEX.relative_to(ROOT))
        if args.mode == "flood_hn"
        else None
    ),
    "output": str(
        output.relative_to(ROOT)
    ),
    "log": str(
        log.relative_to(ROOT)
    ),
    "command": command,
    "environment": {
        key: environment[key]
        for key in (
            "PYTHONHASHSEED",
            "FLOODPSG_TRAIN_SEED",
            "FLOODPSG_KEEP_ZERO_REL",
            "FLOODPSG_NEG_MODE",
            "FLOODPSG_ZERO_NEG_PER_IMAGE",
            "FLOODPSG_HARDNEG_FRACTION",
            "FLOODPSG_HARDNEG_INDEX",
        )
    },
    "input_sha256": {
        str(ANNOTATION.relative_to(ROOT)):
            sha256(ANNOTATION),
        str(CONFIG.relative_to(ROOT)):
            sha256(CONFIG),
        str(SEEDED_RUNNER.relative_to(ROOT)):
            sha256(SEEDED_RUNNER),
    },
}

if args.mode == "flood_hn":
    run_manifest["input_sha256"][
        str(INDEX.relative_to(ROOT))
    ] = sha256(INDEX)

with manifest.open(
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        run_manifest,
        file,
        ensure_ascii=False,
        indent=2,
    )
    file.write("\n")

print("=" * 80)
print("DSFORMER FLOODPSG NEGATIVE-SAMPLING RUN")
print("=" * 80)
print("mode:", args.mode)
print("stage:", args.stage)
print("epochs:", epochs)
print("seed:", args.seed)
print(
    "zero negatives/image:",
    args.zero_negatives_per_image,
)
print(
    "hard-negative fraction:",
    hard_fraction,
)
print("annotation:", ANNOTATION)
print("output:", output)
print("log:", log)
print("manifest:", manifest)
print("final-test annotation is not passed")
print("command:")
print(" ".join(command))
print("=" * 80)

with log.open(
    "w",
    encoding="utf-8",
) as log_file:
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    assert process.stdout is not None

    for line in process.stdout:
        print(line, end="")
        log_file.write(line)
        log_file.flush()

    return_code = process.wait()

if return_code != 0:
    raise SystemExit(
        f"Training failed with exit code "
        f"{return_code}.\nLog: {log}"
    )

print("\nRun completed successfully.")
print("Output:", output)
print("Log:", log)
print(
    "Do not run final-test inference during "
    "sampler development."
)
