#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
from pathlib import Path
from typing import Any

import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


PROTOCOL = "FloodPSG D1-v2a 3-epoch checkpoint-selection protocol audit V1"
CRITICAL_METRIC = "rel_mean_recall/50"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_torch(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def close(a: float, b: float, tol: float = 1e-7) -> bool:
    return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=tol)


def load_scalars(output_dir: Path) -> tuple[dict[str, list[dict[str, float]]], list[str]]:
    event_files = sorted(output_dir.glob("events.out.tfevents.*"))
    if not event_files:
        raise FileNotFoundError(f"No TensorBoard event file in {output_dir}")

    accumulator = EventAccumulator(
        str(output_dir),
        size_guidance={"scalars": 0},
    )
    accumulator.Reload()
    tags = sorted(accumulator.Tags().get("scalars", []))

    scalars: dict[str, list[dict[str, float]]] = {}
    for tag in tags:
        scalars[tag] = [
            {
                "step": int(event.step),
                "value": float(event.value),
                "wall_time": float(event.wall_time),
            }
            for event in accumulator.Scalars(tag)
        ]

    return scalars, tags


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def audit_run(
    *,
    label: str,
    output_dir: Path,
    expected_epochs: int,
    expected_workers: int,
    expected_commit: str,
    expected_annotation_sha: str,
    expected_mode: str,
) -> dict[str, Any]:
    required = {
        "args": output_dir / "args.json",
        "config": output_dir / "config.json",
        "commit": output_dir / "commit.txt",
        "best_state": output_dir / "best_state.pth",
        "last_state": output_dir / "last_state.pth",
        "best_metrics": output_dir / "best_metrics.pth",
        "done": output_dir / "done.txt",
    }

    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"{label}: missing output files {missing}")

    args = read_json(required["args"])
    config = read_json(required["config"])
    commit = required["commit"].read_text(encoding="utf-8").strip()
    done = required["done"].read_text(encoding="utf-8").strip()
    best_state = load_torch(required["best_state"])
    last_state = load_torch(required["last_state"])
    best_metrics = load_torch(required["best_metrics"])
    scalars, scalar_tags = load_scalars(output_dir)

    if CRITICAL_METRIC not in scalars:
        raise RuntimeError(
            f"{label}: TensorBoard has no {CRITICAL_METRIC}; tags={scalar_tags}"
        )

    critical_events = scalars[CRITICAL_METRIC]
    steps = [item["step"] for item in critical_events]
    values = [item["value"] for item in critical_events]

    if len(values) != expected_epochs:
        raise RuntimeError(
            f"{label}: expected {expected_epochs} critical-metric values, "
            f"got {len(values)}"
        )

    expected_steps = list(range(expected_epochs))
    first_max_index = max(range(len(values)), key=lambda index: values[index])
    selected_step = steps[first_max_index]
    selected_value = values[first_max_index]

    args_annotation = Path(args["anno"]).expanduser().resolve()
    if not args_annotation.is_file():
        raise FileNotFoundError(args_annotation)

    fibe = config.get("fibe", {})
    mode = str(fibe.get("mode", "token_residual"))
    enabled = bool(fibe.get("enabled", False))

    mode_checks = {
        "C": (not enabled),
        "D1V2A": (
            enabled
            and mode == "geometry_existence_gate"
            and fibe.get("test_cache") is None
        ),
    }

    forbidden_metric_tags = [
        tag
        for tag in scalar_tags
        if any(
            token in tag.lower()
            for token in (
                "water_hn",
                "hardneg",
                "hard_negative",
                "fpr",
                "frozen_hn",
            )
        )
    ]

    checks = {
        "output_done": done == "done",
        "commit_matches_current_run": commit == expected_commit,
        "args_epochs_3": int(args["epochs"]) == expected_epochs,
        "args_workers_4": int(args["workers"]) == expected_workers,
        "args_model_state_none": args.get("model_state") is None,
        "annotation_is_trainval_not_final_test": (
            "trainval" in args_annotation.name.lower()
            and "final_test" not in args_annotation.name.lower()
        ),
        "annotation_sha_matches": sha256(args_annotation) == expected_annotation_sha,
        "critical_metric_has_exact_steps": steps == expected_steps,
        "best_state_epoch_matches_first_max": int(best_state["epoch"]) == selected_step,
        "best_state_metric_matches_event": close(
            float(best_state["metric"]), selected_value
        ),
        "best_metrics_matches_event": close(
            float(best_metrics[CRITICAL_METRIC]), selected_value
        ),
        "last_state_epoch_is_2": int(last_state["epoch"]) == expected_epochs - 1,
        "mode_contract": mode_checks[expected_mode],
        "no_forbidden_hn_selection_tags": len(forbidden_metric_tags) == 0,
        "changes_patch_absent": not (output_dir / "changes.patch").exists(),
    }

    return {
        "label": label,
        "output_dir": str(output_dir),
        "output_files_sha256": {
            name: sha256(path)
            for name, path in required.items()
        },
        "args": args,
        "config_sha256": sha256(required["config"]),
        "commit": commit,
        "critical_metric": CRITICAL_METRIC,
        "critical_metric_events": critical_events,
        "selected_best_epoch": selected_step,
        "selected_best_value": selected_value,
        "best_state_metric": float(best_state["metric"]),
        "best_metrics_value": float(best_metrics[CRITICAL_METRIC]),
        "last_state_epoch": int(last_state["epoch"]),
        "scalar_tags": scalar_tags,
        "forbidden_hn_selection_tags": forbidden_metric_tags,
        "checks": checks,
        "status": "PASS" if all(checks.values()) else "FAIL",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--c-output", type=Path, required=True)
    parser.add_argument("--d1v2a-output", type=Path, required=True)
    parser.add_argument("--trainer-source", type=Path, required=True)
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--expected-epochs", type=int, default=3)
    parser.add_argument("--expected-workers", type=int, default=4)
    args = parser.parse_args()

    output_path = args.output.expanduser().resolve()
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite {output_path}")

    trainer_source = args.trainer_source.expanduser().resolve()
    annotation = args.annotation.expanduser().resolve()
    if not trainer_source.is_file():
        raise FileNotFoundError(trainer_source)
    if not annotation.is_file():
        raise FileNotFoundError(annotation)

    trainer_text = trainer_source.read_text(encoding="utf-8")
    source_checks = {
        "critical_metric_assignment_present": (
            'self.critical_metric = "rel_mean_recall/50"' in trainer_text
        ),
        "checkpoint_reads_critical_metric": (
            "crit_value = metrics[self.critical_metric]" in trainer_text
        ),
        "best_checkpoint_uses_strict_greater": (
            "crit_value > self.best_metric_value" in trainer_text
        ),
    }

    annotation_sha = sha256(annotation)

    c_report = audit_run(
        label="C",
        output_dir=args.c_output.expanduser().resolve(),
        expected_epochs=args.expected_epochs,
        expected_workers=args.expected_workers,
        expected_commit=args.expected_commit,
        expected_annotation_sha=annotation_sha,
        expected_mode="C",
    )

    d_report = audit_run(
        label="D1V2A",
        output_dir=args.d1v2a_output.expanduser().resolve(),
        expected_epochs=args.expected_epochs,
        expected_workers=args.expected_workers,
        expected_commit=args.expected_commit,
        expected_annotation_sha=annotation_sha,
        expected_mode="D1V2A",
    )

    checks = {
        **{f"trainer_source/{key}": value for key, value in source_checks.items()},
        "C_pass": c_report["status"] == "PASS",
        "D1V2A_pass": d_report["status"] == "PASS",
        "same_annotation_sha": (
            c_report["checks"]["annotation_sha_matches"]
            and d_report["checks"]["annotation_sha_matches"]
        ),
        "same_commit": (
            c_report["commit"] == d_report["commit"] == args.expected_commit
        ),
        "final_test_locked": True,
    }

    report = {
        "protocol": PROTOCOL,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "scope": (
            "3-epoch train/validation audit only; checkpoint selection must use "
            "validation rel_mean_recall/50; final test locked"
        ),
        "final_test_allowed": False,
        "expected_commit": args.expected_commit,
        "expected_epochs": args.expected_epochs,
        "expected_workers": args.expected_workers,
        "annotation": str(annotation),
        "annotation_sha256": annotation_sha,
        "trainer_source": str(trainer_source),
        "trainer_source_sha256": sha256(trainer_source),
        "runs": {
            "C": c_report,
            "D1V2A": d_report,
        },
        "checks": checks,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))

    if report["status"] != "PASS":
        raise SystemExit(1)

    print()
    print("D1-v2a CHECKPOINT-SELECTION PROTOCOL AUDIT V1: PASS")


if __name__ == "__main__":
    main()
