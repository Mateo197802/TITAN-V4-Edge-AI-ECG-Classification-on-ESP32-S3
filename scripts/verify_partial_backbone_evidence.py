#!/usr/bin/env python3
"""Verify the published CEDIA refinement artifact and recompute its metrics."""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "outputs/reviewer_verification/partial_backbone_refinement_2026-10-12"
PHASE3_CODE = BUNDLE / "training/phase3"
sys.path.insert(0, str(PHASE3_CODE))
import contract as contract  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_bundle_hashes() -> int:
    manifest_path = BUNDLE / "SHA256SUMS.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    actual = {path.relative_to(BUNDLE).as_posix() for path in BUNDLE.rglob("*")
              if path.is_file() and path != manifest_path
              and "__pycache__" not in path.parts and path.suffix != ".pyc"}
    if set(manifest) != actual:
        raise AssertionError("Bundle checksum inventory is incomplete or includes absent files")
    for relative, expected in manifest.items():
        path = BUNDLE / relative
        if not path.resolve().is_relative_to(BUNDLE.resolve()):
            raise AssertionError("Checksum manifest path escaped the bundle")
        if not path.is_file() or sha256(path) != expected:
            raise AssertionError(f"Artifact hash mismatch: {relative}")
    return len(manifest)


def changed_parameters(left: torch.nn.Module, right: torch.nn.Module) -> list[str]:
    left_state = left.state_dict()
    right_state = right.state_dict()
    if set(left_state) != set(right_state):
        raise AssertionError("Checkpoint state keys differ")
    return [name for name in left_state if not torch.equal(left_state[name], right_state[name])]


def verify_model_contract() -> dict[str, object]:
    gold = contract.load_model(ROOT / "models/gold_master/gold_master_primary9_model.pth")
    warm_start = contract.load_model(BUNDLE / "training/warm_start_phase2.pth")
    candidate = contract.load_model(BUNDLE / "model/partial_backbone_candidate.pth")
    for model in (gold, warm_start, candidate):
        if contract.parameter_count(model) != contract.PARAMETERS:
            raise AssertionError("Original parameter count changed")
    phase2_changes = changed_parameters(gold, warm_start)
    if any(not name.startswith("head_rhythm.") for name in phase2_changes):
        raise AssertionError("Warm-start differs from Gold outside the original rhythm head")
    candidate_changes = changed_parameters(warm_start, candidate)
    if any(not name.startswith(("res_layers.4.", "head_rhythm.")) for name in candidate_changes):
        raise AssertionError("Candidate differs outside the two authorized original modules")
    parameters = dict(candidate.named_parameters())
    changed_count = sum(
        parameters[name].numel()
        for name in candidate_changes
        if name in parameters
    )
    if changed_count != 460_065:
        raise AssertionError(f"Unexpected changed parameter count: {changed_count}")
    with torch.inference_mode():
        outputs = candidate(torch.zeros(2, 6, 1250))
    if len(outputs) != 4 or outputs[0].shape != (2, 9):
        raise AssertionError("Original four-head inference contract changed")
    return {
        "total_parameters": contract.parameter_count(candidate),
        "gold_to_warm_start_changed_tensors": len(phase2_changes),
        "warm_start_to_candidate_changed_tensors": len(candidate_changes),
        "candidate_changed_parameters": changed_count,
        "four_output_heads": len(outputs),
        "rhythm_output_shape": list(outputs[0].shape),
    }


def verify_metrics() -> dict[str, object]:
    metrics_path = BUNDLE / "evaluation/followup_metrics.json"
    published = json.loads(metrics_path.read_text(encoding="utf-8"))
    rows_by_cohort: dict[str, list[dict[str, str]]] = {}
    with (BUNDLE / "evaluation/followup_predictions.csv").open(
        newline="", encoding="utf-8-sig"
    ) as handle:
        for row in csv.DictReader(handle):
            rows_by_cohort.setdefault(row["cohort"], []).append(row)

    correct = phase2_correct = changed = new_correct = regressions = 0
    total = 0
    for cohort, rows in rows_by_cohort.items():
        y = np.asarray([contract.CLASSES.index(row["reference"]) for row in rows])
        probabilities = np.asarray([
            [float(row[f"prob_{label}"]) for label in contract.CLASSES]
            for row in rows
        ])
        recomputed = contract.metrics(y, probabilities)
        saved = published["cohorts"][cohort]["phase3"]
        for key in ("n", "correct", "confusion_matrix", "support"):
            if recomputed[key] != saved[key]:
                raise AssertionError(f"{cohort}: published {key} does not recalculate")
        for key in ("accuracy", "macro_f1", "weighted_f1"):
            if abs(recomputed[key] - saved[key]) > 1e-12:
                raise AssertionError(f"{cohort}: published {key} does not recalculate")

        predicted = np.argmax(probabilities, axis=1)
        for index, row in enumerate(rows):
            candidate_label = contract.CLASSES[predicted[index]]
            if row["phase3_prediction"] != candidate_label:
                raise AssertionError(f"{cohort}: row prediction disagrees with probabilities")
            candidate_ok = candidate_label == row["reference"]
            phase2_ok = row["phase2_prediction"] == row["reference"]
            correct += candidate_ok
            phase2_correct += phase2_ok
            changed += row["phase3_prediction"] != row["phase2_prediction"]
            new_correct += candidate_ok and not phase2_ok
            regressions += phase2_ok and not candidate_ok
        total += len(rows)

    if total != published["records"]:
        raise AssertionError("Follow-up record count differs from the published summary")
    return {
        "records": total,
        "candidate_correct": correct,
        "candidate_accuracy": correct / total,
        "phase2_correct": phase2_correct,
        "phase2_accuracy": phase2_correct / total,
        "changed_predictions": changed,
        "newly_correct": new_correct,
        "regressions": regressions,
        "net_correct_change": correct - phase2_correct,
    }


def main() -> None:
    count = verify_bundle_hashes()
    model = verify_model_contract()
    metrics = verify_metrics()
    print(json.dumps({"verified_artifact_files": count, "model": model, "followup": metrics}, indent=2))


if __name__ == "__main__":
    main()
