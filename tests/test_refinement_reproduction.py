from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from titan_v4.reproducibility import refinement

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location("refinement_test_" + name, ROOT / "scripts" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def row():
    return {"ecg_id": "1", "dataset": "ptb-xl", "record_path": "records500/00000/00001_hr",
            "source_release": "ptb-xl", "version": "1.0.3", "doi": "10.13026/kfzx-aw45",
            "signal_filename": "00001_hr.dat", "header_sha256": hashlib.sha256(b"header").hexdigest(),
            "signal_sha256": hashlib.sha256(b"signal").hexdigest()}


def test_download_verifies_both_source_files_before_caching(tmp_path, monkeypatch, row):
    monkeypatch.setattr(refinement, "_http_bytes", lambda url: b"header" if url.endswith(".hea") else b"signal")
    base = refinement.ensure_record(row, tmp_path)
    assert base.with_suffix(".hea").read_bytes() == b"header"
    assert base.with_suffix(".dat").read_bytes() == b"signal"
    monkeypatch.setattr(refinement, "_http_bytes", lambda url: pytest.fail("Offline cache must avoid network"))
    assert refinement.ensure_record(row, tmp_path, download=False) == base


def test_wrong_download_hash_is_rejected_and_not_cached(tmp_path, monkeypatch, row):
    monkeypatch.setattr(refinement, "_http_bytes", lambda url: b"wrong content")
    with pytest.raises(ValueError, match="Downloaded source hash mismatch"):
        refinement.ensure_record(row, tmp_path)
    assert not list(tmp_path.rglob("*.hea"))


@pytest.mark.parametrize("path", ["../../secret", "/records500/00000/00001_hr", "records500\\00000\\00001_hr"])
def test_record_path_cannot_escape_public_release(tmp_path, row, path):
    row["record_path"] = path
    with pytest.raises(ValueError, match="Unsafe record path"):
        refinement.ensure_record(row, tmp_path)


def test_wrong_release_version_fails_before_network(tmp_path, monkeypatch, row):
    row["version"] = "1.0.2"
    monkeypatch.setattr(refinement, "_http_bytes", lambda url: pytest.fail("Wrong version must not download"))
    with pytest.raises(ValueError, match="not pinned"):
        refinement.ensure_record(row, tmp_path)


def test_signal_filename_cannot_point_to_other_records(tmp_path, row):
    row["signal_filename"] = "../other.dat"
    with pytest.raises(ValueError, match="signal filename"):
        refinement.ensure_record(row, tmp_path)


def sidecar():
    return {"x": np.zeros((2, 192), dtype=np.float32), "y": np.array([0, 1]),
            "dataset": np.array(["ptb-xl", "ptb-xl"]), "ecg_id": np.array(["1", "2"]),
            "gold": np.ones((2, 9), dtype=np.float32) / 9,
            "teacher": np.ones((2, 9), dtype=np.float32) / 9}


def train_rows():
    return [{"ecg_id": str(index + 1), "dataset": "ptb-xl", "reference_label": label,
             "tensor_sha256": "a" * 64} for index, label in enumerate(["AFIB", "SB"])]


def test_rebuild_preserves_training_targets_and_teacher_values():
    module = load_script("prepare_refinement_training_inputs")
    data = sidecar()
    output = module.build_split(train_rows(), data, np.zeros((2, 6, 1250), dtype=np.float32))
    assert np.array_equal(output["y"], data["y"])
    assert np.array_equal(output["teacher"], data["teacher"])
    assert np.array_equal(output["embedding"], data["x"])


def test_reordered_sidecar_cannot_silently_change_reference_labels():
    module = load_script("prepare_refinement_training_inputs")
    data = sidecar()
    data["ecg_id"] = data["ecg_id"][::-1]
    with pytest.raises(ValueError, match="record ordering or labels differ"):
        module.build_split(train_rows(), data, np.zeros((2, 6, 1250), dtype=np.float32))


def test_out_of_range_training_target_is_rejected():
    module = load_script("prepare_refinement_training_inputs")
    data = sidecar()
    data["y"][0] = -1
    with pytest.raises(ValueError, match="outside Primary-9"):
        module.build_split(train_rows(), data, np.zeros((2, 6, 1250), dtype=np.float32))


def test_fractional_training_target_is_rejected():
    module = load_script("prepare_refinement_training_inputs")
    data = sidecar()
    data["y"] = np.array([0.2, 1.0])
    with pytest.raises(ValueError, match="outside Primary-9"):
        module.build_split(train_rows(), data, np.zeros((2, 6, 1250), dtype=np.float32))


def prediction_rows():
    from titan_v4.data.primary9_registry import PRIMARY9_CLASSES
    return [dict(cohort="test", dataset="ptb-xl", ecg_id=str(index), reference="AFIB",
                 **{f"prob_{label}": 1.0 if label == "AFIB" else 0.0 for label in PRIMARY9_CLASSES})
            for index in (1, 2)]


def test_inference_comparison_joins_by_record_not_csv_order():
    module = load_script("run_refinement_inference")
    expected = prediction_rows()
    rows = [dict(cohort="test", dataset="ptb-xl", ecg_id=str(index), reference_label="AFIB") for index in (2,1)]
    probabilities = np.zeros((2,9)); probabilities[:,0] = 1
    assert module.compare_predictions(rows, probabilities, expected, 2e-5)["passed"]


def test_inference_comparison_rejects_relabeling_and_duplicate_keys():
    module = load_script("run_refinement_inference")
    expected = prediction_rows()
    rows = [dict(cohort="test", dataset="ptb-xl", ecg_id="1", reference_label="SB")]
    with pytest.raises(ValueError, match="Reference label changed"):
        module.compare_predictions(rows, np.ones((1,9))/9, expected, 2e-5)
    rows[0]["reference_label"] = "AFIB"
    with pytest.raises(ValueError, match="duplicate record keys"):
        module.compare_predictions(rows, np.ones((1,9))/9, [expected[0],expected[0]], 2e-5)


def test_recorded_validation_probabilities_recalculate_interpolation_metrics():
    import json
    root = ROOT / "outputs/reviewer_verification/partial_backbone_refinement_2026-10-12/training/weight_interpolation/results"
    selection = json.loads((root / "selection.json").read_text())
    module = load_script("verify_partial_backbone_evidence")
    for candidate in selection["candidates"]:
        with np.load(root / (candidate["name"] + "_validation.npz"), allow_pickle=False) as data:
            metrics = module.contract.metrics(data["y"], data["probabilities"])
            for key in ("n", "correct", "confusion_matrix", "support"):
                assert metrics[key] == candidate["metrics"][key]
            assert metrics["macro_f1"] == pytest.approx(candidate["metrics"]["macro_f1"], abs=1e-12)
    assert selection["selected"] == "base"
    assert selection["gradient_updates"] is False
