from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from titan_v4.reproducibility.inference import load_primary9_model, predict_primary9, primary9_metrics
from titan_v4.reproducibility.physionet import (
    dataset_for_record,
    ensure_local_wfdb_header,
    ensure_local_wfdb_record,
    preprocess_primary9_signal,
    resolve_indexed_records,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("source", "record_id", "expected"),
    [
        ("data_test", "a1707", "cpsc_2018"),
        ("data_test", "e02097", "georgia"),
        ("data_test", "hr00001", "ptb-xl"),
        ("training/chapman_shaoxing", "js00012", "chapman_shaoxing"),
        ("training/cpsc_2018_extra", "q0001", "cpsc_2018_extra"),
        ("training/ningbo", "js10652", "ningbo"),
    ],
)
def test_dataset_for_record_maps_every_published_source(source, record_id, expected):
    assert dataset_for_record(source, record_id) == expected


def test_unknown_data_test_record_fails_instead_of_guessing_a_dataset():
    with pytest.raises(ValueError, match="Cannot resolve data_test source"):
        dataset_for_record("data_test", "unknown-001")


def test_index_resolution_uses_authoritative_record_lists_not_numeric_buckets():
    indexes = {
        "g1": ["JS10647", "JS10652", "JS10653"],
        "g2": ["JS11647", "JS11652"],
    }

    resolved = resolve_indexed_records("ningbo", ["js10652", "JS11652"], indexes)

    assert resolved == {
        "JS10652": "training/ningbo/g1/JS10652",
        "JS11652": "training/ningbo/g2/JS11652",
    }


def test_index_resolution_rejects_missing_or_ambiguous_record_ids():
    with pytest.raises(ValueError, match="not present in the official record indexes"):
        resolve_indexed_records("georgia", ["E99999"], {"g1": ["E00001"]})

    with pytest.raises(ValueError, match="appears in multiple official record indexes"):
        resolve_indexed_records("georgia", ["E00001"], {"g1": ["E00001"], "g2": ["E00001"]})


def test_preprocessing_returns_normalized_model_shape_for_named_frontal_leads():
    sample_rate = 500
    t = np.arange(sample_rate * 10, dtype=np.float32) / sample_rate
    signals = np.column_stack([np.sin(2 * np.pi * (1 + i) * t) for i in range(6)])
    names = ["I", "II", "III", "aVR", "aVL", "aVF"]

    result = preprocess_primary9_signal(signals, names, sample_rate)

    assert result.shape == (6, 1250)
    assert result.dtype == np.float32
    assert np.isfinite(result).all()
    np.testing.assert_allclose(result.mean(axis=1), np.zeros(6), atol=1e-5)
    np.testing.assert_allclose(result.std(axis=1), np.ones(6), atol=1e-5)


def test_preprocessing_rejects_missing_required_leads_instead_of_reusing_other_channels():
    signal = np.ones((5000, 2), dtype=np.float32)

    with pytest.raises(ValueError, match="Missing required frontal leads"):
        preprocess_primary9_signal(signal, ["I", "II"], 500)


def test_offline_record_validation_hashes_files_and_never_fills_missing_signals(tmp_path):
    base = tmp_path / "training/ptb-xl/g1/HR00001"
    base.parent.mkdir(parents=True)
    header = (
        "HR00001 2 500 5000\n"
        "HR00001.mat 16x1+24 1000(0)/mV 16 0 0 0 0 I\n"
        "HR00001.mat 16x1+24 1000(0)/mV 16 0 0 0 0 II\n"
    ).encode("ascii")
    wave = b"wfdb-waveform-fixture"
    base.with_suffix(".hea").write_bytes(header)
    base.with_suffix(".mat").write_bytes(wave)

    resolved, hashes = ensure_local_wfdb_record(
        "training/ptb-xl/g1/HR00001", tmp_path, download=False
    )

    assert resolved == base
    assert hashes["training/ptb-xl/g1/HR00001.hea"] == hashlib.sha256(header).hexdigest().upper()
    assert hashes["training/ptb-xl/g1/HR00001.mat"] == hashlib.sha256(wave).hexdigest().upper()


def test_offline_header_validation_checks_id_without_requiring_waveform(tmp_path):
    base = tmp_path / "training/ptb-xl/g1/HR00001"
    base.parent.mkdir(parents=True)
    header = b"HR00001 1 500 5000\nHR00001.mat 16x1+24 1000(0)/mV 16 0 0 0 0 I\n"
    base.with_suffix(".hea").write_bytes(header)

    resolved, digest = ensure_local_wfdb_header(
        "training/ptb-xl/g1/HR00001", tmp_path, download=False
    )

    assert resolved == base
    assert digest == hashlib.sha256(header).hexdigest().upper()


def test_offline_header_validation_rejects_wrong_record_id(tmp_path):
    base = tmp_path / "training/ptb-xl/g1/HR00001"
    base.parent.mkdir(parents=True)
    base.with_suffix(".hea").write_text("HR99999 1 500 5000\n", encoding="ascii")

    with pytest.raises(ValueError, match="does not match expected"):
        ensure_local_wfdb_header("training/ptb-xl/g1/HR00001", tmp_path, download=False)


def test_offline_record_validation_rejects_a_missing_waveform(tmp_path):
    base = tmp_path / "training/ptb-xl/g1/HR00001"
    base.parent.mkdir(parents=True)
    base.with_suffix(".hea").write_text(
        "HR00001 1 500 5000\nHR00001.mat 16x1+24 1000(0)/mV 16 0 0 0 0 I\n",
        encoding="ascii",
    )

    with pytest.raises(FileNotFoundError, match="Missing WFDB signal file"):
        ensure_local_wfdb_record("training/ptb-xl/g1/HR00001", tmp_path, download=False)


def test_published_checkpoint_loads_strictly_and_runs_real_inference():
    model = load_primary9_model(ROOT / "models/gold_master/gold_master_primary9_model.pth")
    probabilities = predict_primary9(model, np.zeros((1, 6, 1250), dtype=np.float32))

    assert probabilities.shape == (1, 9)
    np.testing.assert_allclose(probabilities.sum(axis=1), np.ones(1), atol=1e-6)


def test_primary9_metrics_are_computed_from_record_level_labels():
    summary = primary9_metrics(["AFIB", "NSR", "NSR"], ["AFIB", "AFIB", "NSR"])

    assert summary["total_records"] == 3
    assert summary["correct_predictions"] == 2
    assert summary["accuracy"] == pytest.approx(2 / 3)
    assert sum(map(sum, summary["confusion_matrix"])) == 3
