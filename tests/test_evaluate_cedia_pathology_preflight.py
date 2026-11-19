from __future__ import annotations

from scripts.evaluate_cedia_pathology_validation import (
    PATHOLOGY_CLASS_NAMES,
    _preflight_windows,
    _write_input_manifest,
    _write_scores_csv,
    _write_artifact_sha256_manifest,
)


class _Window:
    def to(self, _device: str):
        return self


class _Evaluator:
    def read_first_window(self, record_base: str):
        if record_base == "bad":
            raise TypeError("signal length unavailable")
        return _Window()


def test_preflight_keeps_valid_windows_and_records_unreadable_inputs(capsys):
    windows, failures = _preflight_windows(
        ("00001", "00002"),
        record_bases={"00001": "ok", "00002": "bad"},
        evaluator=_Evaluator(),
    )

    assert set(windows) == {"00001"}
    assert failures == {"00002": {"error_type": "TypeError"}}
    assert "Excluded unreadable PTB-XL record 00002: TypeError" in capsys.readouterr().out


def test_artifact_hash_manifest_covers_outputs_and_excludes_itself(tmp_path):
    artifact = tmp_path / "report.json"
    artifact.write_text('{"result": true}\n', encoding="utf-8")

    manifest = _write_artifact_sha256_manifest(tmp_path)

    assert manifest.name == "SHA256SUMS.csv"
    assert "report.json" in manifest.read_text(encoding="utf-8")
    assert "SHA256SUMS.csv" not in manifest.read_text(encoding="utf-8")
    assert b"\r\n" not in manifest.read_bytes()


def test_cedia_output_csvs_use_portable_lf_line_endings(tmp_path):
    score_row = {"record_id": "00001", "split": "test"}
    for name in PATHOLOGY_CLASS_NAMES:
        score_row[f"true_{name}"] = 0
        score_row[f"score_{name}"] = 0.1
    scores_path = tmp_path / "scores.csv"
    _write_scores_csv(scores_path, [score_row])

    record_base = tmp_path / "00001_lr"
    record_base.with_suffix(".hea").write_bytes(b"header")
    record_base.with_suffix(".dat").write_bytes(b"signal")
    input_manifest_path = tmp_path / "inputs.csv"
    _write_input_manifest(
        input_manifest_path,
        ("00001",),
        {"00001": "test"},
        {"00001": record_base},
        {},
    )

    assert b"\r\n" not in scores_path.read_bytes()
    assert b"\r\n" not in input_manifest_path.read_bytes()
