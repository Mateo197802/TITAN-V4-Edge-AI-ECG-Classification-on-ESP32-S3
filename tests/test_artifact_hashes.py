from __future__ import annotations

from pathlib import Path

import hashlib

from titan_v4.metrics.external_validation import manifest_file_fingerprint, verify_hash_manifest


ROOT = Path(__file__).resolve().parents[1]


def test_artifact_hash_manifest_passes():
    failures = verify_hash_manifest(
        ROOT,
        ROOT / "outputs/gold_master_external_validation/artifact_hashes/ARTIFACT_HASHES.csv",
    )
    assert failures == []


def test_text_artifact_hash_normalizes_windows_line_endings(tmp_path):
    artifact = tmp_path / "notes.md"
    artifact.write_bytes(b"first line\r\nsecond line\r\n")

    digest, size_bytes = manifest_file_fingerprint(artifact)

    assert digest == hashlib.sha256(b"first line\nsecond line\n").hexdigest().upper()
    assert size_bytes == len(b"first line\nsecond line\n")


def test_binary_artifact_hash_preserves_exact_bytes(tmp_path):
    artifact = tmp_path / "weights.pth"
    content = b"binary\r\ncontent"
    artifact.write_bytes(content)

    digest, size_bytes = manifest_file_fingerprint(artifact)

    assert digest == hashlib.sha256(content).hexdigest().upper()
    assert size_bytes == len(content)

