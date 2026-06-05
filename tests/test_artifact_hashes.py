from __future__ import annotations

from pathlib import Path

from titan_v4.metrics.external_validation import verify_hash_manifest


ROOT = Path(__file__).resolve().parents[1]


def test_artifact_hash_manifest_passes():
    failures = verify_hash_manifest(
        ROOT,
        ROOT / "outputs/gold_master_external_validation/artifact_hashes/ARTIFACT_HASHES.csv",
    )
    assert failures == []

