# Reproducibility and Evidence Status

## What the Checkout Can Reproduce

- SHA-256 and canonical text-size checks for files listed in `ARTIFACT_HASHES.csv`.
- Unit and contract tests over packaged code and static outputs.
- Printed aggregate JSON summaries and the Markdown results table from the checked-in aggregate JSON reports. The combined-summary script can optionally write its JSON output with `--write`.
- ESP32 firmware compilation after installing the pinned PlatformIO CLI and Espressif32 platform; no firmware build was completed in this audit environment.

Python top-level requirements are pinned in `requirements.txt` and the build backend is pinned in `pyproject.toml`. The tested environment for this audit was CPython 3.12.10 on Windows with the versions recorded in `requirements.txt`. Transitive wheels are not hash-locked, so installation is version-pinned but not a bit-for-bit hermetic environment. Hardware and CEDIA dependencies are separate optional environments.

## What It Cannot Reproduce From This Public Snapshot

- The external inference pass: the report scripts only read precomputed JSON; they do not load `gold_master_primary9_model.pth` or ECG waveforms.
- The exact per-record predictions and complete confusion matrices for all reported rows: not all prediction-level outputs are included.
- Training from source data: training signals, frozen train/validation manifests, and teacher sidecar arrays referenced in the training summary are absent.
- Independence of the external evaluation: a uniqueness test exists, but no complete train-versus-test ID/path/hash overlap evidence is packaged.
- Numerical equivalence between offline resampling (`scipy.signal.resample_poly`) and the ESP32 linear downsampling implementation. The edge protocol explicitly leaves this measurement pending.
- CEDIA connectivity, authentication, host-key enrollment, and SLURM execution; no remote account was accessed.

The model checkpoint and results are useful research artifacts, but checking their file hashes does not reproduce the scientific computation or establish the correctness of the reported metrics. The Primary-9, Pathology Primary-5, and Cascade/OOD claim boundaries are kept separate in the report files; Cascade/OOD remains a safety annex.

## Commands

```text
python -m pytest -q
python scripts/verify_artifact_hashes.py
python scripts/run_primary9_external_validation.py
python scripts/run_pathology_primary5_external_validation.py
python scripts/run_combined_external_validation.py
python scripts/build_result_tables.py
```

`scripts/build_artifact_hashes.py --write` intentionally replaces the hash inventory with hashes of current tracked and non-ignored files. Run it only when the artifacts are intentionally changing and after reviewing the resulting diff. The inventory excludes its own CSV to avoid self-reference.

## Audit Snapshot

The public repository was cloned at commit `3a2ef71b35e0d9d05dca511120f976fc27f7fab9` for the audit. Initial execution found a stale recorder-test path, a missing default model-converter candidate, conflicting pytest module names, Windows line-ending hash failures, two stale hashes, and unpinned/floating environment configuration. The current local branch contains focused fixes; the final test/hash results are recorded in the audit response after validation.
