# NeuroGuardian TITAN V4 ECG Evidence Package

This repository packages aggregate reports, a Primary-9 checkpoint, validation labels and summaries, ESP32-S3 firmware, and engineering recordings. It is an evidence snapshot, not a complete end-to-end reproduction package: the public scripts summarize checked-in JSON reports and do not run inference or retrain the checkpoint.

## Reported Results

| Evaluation | Scope | Accuracy | Macro-F1 | Role |
|---|---:|---:|---:|---|
| Arrhythmia Primary-9 | 672 records; 605 correct | 0.90030 | 0.87819 | Main rhythm result |
| Pathology Primary-5 | Configured five-label set | 0.90256 | 0.66874 | Separate pathology result |
| Cascade/OOD | 211 diagnostic records; 107 accepted | Coverage 0.50711 | Selective-risk metrics | Safety annex only |
| ESP32 edge | Engineering recordings | Not an accuracy estimate | Not an accuracy estimate | Acquisition and deployment evidence |

The Primary-9 aggregate is reported over all 672 records in the configured reportable set. The Pathology Primary-5 macro-F1 is 0.66874; it meets the documented 0.65 promoted threshold, not the earlier strict 0.70 target. Cascade/OOD coverage is 107/211 and must not be presented as primary full-support performance. These are the values in the bundled reports; the repository does not independently recompute them from per-record predictions.

The project is research software, not a cleared medical device. Its classification output must not be used to diagnose, treat, or rule out a medical condition.

## Contents

| Path | Contents |
|---|---|
| `src/` | Data, model, training, and metric utilities. |
| `scripts/` | Aggregate-report, table, and hash-manifest tools. |
| `tests/` | Python and ESP32 contract tests. |
| `data/external_validation/` | Final external validation labels and aggregate inputs; no ECG waveforms. |
| `models/gold_master/` | Primary-9 checkpoint and training summary. |
| `outputs/` | Machine-readable aggregate reports and hash manifest. |
| `reports/` | Methods, results, and evidence traceability. |
| `hardware/esp32/` | Firmware, recorder, model files, and engineering recordings. |
| `cedia/` | Optional cluster sync and job tools; setup is documented separately. |

## Reproduce Packaged Checks

Use Python 3.12. Create and activate an environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

On macOS/Linux, activate with `source .venv/bin/activate`. Then install the pinned top-level requirements and package:

```text
python -m pip install -r requirements.txt
python -m pip install -e .
python -m pytest -q
python scripts/verify_artifact_hashes.py
```

The report scripts validate and print the bundled aggregate inputs. `build_result_tables.py` regenerates the Markdown table from those same inputs:

```text
python scripts/run_primary9_external_validation.py
python scripts/run_pathology_primary5_external_validation.py
python scripts/run_combined_external_validation.py
python scripts/build_result_tables.py
```

To inspect or intentionally refresh the whole-repository hash inventory, use `python scripts/build_artifact_hashes.py` to check it, or pass `--write` only after reviewing the intended changes. Text hashes use LF-normalized UTF-8; binary hashes use the exact file bytes. The manifest does not hash itself.

## Reproduction Limits

The repository does not contain source ECG waveforms, prediction-level outputs for all 672 records, the exact training/validation manifests, or all training and teacher-sidecar data. The aggregate scripts do not load the `.pth` checkpoint. Therefore this checkout can verify packaged files, test code contracts, and rebuild summaries/tables, but cannot reproduce inference, training, the claimed record-level predictions, or a zero-overlap audit. See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) and [DATA_PROVENANCE.md](DATA_PROVENANCE.md) for the evidence gaps and source-by-source status.

## Edge Firmware

Install the pinned PlatformIO CLI with `python -m pip install -r requirements-firmware.txt`, then build with `python -m platformio run -d hardware/esp32/firmware -e esp32s3`. The general and campus profiles block sensitive HTTP routes. Only explicitly named private-hotspot profiles enable the unauthenticated raw-ECG and result endpoints; use them on an isolated private network only. Firmware compilation and physical-device behavior were not verified in this audit environment. See [hardware/esp32/README.md](hardware/esp32/README.md).

## Data, Citation, and License

The MIT license applies only to original software source code, not to datasets, row-level labels, model weights, recordings, results, or evidence documents. See [LICENSE_SCOPE.md](LICENSE_SCOPE.md) for the boundary and [DATA_PROVENANCE.md](DATA_PROVENANCE.md) for the unresolved rights of the 312 `data_test` rows. Dataset and method citations are in [REFERENCES.md](REFERENCES.md); the repository currently has no associated manuscript citation.

See [CITATION.cff](CITATION.cff) for software citation metadata and [LICENSE](LICENSE) for the standard MIT terms. Redistribution of excluded data/model artifacts must follow their own source terms and is not authorized by the code license.
