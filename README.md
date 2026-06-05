# NeuroGuardian TITAN V4 ECG External Validation

This repository is a reproducible research artifact for TITAN V4 ECG validation. It contains the final external validation label set, model checkpoint, metric reports, safety annex outputs, ESP32 edge deployment code, and automated tests needed to audit the reported results.

## Results

| Module | External scope | Accuracy | Macro-F1 | Evidence role |
|---|---:|---:|---:|---|
| Arrhythmia Primary-9 | 672 records | 0.9003 | 0.8782 | Main external rhythm result |
| Pathology Primary-5 | Primary labels | 0.9026 | 0.6687 | Main external pathology result |
| Cascade/OOD | 211 diagnostic records | coverage 0.5071 | selective-risk metrics | Safety annex |
| ESP32 edge | Engineering validation | CSV/MAT/HEA recordings | signal quality telemetry | Hardware evidence |

Primary-9 rhythm result:

```text
correct_predictions = 605
total_records = 672
accuracy = 605 / 672 = 0.9002976190
macro-F1 = 0.8781916793
weighted-F1 = 0.9008533778
```

Pathology Primary-5 result:

```text
classes = IMI, ALMI, ILMI, LAE, ISC_
accuracy = 0.90256
macro-F1 = 0.66874
```

Cascade/OOD is reported only as a safety annex:

```text
diagnostic_subset_records = 211
accepted_predictions = 107
coverage = 0.5071090047
quarantine_rate = 0.4928909953
```

## Repository Map

```text
configs/        Validation configuration files.
src/            Python package with model, training, and metric utilities.
scripts/        Reproducibility entrypoints.
tests/          Audit and validation tests.
data/           Final external validation label set and input summaries.
models/         Gold master PTH checkpoint and hash table.
outputs/        Machine-readable external validation outputs.
reports/        Methods, result tables, and traceability notes.
hardware/       ESP32 firmware, recorder, edge models, and ECG recordings.
cedia/          Minimal CEDIA synchronization and job utilities.
archive/        Historical-file manifest only.
```

## Reproducibility

Create an environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

Verify artifact hashes:

```powershell
python scripts\verify_artifact_hashes.py
```

Run the external validation summaries:

```powershell
python scripts\run_primary9_external_validation.py
python scripts\run_pathology_primary5_external_validation.py
python scripts\run_combined_external_validation.py
python scripts\build_result_tables.py
```

Run tests:

```powershell
python -m pytest tests -q
```

## Claim Boundary

The arrhythmia result is full-support external validation over the reportable Primary-9 rhythm set. The pathology result is reported for the configured Primary-5 pathology set. Cascade/OOD is a safety annex and is not used as the primary performance source. External validation records are not used for model training or threshold tuning.

The public manuscript term for labels is `final external validation label set`.

## Key Artifacts

| Artifact | Path |
|---|---|
| Gold Primary-9 checkpoint | `models/gold_master/gold_master_primary9_model.pth` |
| Primary-9 external report | `outputs/gold_master_external_validation/arrhythmia_primary9/primary9_external_validation_report.json` |
| Pathology Primary-5 summary | `outputs/gold_master_external_validation/pathology_primary5/pathology_primary5_external_summary.json` |
| Combined summary | `outputs/gold_master_external_validation/combined/combined_external_validation_summary.json` |
| Final validation labels | `data/external_validation/final_external_validation_labels.csv` |
| Artifact hashes | `outputs/gold_master_external_validation/artifact_hashes/ARTIFACT_HASHES.csv` |

