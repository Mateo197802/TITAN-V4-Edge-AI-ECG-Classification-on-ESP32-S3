# Artifact Traceability

## Gold Model

| Artifact | Path |
|---|---|
| Gold Primary-9 PTH | `models/gold_master/gold_master_primary9_model.pth` |
| Protected baseline PTH | `models/gold_master/protected_baseline_primary9_model.pth` |
| Training summary | `models/gold_master/gold_master_primary9_training_summary.json` |

## Verification

The current Primary-9 inference evidence is under `outputs/reproduced/primary9/`. The old aggregate reports under `outputs/gold_master_external_validation/` are explicitly marked legacy.

Run:

```powershell
python scripts\verify_artifact_hashes.py
```

The expected hashes are stored in:

```text
outputs/gold_master_external_validation/artifact_hashes/ARTIFACT_HASHES.csv
```

