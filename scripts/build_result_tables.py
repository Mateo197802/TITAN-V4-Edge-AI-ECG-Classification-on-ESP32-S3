from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from titan_v4.metrics.external_validation import cascade_summary, load_json, pathology_summary, primary9_summary


def main() -> int:
    primary9 = primary9_summary(
        load_json("outputs/reproduced/primary9/primary9_recomputed_report.json")
    )
    legacy_primary9 = load_json(
        "outputs/gold_master_external_validation/arrhythmia_primary9/primary9_external_validation_report.json"
    )
    pathology = pathology_summary(
        load_json("outputs/gold_master_external_validation/pathology_primary5/pathology_primary5_external_summary.json")
    )
    cascade = cascade_summary(
        load_json("outputs/gold_master_external_validation/cascade_annex/cascade_safety_annex_report.json")
    )
    cedia = load_json("reports/evidence/cedia-validation-summary.json")
    checkpoint_training = load_json("models/gold_master/gold_master_primary9_training_summary.json")
    table = "\n".join(
        [
            "| Evidence | Unit | Accuracy (reported) | Macro-F1 | Weighted-F1 | Interpretation |",
            "|---|---:|---:|---:|---:|---|",
            f"| CEDIA internal validation | {cedia['validation_windows']:,} windows | {cedia['accuracy']:.5f} | {cedia['macro_f1']:.5f} | {cedia['weighted_f1']:.5f} | Different checkpoint; internal window-level aggregate |",
            f"| Distributed-checkpoint selection summary | {checkpoint_training['val_windows']:,} windows | n/a | {checkpoint_training['best_val_f1_macro']:.5f} | n/a | Stored best-epoch metric; validation predictions unavailable |",
            f"| Recomputed Primary-9 diagnostic | {primary9['records']} records | {primary9['accuracy']:.5f} | {primary9['macro_f1']:.5f} | {primary9['weighted_f1']:.5f} | Reproducible project single-label diagnostic; not official Challenge score |",
            f"| Historical Primary-9 aggregate | 672 rows | {legacy_primary9['accuracy']:.5f} | {legacy_primary9['macro_f1']:.5f} | {legacy_primary9['weighted_f1']:.5f} | Unverified; 129 label differences and no prediction-level artifact |",
            f"| Pathology Primary-5 aggregate | 5 binary labels | {pathology['accuracy']:.5f} | {pathology['macro_f1']:.5f} | n/a | Legacy per-label accuracy; predictions/checkpoint/threshold provenance unavailable |",
            f"| Cascade/OOD | {cascade['diagnostic_subset_records']} records | n/a | n/a | n/a | Historical safety annex only; coverage={cascade['coverage']:.4f} |",
        ]
    )
    out = Path("reports/tables/external_validation_metrics.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "# Metric Evidence Reconciliation\n\n"
        "These figures use different checkpoints, units and protocols; they are not directly comparable. "
        "The 672-record single-label diagnostic is not the official PhysioNet Challenge score or verified independent external performance. "
        "See [the evidence reconciliation](../evidence/metric-reconciliation.md).\n\n"
        + table
        + "\n",
        encoding="utf-8",
    )
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
