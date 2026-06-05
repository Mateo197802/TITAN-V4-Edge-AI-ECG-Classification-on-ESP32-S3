from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from titan_v4.metrics.external_validation import cascade_summary, load_json, pathology_summary, primary9_summary


def main() -> int:
    primary9 = primary9_summary(
        load_json("outputs/gold_master_external_validation/arrhythmia_primary9/primary9_external_validation_report.json")
    )
    pathology = pathology_summary(
        load_json("outputs/gold_master_external_validation/pathology_primary5/pathology_primary5_external_summary.json")
    )
    cascade = cascade_summary(
        load_json("outputs/gold_master_external_validation/cascade_annex/cascade_safety_annex_report.json")
    )
    table = "\n".join(
        [
            "| Module | External scope | Accuracy | Macro-F1 | Evidence role |",
            "|---|---:|---:|---:|---|",
            f"| Arrhythmia Primary-9 | {primary9['records']} records | {primary9['accuracy']:.4f} | {primary9['macro_f1']:.4f} | Main external rhythm result |",
            f"| Pathology Primary-5 | Primary labels | {pathology['accuracy']:.4f} | {pathology['macro_f1']:.4f} | Main external pathology result |",
            f"| Cascade/OOD | {cascade['diagnostic_subset_records']} diagnostic records | coverage {cascade['coverage']:.4f} | selective-risk metrics | Safety annex |",
        ]
    )
    out = Path("reports/tables/external_validation_metrics.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("# External Validation Metrics\n\n" + table + "\n", encoding="utf-8")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
