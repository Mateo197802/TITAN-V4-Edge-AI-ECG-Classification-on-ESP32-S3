from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from titan_v4.metrics.external_validation import cascade_summary, load_json, pathology_summary, primary9_summary


def main() -> int:
    primary9 = primary9_summary(
        load_json("outputs/reproduced/primary9/primary9_recomputed_report.json")
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
            f"| Arrhythmia Primary-9 | {primary9['records']} records | {primary9['accuracy']:.4f} | {primary9['macro_f1']:.4f} | Recomputed full-support inference |",
            f"| Pathology Primary-5 | Historical aggregate | n/a | n/a | Legacy, not reproduced (reported {pathology['accuracy']:.5f}/{pathology['macro_f1']:.5f}) |",
            f"| Cascade/OOD | Historical aggregate | n/a | n/a | Safety annex only; legacy, not reproduced (n={cascade['diagnostic_subset_records']}, coverage={cascade['coverage']:.4f}) |",
        ]
    )
    out = Path("reports/tables/external_validation_metrics.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "# External Validation Metrics\n\n"
        "Only Primary-9 was recomputed from the distributed checkpoint and all 672 source records. "
        "Pathology and Cascade/OOD are preserved historical summaries, not current reproduced results.\n\n"
        + table
        + "\n",
        encoding="utf-8",
    )
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
