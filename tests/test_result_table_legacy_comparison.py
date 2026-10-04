from __future__ import annotations

from scripts.build_result_tables import render_table


def test_rendered_metrics_table_separates_current_score_from_historical_aggregate():
    rendered = render_table(
        primary9={"records": 672, "accuracy": 0.69345, "macro_f1": 0.68884, "weighted_f1": 0.68959},
        legacy_primary9={"accuracy": 0.90030, "macro_f1": 0.87819, "weighted_f1": 0.90085},
        pathology={"accuracy": 0.90256, "macro_f1": 0.66874},
        cascade={"diagnostic_subset_records": 211, "coverage": 0.5071},
        cedia={"validation_windows": 58855, "accuracy": 0.89836, "macro_f1": 0.73898, "weighted_f1": 0.90301},
        checkpoint_training={"val_windows": 706, "best_val_f1_macro": 0.89478},
        legacy_comparison={
            "current_checkpoint_vs_historical_labels": {
                "total_records": 672,
                "correct_predictions": 568,
                "accuracy": 0.84524,
                "macro_f1": 0.82352,
                "weighted_f1": 0.84641,
            }
        },
    )

    assert "Current checkpoint vs prior Primary-9 labels" in rendered
    assert "568/672 correct" in rendered
    assert "not a reproduction of the historical 605/672 aggregate" in rendered
    assert "Historical Primary-9 aggregate" in rendered
