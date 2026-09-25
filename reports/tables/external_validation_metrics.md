# External Validation Metrics

Only Primary-9 was recomputed from the distributed checkpoint and all 672 source records. Pathology and Cascade/OOD are preserved historical summaries, not current reproduced results.

| Module | External scope | Accuracy | Macro-F1 | Evidence role |
|---|---:|---:|---:|---|
| Arrhythmia Primary-9 | 672 records | 0.6935 | 0.6888 | Recomputed full-support inference |
| Pathology Primary-5 | Historical aggregate | n/a | n/a | Legacy, not reproduced (reported 0.90256/0.66874) |
| Cascade/OOD | Historical aggregate | n/a | n/a | Safety annex only; legacy, not reproduced (n=211, coverage=0.5071) |
