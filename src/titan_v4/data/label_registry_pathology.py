"""
Pathology / morphology label registry for TITAN V4 (edge target).

This head is multi-label and intentionally smaller than rhythm head.
We start with 10 PTB-XL diagnostic/form statements with high support.
"""

from __future__ import annotations

from typing import Mapping, Sequence


PATHOLOGY_CLASS_NAMES = [
    "IMI",       # Inferior myocardial infarction
    "ASMI",      # Anteroseptal myocardial infarction
    "LVH",       # Left ventricular hypertrophy
    "ISC_",      # Non-specific ischemia
    "ISCAL",     # Ischemia anterior-lateral
    "NST_",      # Non-specific ST changes
    "ILMI",      # Infero-lateral myocardial infarction
    "AMI",       # Anterior myocardial infarction
    "ALMI",      # Antero-lateral myocardial infarction
    "LAE",       # Left atrial overload/enlargement
]

NUM_PATHOLOGY_CLASSES = len(PATHOLOGY_CLASS_NAMES)

# PTB-XL SCP statement -> pathology head index
SCP_TO_PATHOLOGY: dict[str, int] = {
    "IMI": 0,
    "ASMI": 1,
    "LVH": 2,
    "ISC_": 3,
    "ISCAL": 4,
    "NST_": 5,
    "ILMI": 6,
    "AMI": 7,
    "ALMI": 8,
    "LAO/LAE": 9,
}

# Optional SNOMED codes for pathology-only labels (not rhythms).
SNOMED_TO_PATHOLOGY: dict[str, int] = {
    "429622005": 5,  # ST Depression -> NST_ bucket (coarse)
    "164931005": 5,  # ST Elevation -> NST_ bucket (coarse, edge-friendly)
}


def pathology_labels_from_scp_codes(
    scp_scores: Mapping[str, float],
    mapping: Mapping[str, int] | None = None,
    min_score: float = 0.0,
) -> list[int]:
    label_mapping = SCP_TO_PATHOLOGY if mapping is None else mapping
    out: list[int] = []
    for code, score in scp_scores.items():
        if code not in label_mapping:
            continue
        try:
            if float(score) < float(min_score):
                continue
        except (TypeError, ValueError):
            continue
        out.append(int(label_mapping[code]))
    return sorted(set(out))


def pathology_labels_from_snomed_codes(
    codes: Sequence[str],
    mapping: Mapping[str, int] | None = None,
) -> list[int]:
    label_mapping = SNOMED_TO_PATHOLOGY if mapping is None else mapping
    out = [int(label_mapping[code]) for code in codes if code in label_mapping]
    return sorted(set(out))

