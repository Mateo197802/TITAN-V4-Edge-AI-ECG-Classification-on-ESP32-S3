"""
Central label registry for TITAN V4 rhythm classification.

This file is the single source of truth for the rhythm head. Records
without a trusted label are excluded from training instead of being silently
converted to NSR.
"""
from __future__ import annotations

import os
from typing import Mapping, Sequence


RHYTHM_CLASS_NAMES = [
    "AFIB",
    "SB",
    "STACH",
    "NSR",
    "PVC",
    "RBBB",
    "LBBB",
    "PAC",
    "1AVB",
    "2AVB",
    "3AVB",
    "Flutter",
    "LQTS",
    "Paced",
]
NUM_RHYTHM_CLASSES = len(RHYTHM_CLASS_NAMES)


SNOMED_TO_RHYTHM = {
    "164889003": 0,   # AFIB
    "426177001": 1,   # Sinus bradycardia
    "427084000": 2,   # Sinus tachycardia
    "426783006": 3,   # Sinus rhythm
    "164884008": 4,   # PVC
    "427172004": 4,   # Premature ventricular contractions
    "17338001": 4,    # Ventricular premature beats
    "59118001": 5,    # RBBB
    "713427006": 5,   # Complete RBBB
    "713426002": 5,   # Incomplete RBBB
    "164909002": 6,   # LBBB
    "284470004": 7,   # PAC
    "63593006": 7,    # Supraventricular premature beats
    "270492004": 8,   # First-degree AV block
    "195042002": 9,   # Second-degree AV block
    "28189009": 9,    # Mobitz type II second-degree AV block
    "27885002": 10,   # Third-degree/complete AV block
    "164890007": 11,  # Atrial flutter
    "111975006": 12,  # Prolonged QT interval
    "10370003": 13,   # Paced rhythm
}


EXCLUDED_SNOMED_CODES = {
    "429622005": "ST depression is a morphology/pathology label, not a rhythm class",
    "164931005": "ST elevation is a morphology/pathology label, not a rhythm class",
    "426627000": "Challenge 2020 uses this as bradycardia, not sinus tachycardia",
    "233896004": "AVNRT is not third-degree AV block",
    "164947007": "Prolonged PR interval is not atrial flutter",
    "251146004": "Low QRS voltage is not long QT syndrome",
    "698252002": "Nonspecific intraventricular conduction disorder is not WPW",
    "74390002": "WPW removed from the target rhythm head due insufficient evidence",
}


SCP_TO_RHYTHM = {
    "AFIB": 0,
    "SBRAD": 1,
    "STACH": 2,
    "SR": 3,
    "PVC": 4,
    "VPB": 4,
    "RBBB": 5,
    "CRBBB": 5,
    "IRBBB": 5,
    "LBBB": 6,
    "CLBBB": 6,
    "ILBBB": 6,
    "PAC": 7,
    "SVPB": 7,
    "1AVB": 8,
    "2AVB": 9,
    "3AVB": 10,
    "AFLT": 11,
    "LNGQT": 12,
    "LQT": 12,
    "PACE": 13,
}


SOURCE_POLICIES = {
    "challenge2020": "read_dx_snomed",
    "cpsc2018": "read_dx_snomed",
    "georgia": "read_dx_snomed",
    "ptb-xl": "read_scp_metadata",
    "physionet_no_dx": "skip_until_adapter",
}


def normalize_record_path(record_base: str) -> str:
    return os.path.normcase(os.path.abspath(record_base))


def parse_dx_codes_from_header(record_base: str) -> list[str]:
    header_path = record_base if record_base.endswith(".hea") else record_base + ".hea"
    try:
        with open(header_path, "r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                stripped = line.strip()
                if (
                    stripped.startswith("# Dx:")
                    or stripped.startswith("#Dx:")
                    or stripped.startswith("Dx:")
                ):
                    return [
                        code.strip()
                        for code in stripped.split(":", 1)[1].split(",")
                        if code.strip()
                    ]
    except OSError:
        return []
    return []


def labels_from_codes(
    codes: Sequence[str],
    mapping: Mapping[str, int] | None = None,
) -> list[int]:
    label_mapping = SNOMED_TO_RHYTHM if mapping is None else mapping
    return [int(label_mapping[code]) for code in codes if code in label_mapping]


def exclusion_reason_for_codes(codes: Sequence[str], mapping: Mapping[str, int] | None = None) -> str:
    if not codes:
        return "no_dx"
    label_mapping = SNOMED_TO_RHYTHM if mapping is None else mapping
    if any(code in label_mapping for code in codes):
        return "labeled"
    if any(code in EXCLUDED_SNOMED_CODES for code in codes):
        return "excluded_non_rhythm"
    return "unknown_dx"


def primary_label_from_codes(
    codes: Sequence[str],
    mapping: Mapping[str, int] | None = None,
) -> int | None:
    labels = labels_from_codes(codes, mapping=mapping)
    if not labels:
        return None
    for label in labels:
        if int(label) != 3:
            return int(label)
    return int(labels[0])


def primary_scp_label_from_scores(
    scp_scores: Mapping[str, float],
    mapping: Mapping[str, int] | None = None,
) -> tuple[int, str] | None:
    label_mapping = SCP_TO_RHYTHM if mapping is None else mapping
    candidates = [
        (float(score), code, int(label_mapping[code]))
        for code, score in scp_scores.items()
        if code in label_mapping
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda item: (-item[0], item[1]))
    _score, code, label = candidates[0]
    return label, code
