from __future__ import annotations

from collections.abc import Iterable

from titan_v4.data.label_registry import SNOMED_TO_RHYTHM
from titan_v4.data.primary9_registry import PRIMARY9_CLASSES


PROLONGED_PR_CODE = "164947007"
FIRST_DEGREE_AV_BLOCK_INDEX = PRIMARY9_CLASSES.index("1AVB")
NORMAL_RHYTHM_INDEX = PRIMARY9_CLASSES.index("NSR")


def parse_dx_codes_from_comments(comments: Iterable[str]) -> list[str]:
    for comment in comments:
        key, separator, value = str(comment).partition(":")
        if separator and key.strip().lstrip("# ").casefold() == "dx":
            codes = [part.strip() for part in value.replace(",", "|").split("|")]
            return [code for code in codes if code]
    raise ValueError("WFDB header has no Dx diagnosis codes")


def primary9_names_from_dx_codes(codes: Iterable[str]) -> list[str]:
    """Map upstream SNOMED diagnoses into the ordered Primary-9 label set."""
    names: list[str] = []
    for raw_code in codes:
        code = str(raw_code).strip()
        if code == PROLONGED_PR_CODE:
            index = FIRST_DEGREE_AV_BLOCK_INDEX
        else:
            index = SNOMED_TO_RHYTHM.get(code)
        if index is None or index >= len(PRIMARY9_CLASSES):
            continue
        name = PRIMARY9_CLASSES[index]
        if name not in names:
            names.append(name)
    return names


def primary9_label_from_dx_codes(codes: Iterable[str]) -> str:
    """Choose the first non-normal rhythm; use prolonged PR as a fallback 1AVB label.

    SNOMED 164947007 is a prolonged-PR annotation, not an independently scored
    rhythm diagnosis. It maps to 1AVB only when no other non-normal Primary-9
    rhythm code is present. This avoids allowing that secondary annotation to
    override a more specific rhythm diagnosis in the same WFDB header.
    """
    normalized = [str(code).strip() for code in codes if str(code).strip()]
    explicit_codes = [code for code in normalized if code != PROLONGED_PR_CODE]
    for name in primary9_names_from_dx_codes(explicit_codes):
        if name != "NSR":
            return name
    if PROLONGED_PR_CODE in normalized:
        return PRIMARY9_CLASSES[FIRST_DEGREE_AV_BLOCK_INDEX]
    if "NSR" in primary9_names_from_dx_codes(explicit_codes):
        return PRIMARY9_CLASSES[NORMAL_RHYTHM_INDEX]
    raise ValueError("WFDB Dx codes do not map to a Primary-9 rhythm label")
