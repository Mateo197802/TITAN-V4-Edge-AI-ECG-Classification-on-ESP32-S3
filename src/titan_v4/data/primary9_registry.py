"""Central registry for the TITAN V4 Primary-9 rhythm taxonomy."""

from __future__ import annotations

from collections.abc import Sequence


PRIMARY9_CLASSES = [
    "AFIB",
    "SB",
    "STACH",
    "NSR",
    "PVC",
    "RBBB",
    "LBBB",
    "PAC",
    "1AVB",
]

RARE_QUARANTINE_CLASSES = [
    "2AVB",
    "3AVB",
    "Flutter",
    "LQTS",
    "Paced",
]

FULL14_CLASSES = [
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

PRIMARY9_INDEX_BY_NAME = {name: idx for idx, name in enumerate(PRIMARY9_CLASSES)}
FULL14_INDEX_BY_NAME = {name: idx for idx, name in enumerate(FULL14_CLASSES)}
PRIMARY9_TO_FULL14_INDEX = [FULL14_INDEX_BY_NAME[name] for name in PRIMARY9_CLASSES]

PRIMARY9_TEACHER_GROUPS = {
    "rate": ["SB", "NSR", "STACH"],
    "ectopy": ["PAC", "PVC", "NSR"],
    "conduction": ["RBBB", "LBBB", "1AVB", "NSR"],
    "atrial": ["AFIB", "NSR"],
}

PRIMARY9_TEACHER_GROUP_INDICES = {
    name: [PRIMARY9_INDEX_BY_NAME[class_name] for class_name in classes]
    for name, classes in PRIMARY9_TEACHER_GROUPS.items()
}


def normalize_rhythm_name(name: object) -> str:
    """Return a canonical rhythm name string for manifest values."""
    return str(name or "").strip()


def project_full14_label_to_primary9(label_name: object) -> int | None:
    """Map a full-14 class name to the Primary-9 index, or None for rare/unknown."""
    name = normalize_rhythm_name(label_name)
    return PRIMARY9_INDEX_BY_NAME.get(name)


def project_full14_probs_to_primary9(full14_probs: Sequence[float]) -> list[float]:
    """Project a 14-class probability/logit vector into Primary-9 class order."""
    if len(full14_probs) < len(FULL14_CLASSES):
        raise ValueError(f"Expected at least {len(FULL14_CLASSES)} values, got {len(full14_probs)}")
    return [float(full14_probs[idx]) for idx in PRIMARY9_TO_FULL14_INDEX]


def is_primary9_class(label_name: object) -> bool:
    return project_full14_label_to_primary9(label_name) is not None


def is_rare_quarantine_class(label_name: object) -> bool:
    return normalize_rhythm_name(label_name) in set(RARE_QUARANTINE_CLASSES)
