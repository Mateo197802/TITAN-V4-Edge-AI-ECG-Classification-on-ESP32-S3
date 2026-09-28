"""Shape-checked architecture metadata for the preserved CEDIA checkpoint."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class CediaCheckpointSpec:
    architecture: str
    encoder_dim: int
    rhythm_classes: int
    pathology_classes: int
    morphology_dim: int


def inspect_cedia_checkpoint_shapes(state_dict: Mapping[str, object]) -> CediaCheckpointSpec:
    """Identify the saved Max checkpoint from its heads before strict loading."""

    required_shapes = {
        "head_rhythm.0.weight": (256, 1024),
        "head_quality.0.weight": (64, 1024),
        "head_biometrics.0.weight": (64, 1024),
        "head_pathology.0.weight": (256, 1024),
    }
    actual_shapes: dict[str, tuple[int, ...]] = {}
    for key in (
        *required_shapes,
        "head_rhythm.3.weight",
        "head_pathology.3.weight",
    ):
        value = state_dict.get(key)
        shape = getattr(value, "shape", None)
        if shape is None:
            raise ValueError(f"CEDIA checkpoint is missing tensor {key}")
        actual_shapes[key] = tuple(int(dimension) for dimension in shape)

    for key, expected in required_shapes.items():
        if actual_shapes[key] != expected:
            raise ValueError(
                f"Unsupported CEDIA checkpoint architecture at {key}: "
                f"expected {expected}, found {actual_shapes[key]}"
            )
    rhythm_classes = actual_shapes["head_rhythm.3.weight"][0]
    pathology_classes = actual_shapes["head_pathology.3.weight"][0]
    if actual_shapes["head_rhythm.3.weight"][1] != 256:
        raise ValueError("Unexpected CEDIA rhythm output-layer input dimension")
    if actual_shapes["head_pathology.3.weight"][1] != 256:
        raise ValueError("Unexpected CEDIA pathology output-layer input dimension")

    return CediaCheckpointSpec(
        architecture="TitanV4Max",
        encoder_dim=1024,
        rhythm_classes=rhythm_classes,
        pathology_classes=pathology_classes,
        morphology_dim=0,
    )
