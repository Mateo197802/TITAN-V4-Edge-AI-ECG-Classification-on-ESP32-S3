from __future__ import annotations

from types import SimpleNamespace

import pytest

from titan_v4.evaluation.cedia_checkpoint import inspect_cedia_checkpoint_shapes


def _state_shapes():
    return {
        "head_rhythm.0.weight": (256, 1024),
        "head_quality.0.weight": (64, 1024),
        "head_biometrics.0.weight": (64, 1024),
        "head_pathology.0.weight": (256, 1024),
        "head_rhythm.3.weight": (9, 256),
        "head_pathology.3.weight": (10, 256),
    }


def test_checkpoint_shape_inspection_identifies_max_and_head_sizes():
    spec = inspect_cedia_checkpoint_shapes(
        {key: SimpleNamespace(shape=shape) for key, shape in _state_shapes().items()}
    )

    assert spec.architecture == "TitanV4Max"
    assert spec.encoder_dim == 1024
    assert spec.rhythm_classes == 9
    assert spec.pathology_classes == 10
    assert spec.morphology_dim == 0


def test_checkpoint_shape_inspection_rejects_wrong_architecture():
    shapes = _state_shapes()
    shapes["head_quality.0.weight"] = (64, 320)

    with pytest.raises(ValueError, match="Unsupported CEDIA checkpoint architecture"):
        inspect_cedia_checkpoint_shapes(
            {key: SimpleNamespace(shape=shape) for key, shape in shapes.items()}
        )
