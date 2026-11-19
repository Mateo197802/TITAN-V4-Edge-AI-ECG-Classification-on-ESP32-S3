"""Audit target gaps and the recorded CE/KL objective without changing labels."""
import math

import numpy as np


def additional_correct(records: int, correct: int, accuracy: float = .9) -> int:
    if records < 1 or not 0 <= correct <= records or not 0 <= accuracy <= 1:
        raise ValueError("Invalid evaluation counts or target")
    return max(0, math.ceil(records * accuracy) - correct)


def effective_rhythm_target(y, gold, teacher, gold_weight, teacher_weight):
    y = np.asarray(y)
    gold, teacher = np.asarray(gold, dtype=np.float64), np.asarray(teacher, dtype=np.float64)
    if (gold.ndim != 2 or gold.shape != teacher.shape or gold.shape[0] != len(y)
        or not np.issubdtype(y.dtype, np.integer) or np.any(y < 0) or np.any(y >= gold.shape[1])):
        raise ValueError("Labels and probability dimensions differ")
    if not np.isfinite([gold_weight,teacher_weight]).all() or min(gold_weight,teacher_weight) < 0:
        raise ValueError("KL weights must be finite and nonnegative")
    for probabilities in (gold,teacher):
        if (not np.isfinite(probabilities).all() or np.any(probabilities < 0)
            or not np.allclose(probabilities.sum(1),1,atol=2e-6,rtol=0)):
            raise ValueError("Invalid teacher probabilities")
    # For free per-record q, CE(y,q) + a KL(g||q) + b KL(t||q) minimizes at this mixture.
    target = np.eye(gold.shape[1])[y] + gold_weight * gold + teacher_weight * teacher
    return target / (1 + gold_weight + teacher_weight)
