from __future__ import annotations

from functools import lru_cache
import importlib
from pathlib import Path

import numpy as np
import torch

from titan_v4.reproducibility.refinement import sha256
from .contracts import FOUNDER_SHA256

SOURCE_SHA256 = '832bf90e05e20e62db6f9c7007acf7f0b1f739938ab80fa9b2c1ad438e8da7f7'


def build_founder(*, outputs: int = 150) -> torch.nn.Module:
    if outputs not in (150, 9, 10):
        raise ValueError('Unsupported teacher outputs')
    source = Path(__file__).parent / 'vendor/ecgfounder/net1d.py'
    if sha256(source) != SOURCE_SHA256:
        raise ValueError('Pinned teacher source hash differs')
    from .vendor.ecgfounder.net1d import Net1D

    return Net1D(in_channels=12, base_filters=64, ratio=1,
                 filter_list=[64, 160, 160, 400, 400, 1024, 1024],
                 m_blocks_list=[2, 2, 2, 3, 3, 4, 4], kernel_size=16,
                 stride=2, groups_width=16, n_classes=outputs,
                 use_bn=False, use_do=False, return_features=True, verbose=False).float()


def restricted_load(path: Path) -> dict:
    allowed_names = {'numpy.dtype', 'numpy.core.multiarray.scalar', 'numpy._core.multiarray.scalar'}
    unsafe = set(torch.serialization.get_unsafe_globals_in_checkpoint(path))
    if unsafe - allowed_names:
        raise ValueError('Checkpoint has unsafe globals: ' + ', '.join(sorted(unsafe - allowed_names)))
    core = importlib.import_module('numpy._core.multiarray' if hasattr(np, '_core') else 'numpy.core.multiarray')
    allowed = [np.dtype, (core.scalar, 'numpy.core.multiarray.scalar'),
               (core.scalar, 'numpy._core.multiarray.scalar'), type(np.dtype('float64')), type(np.dtype('int64'))]
    with torch.serialization.safe_globals(allowed):
        result = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(result, dict):
        raise ValueError('Restricted checkpoint must be a dictionary')
    return result


@lru_cache(maxsize=1)
def _schema():
    with torch.random.fork_rng(devices=[]):
        state = build_founder(outputs=150).state_dict()
    return {k: (tuple(v.shape), v.dtype) for k, v in state.items()}


def validate_founder_state(state: dict[str, torch.Tensor]) -> None:
    schema = _schema()
    if not isinstance(state, dict) or state.keys() != schema.keys():
        raise ValueError('Teacher body state keys differ')
    for key, (shape, dtype) in schema.items():
        value = state[key]
        if (not isinstance(value, torch.Tensor) or tuple(value.shape) != shape or value.dtype != dtype
                or not torch.isfinite(value).all()):
            raise ValueError('Teacher body state shape/dtype/value differs: ' + key)


def load_founder(checkpoint: Path, *, outputs: int, seed: int) -> torch.nn.Module:
    if outputs not in (150, 9, 10):
        raise ValueError('Unsupported teacher outputs')
    if sha256(checkpoint) != FOUNDER_SHA256:
        raise ValueError('Pretrained teacher checkpoint hash differs')
    loaded = restricted_load(checkpoint)
    state = loaded.get('state_dict')
    validate_founder_state(state)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = build_founder(outputs=150)
        model.load_state_dict(state, strict=True)
        if outputs != 150:
            model.dense = torch.nn.Linear(1024, outputs)
            model.n_classes = outputs
    return model


def founder_features(model: torch.nn.Module, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if x.ndim != 3 or tuple(x.shape[1:]) != (12, 5000) or x.dtype != torch.float32 or not torch.isfinite(x).all():
        raise ValueError('Invalid teacher input; require finite float32 [B,12,5000]')
    logits, features = model(x)
    if (logits.shape != (len(x), model.dense.out_features) or features.shape != (len(x), 1024)
            or not torch.isfinite(logits).all() or not torch.isfinite(features).all()):
        raise ValueError('Teacher feature/logit output contract differs')
    return logits, features
