from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / 'data/cache/external_teachers/ecgfounder/12_lead_ECGFounder.pth'


def api():
    return importlib.import_module('titan_v4.reproducibility.pretrained_transfer.founder')


def test_wrong_checkpoint_hash_is_rejected(tmp_path):
    module = api()
    path = tmp_path / 'wrong.pth'
    torch.save({'state_dict': {'first_conv.conv.weight': torch.zeros(64, 12, 16)}}, path)
    with pytest.raises(ValueError, match='hash'):
        module.load_founder(path, outputs=9, seed=20261014)


def test_unknown_endpoint_is_rejected_before_loading():
    module = api()
    with pytest.raises(ValueError, match='outputs'):
        module.load_founder(Path('not_needed.pth'), outputs=14, seed=20261014)


@pytest.mark.parametrize('problem', ['extra', 'missing', 'shape', 'dtype', 'nonfinite'])
def test_body_schema_mutations_are_rejected(problem):
    module = api()
    state = module.build_founder(outputs=150).state_dict()
    name = 'first_conv.conv.weight'
    if problem == 'extra':
        state['unexpected.body.weight'] = torch.zeros(2)
    elif problem == 'missing':
        del state[name]
    elif problem == 'shape':
        state[name] = torch.zeros(64, 6, 16)
    elif problem == 'dtype':
        state[name] = state[name].double()
    else:
        state[name] = state[name].clone()
        state[name][0, 0, 0] = float('nan')
    with pytest.raises(ValueError, match='body state'):
        module.validate_founder_state(state)


def test_restricted_numpy_scalar_serialization(tmp_path):
    module = api()
    path = tmp_path / 'numpy.pth'
    torch.save({'value': np.float64(.5), 'tensor': torch.ones(2)}, path)
    result = module.restricted_load(path)
    assert result['value'] == .5
    assert torch.equal(result['tensor'], torch.ones(2))


class UnsafePickle:
    def __reduce__(self):
        return eval, ('1 + 1',)


def test_unapproved_pickle_global_is_not_executed(tmp_path):
    module = api()
    path = tmp_path / 'unsafe.pth'
    torch.save({'value': UnsafePickle()}, path)
    with pytest.raises(ValueError, match='unsafe globals'):
        module.restricted_load(path)


def test_adaptation_changes_only_dense_and_preserves_random_state():
    module = api()
    torch.manual_seed(47)
    initial_rng = torch.get_rng_state().clone()
    rhythm = module.load_founder(CHECKPOINT, outputs=9, seed=20261014)
    assert torch.equal(torch.get_rng_state(), initial_rng)
    pathology = module.load_founder(CHECKPOINT, outputs=10, seed=20261014)
    assert rhythm.first_conv.conv.weight.shape == (64, 12, 16)
    assert rhythm.dense.weight.shape == (9, 1024)
    assert pathology.dense.weight.shape == (10, 1024)
    assert rhythm.return_features and pathology.return_features
    a, b = rhythm.state_dict(), pathology.state_dict()
    assert a.keys() == b.keys()
    assert all(torch.equal(a[k], b[k]) for k in a if not k.startswith('dense.'))


def test_student_geometry_is_not_accepted_as_teacher_geometry():
    module = api()
    with pytest.raises(ValueError, match='teacher input'):
        module.founder_features(torch.nn.Identity(), torch.zeros(1, 6, 1250))
