from __future__ import annotations

from dataclasses import replace
import importlib

import numpy as np
import pytest
import torch


def teacher():
    return importlib.import_module('titan_v4.reproducibility.pretrained_transfer.teachers')


def selection():
    return importlib.import_module('titan_v4.reproducibility.pretrained_transfer.selection')


def test_unknown_pathology_has_zero_gradient():
    x = torch.zeros(3, 10, requires_grad=True)
    y = torch.zeros_like(x); mask = torch.zeros_like(x); mask[0, 0] = 1
    loss = teacher().teacher_pathology_loss(x, y, mask)
    loss.backward()
    assert torch.isfinite(loss) and torch.count_nonzero(x.grad[mask == 0]) == 0
    assert x.grad[0, 0] != 0


def test_empty_pathology_mask_has_finite_differentiable_zero():
    x = torch.randn(3, 10, requires_grad=True)
    loss = teacher().teacher_pathology_loss(x, torch.zeros_like(x), torch.zeros_like(x))
    loss.backward()
    assert loss.item() == 0 and torch.equal(x.grad, torch.zeros_like(x))


@pytest.mark.parametrize('kind', ['shape', 'target', 'mask', 'nonfinite'])
def test_invalid_pathology_contract_rejected(kind):
    x, y, mask = torch.zeros(3, 10), torch.zeros(3, 10), torch.ones(3, 10)
    if kind == 'shape': mask = mask[:, :9]
    if kind == 'target': y[0, 0] = .2
    if kind == 'mask': mask[0, 0] = .2
    if kind == 'nonfinite': x[0, 0] = float('nan')
    with pytest.raises(ValueError):
        teacher().teacher_pathology_loss(x, y, mask)


@pytest.mark.parametrize('stage', ['heads', 'adapt'])
def test_teacher_trainability_is_explicit_and_buffers_stay_frozen(stage):
    class Fixture(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.first_conv = torch.nn.Conv1d(12, 4, 1)
            self.first_bn = torch.nn.BatchNorm1d(4)
            self.stage_list = torch.nn.ModuleList([torch.nn.Linear(4, 4), torch.nn.Linear(4, 4)])
            self.dense = torch.nn.Linear(4, 9)
    model = Fixture()
    teacher().configure_teacher(model, stage)
    for name, p in model.named_parameters():
        assert p.requires_grad == (name.startswith('dense.') or
                                  (stage == 'adapt' and name.startswith('stage_list.1.')))
    assert not model.first_bn.training


def sample_rows():
    from titan_v4.reproducibility.pretrained_transfer.contracts import PairedRecord
    return [PairedRecord('ptb-xl/1.0.3/1', 'patient', 'train', 'record', 'record.dat',
                         'a' * 64, 'b' * 64, 0, 5000, 0, (0.,) * 10, (True,) * 10, 'c' * 64)]


@pytest.mark.parametrize('role', ['tracking', 'historical', 'rhythm_validation', 'train_excluded'])
def test_teacher_fit_rejects_nontraining_rows(role):
    with pytest.raises(ValueError, match='train'):
        teacher().fitting_rows([replace(sample_rows()[0], role=role)], 'rhythm')


def test_validation_metrics_cannot_be_derived_from_tracking():
    with pytest.raises(ValueError, match='validation'):
        selection().endpoint_metrics(np.zeros((2, 9)), np.array([0, 1]),
                                     np.ones(2), np.array(['x', 'x']), np.array(['p1', 'p2']),
                                     endpoint='rhythm', role='tracking')


def test_prediction_writer_cannot_fit_metrics_on_tracking_rows(tmp_path):
    rows = [replace(sample_rows()[0], role='tracking')]
    with pytest.raises(ValueError, match='validation'):
        teacher().save_predictions(tmp_path, rows, np.zeros((1, 9)), 'rhythm')


def metrics(f1=.8, accuracy=.85, ap=.8):
    return {'macro_f1': f1, 'accuracy': accuracy, 'macro_ap': ap,
            'classes': {'a': {'support': 50, 'recall': .8}},
            'sources': {'source': {'support': 200, 'macro_f1': f1}}}


def test_three_seeds_not_best_seed_determine_eligibility():
    result = selection().eligibility([metrics(.99), metrics(.7), metrics(.7)], metrics(.8), 'rhythm')
    assert not result['eligible']
    assert result['mean']['macro_f1'] == pytest.approx((.99 + .7 + .7) / 3)


@pytest.mark.parametrize('endpoint', ['rhythm', 'pathology'])
def test_mean_and_class_guards_apply_to_teacher(endpoint):
    module = selection()
    control = metrics()
    assert module.eligibility([metrics(.83, .86, .82)] * 3, control, endpoint)['eligible']
    low_recall = metrics(.83, .86, .82); low_recall['classes']['a']['recall'] = .70
    assert not module.eligibility([low_recall] * 3, control, endpoint)['eligible']
    assert not module.eligibility([metrics(.83, .83, .8)] * 3, control, endpoint)['eligible']


def test_missing_seed_is_not_teacher_qualification():
    with pytest.raises(ValueError, match='three'):
        selection().eligibility([metrics(), metrics()], metrics(), 'rhythm')


def test_teacher_logits_use_endpoint_semantics_not_original_head_unions():
    logits = np.zeros((2, 10))
    y = np.zeros((2, 10)); y[0] = 1
    with pytest.raises(ValueError, match='shape'):
        selection().endpoint_metrics(logits[:, :9], y, np.ones_like(y), np.array(['ptb-xl'] * 2),
                                     np.array(['p1', 'p2']), endpoint='pathology', role='validation')


def test_teacher_fit_never_runs_without_slurm(tmp_path, monkeypatch):
    monkeypatch.delenv('SLURM_JOB_ID', raising=False)
    with pytest.raises(RuntimeError, match='Slurm'):
        teacher().fit_teacher(tmp_path, endpoint='rhythm', contract='canonical_unfiltered',
                              seed=20261014, stage='heads')


@pytest.mark.parametrize('change', ['class_order', 'role', 'duplicate', 'hash'])
def test_validation_prediction_loader_rejects_stale_or_mismatched_identity(tmp_path, change):
    from titan_v4.reproducibility.pretrained_transfer.contracts import PRIMARY9_CLASSES
    from titan_v4.reproducibility.pretrained_transfer.execution import file_sha
    classes = list(PRIMARY9_CLASSES)
    if change == 'class_order': classes.reverse()
    path = tmp_path / 'validation.npz'
    np.savez(path, classes=np.asarray(classes),
             roles=np.asarray(['tracking' if change == 'role' else 'rhythm_validation'] * 2),
             keys=np.asarray(['a', 'a' if change == 'duplicate' else 'b']))
    record = {'endpoint': 'rhythm', 'validation_npz': str(path),
              'validation_npz_sha256': '0' * 64 if change == 'hash' else file_sha(path)}
    with pytest.raises(ValueError):
        selection().load_predictions(record)


def test_teacher_qualification_cannot_select_tracking_or_a_single_seed():
    with pytest.raises(ValueError, match='validation'):
        selection().qualify_teachers([{'scope': 'tracking'}], {})
    with pytest.raises(ValueError, match='three'):
        selection().qualify_teachers([{'scope': 'validation', 'endpoint': 'rhythm',
            'contract': 'canonical_unfiltered', 'stage': 'heads', 'seed': 20261014}], {})


def test_frozen_teacher_check_detects_buffer_drift():
    model = torch.nn.BatchNorm1d(4)
    initial = {k: v.clone() for k, v in model.state_dict().items()}
    model.running_mean.add_(1)
    with pytest.raises(ValueError, match='frozen'):
        teacher().assert_teacher_frozen(model, initial, {'weight', 'bias'})


@pytest.mark.parametrize('field', ['paired_manifest_sha256', 'classes', 'contract'])
def test_teacher_delta_requires_exact_input_and_label_semantics(tmp_path, field):
    from titan_v4.reproducibility.pretrained_transfer.contracts import PRIMARY9_CLASSES
    from titan_v4.reproducibility.pretrained_transfer.execution import MANIFEST_SHA
    module = teacher()
    state = {'teacher_sha256': module.FOUNDER_SHA256, 'paired_manifest_sha256': MANIFEST_SHA,
             'endpoint': 'rhythm', 'seed': 20261014, 'stage': 'heads',
             'classes': list(PRIMARY9_CLASSES), 'contract': 'canonical_unfiltered', 'delta': {}}
    state[field] = ['wrong'] if field == 'classes' else 'wrong'
    path = tmp_path / 'checkpoint.pth'; torch.save(state, path)
    with pytest.raises(ValueError, match='provenance'):
        module.load_teacher_delta(tmp_path, path, endpoint='rhythm', seed=20261014)
