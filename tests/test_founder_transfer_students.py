from dataclasses import replace
import importlib

import numpy as np
import pytest
import torch


def api():
    return importlib.import_module('titan_v4.reproducibility.pretrained_transfer.students')


def selection():
    return importlib.import_module('titan_v4.reproducibility.pretrained_transfer.selection')


def model():
    from titan_v4.models.titan_v4_net import TitanV4Lite
    return TitanV4Lite(in_channels=6, num_rhythm=9, num_pathology=10)


def test_student_contract_preserves_parameters_heads_and_strides():
    value = model()
    api().assert_deployment_contract(value)
    assert sum(p.numel() for p in value.parameters()) == 1143751
    assert len(value.res_layers) == 5 and value.morphology_dim == 0


@pytest.mark.parametrize('kind', ['projection', 'stride', 'head'])
def test_student_contract_rejects_added_modules_or_changed_operations(kind):
    value = model()
    if kind == 'projection': value.projection = torch.nn.Linear(192, 10)
    if kind == 'stride': value.res_layers[4].conv1.stride = (1,)
    if kind == 'head': value.head_quality = torch.nn.Linear(192, 1)
    with pytest.raises(ValueError):
        api().assert_deployment_contract(value)


def test_student_trainability_and_default_bn_freeze():
    value = model()
    api().configure_student(value)
    for name, p in value.named_parameters():
        assert p.requires_grad == name.startswith(('res_layers.3.', 'res_layers.4.', 'head_rhythm.', 'head_pathology.'))
    assert all(not m.training for m in value.modules() if isinstance(m, torch.nn.BatchNorm1d))
    assert value.head_rhythm.training and not value.head_quality.training and not value.stem.training


@pytest.mark.parametrize('kind', ['quality', 'biometrics', 'bn'])
def test_frozen_auxiliary_parameters_and_buffers_cannot_drift(kind):
    value = model(); api().configure_student(value)
    initial = {k: v.clone() for k, v in value.state_dict().items()}
    with torch.no_grad():
        if kind == 'quality': value.head_quality[0].weight.add_(1)
        if kind == 'biometrics': value.head_biometrics[0].weight.add_(1)
        if kind == 'bn': value.stem[1].running_mean.add_(1)
    with pytest.raises(ValueError, match='frozen'):
        api().assert_student_preserved(value, initial, arm='supervised')


def test_only_named_bn_control_can_change_bn_buffers():
    value = model(); api().configure_student(value)
    initial = {k: v.clone() for k, v in value.state_dict().items()}
    value.stem[1].running_mean.add_(1)
    api().assert_student_preserved(value, initial, arm='supervised_bn')


def test_wrong_student_warm_start_is_rejected_before_loading(tmp_path):
    path = tmp_path / 'start.pth'; path.write_bytes(b'wrong')
    with pytest.raises(ValueError, match='warm'):
        api().load_student(path)


@pytest.mark.parametrize('arm', ['rhythm_transfer', 'pathology_transfer', 'joint_transfer'])
def test_ineligible_teacher_arm_cannot_train(arm):
    with pytest.raises(ValueError, match='teacher'):
        api().require_arm(arm, {})


def test_student_empty_supervision_stays_zero_with_zero_gradients():
    values = {'rhythm': torch.zeros(3, 9, requires_grad=True), 'pathology': torch.zeros(3, 10, requires_grad=True)}
    result = api().supervised_losses(values, torch.full((3,), -1), torch.zeros(3, 10),
                                     torch.zeros(3, 10), torch.ones(3), torch.ones(10))
    (result['rhythm_loss'] + result['pathology_loss']).backward()
    assert result['rhythm_loss'].item() == result['pathology_loss'].item() == 0
    assert values['rhythm'].grad.abs().sum() == values['pathology'].grad.abs().sum() == 0


def test_student_supervision_rejects_wrong_masks():
    values = {'rhythm': torch.zeros(2, 9), 'pathology': torch.zeros(2, 10)}
    with pytest.raises(ValueError):
        api().supervised_losses(values, torch.zeros(2, dtype=torch.long), torch.zeros(2, 10),
                                torch.zeros(2, 9), torch.ones(2), torch.ones(10))


def test_balanced_bn_selection_is_training_only_and_deterministic():
    from titan_v4.reproducibility.pretrained_transfer.contracts import PairedRecord
    a = PairedRecord('ptb-xl/1.0.3/1', None, 'train', 'r1', 'r1.dat', 'a' * 64, 'b' * 64,
                     0, 5000, 0, (0.,) * 10, (True,) * 10, 'c' * 64)
    rows = [a, replace(a, key='ptb-xl/1.0.3/2', rhythm_label=1)]
    indices = api().balanced_bn_indices(rows, 20261014)
    assert len(indices) == 256 and np.array_equal(indices, api().balanced_bn_indices(rows, 20261014))
    assert set(indices.tolist()) == {0, 1}
    with pytest.raises(ValueError, match='train'):
        api().balanced_bn_indices([replace(a, role='tracking')], 20261014)


def endpoint(f1, accuracy=.85):
    return {'macro_f1': f1, 'accuracy': accuracy, 'macro_ap': f1,
            'classes': {'a': {'support': 50, 'recall': .8}},
            'sources': {'source': {'support': 200, 'macro_f1': f1}}}


def measures(r=.8, p=.5):
    return {'rhythm': endpoint(r), 'pathology': endpoint(p, .95)}


def test_student_promotion_requires_gain_without_other_endpoint_regression():
    baseline = [measures()] * 3
    assert selection().student_eligibility([measures(.82, .499)] * 3, baseline)['eligible']
    assert selection().student_eligibility([measures(.799, .52)] * 3, baseline)['eligible']
    assert not selection().student_eligibility([measures(.82, .49)] * 3, baseline)['eligible']


def test_student_best_seed_cannot_replace_three_seed_mean():
    result = selection().student_eligibility([measures(.95), measures(.7), measures(.7)], [measures()] * 3)
    assert not result['eligible']


def test_student_class_preservation_is_not_hidden_by_macro_gain():
    candidate = measures(.85, .52); candidate['rhythm']['classes']['a']['recall'] = .65
    assert not selection().student_eligibility([candidate] * 3, [measures()] * 3)['eligible']


def test_student_selector_cannot_use_tracking_metrics():
    with pytest.raises(ValueError, match='validation'):
        selection().select_students([{'scope': 'tracking'}], {})


def test_student_fit_requires_slurm(tmp_path, monkeypatch):
    monkeypatch.delenv('SLURM_JOB_ID', raising=False)
    with pytest.raises(RuntimeError, match='Slurm'):
        api().fit_student(tmp_path, arm='supervised', seed=20261014)


def test_existing_warm_start_loads_strictly_without_inference():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    value = api().load_student(root / 'outputs/reviewer_verification/supervision_control_2026-10-14/model/joint_selected.pth')
    api().assert_deployment_contract(value)


def test_fixed_deployment_seed_must_qualify_independently(tmp_path, monkeypatch):
    from titan_v4.reproducibility.pretrained_transfer.contracts import SEEDS
    module = selection()
    monkeypatch.setattr(api(), 'load_exported_student', lambda *a: None)
    def data(record):
        return {'keys': np.array(['one']), 'labels': np.array([0]), 'mask': np.array([1]),
                'sources': np.array(['source']), 'patients': np.array(['patient']),
                'roles': np.array([record['endpoint'] + '_validation']), 'classes': np.array(['class']),
                'fixture_metrics': record['fixture_metrics']}
    monkeypatch.setattr(module, 'load_predictions', data)
    monkeypatch.setattr(module, 'prediction_metrics', lambda d, endpoint: d['fixture_metrics'])
    results = []
    for arm in ('supervised', 'rhythm_transfer', 'joint_transfer'):
        for i, seed in enumerate(SEEDS):
            measure = measures() if arm == 'supervised' else (
                measures(.799 if i == 0 else .95) if arm == 'rhythm_transfer' else measures(.82, .52))
            measure['pathology']['thresholds'] = [.5] * 10
            results.append({'scope': 'validation', 'arm': arm, 'seed': seed,
                'checkpoint': str(tmp_path / (arm + str(seed) + '.pth')), 'checkpoint_sha256': 'a' * 64,
                'elapsed_seconds': 1, 'validation': {e: {'endpoint': e, 'fixture_metrics': m} for e, m in measure.items()}})
    result = module.select_students(results, {})
    assert result['selected_arm'] == 'joint_transfer' and result['seed'] == SEEDS[0]
    rejected = next(r for r in result['comparisons'] if r['arm'] == 'rhythm_transfer')
    assert rejected['eligible'] and not rejected['fixed_seed_eligible']


def test_student_checkpoint_cannot_silently_cast_weight_dtype(tmp_path):
    from pathlib import Path
    from titan_v4.reproducibility.pretrained_transfer.execution import file_sha
    root = Path(__file__).resolve().parents[1]
    data = torch.load(root / 'outputs/reviewer_verification/supervision_control_2026-10-14/model/joint_selected.pth',
                      map_location='cpu', weights_only=True)
    data['state_dict']['stem.0.weight'] = data['state_dict']['stem.0.weight'].half()
    path = tmp_path / 'wrong_dtype.pth'; torch.save(data, path)
    with pytest.raises(ValueError, match='state'):
        api().load_exported_student(path, file_sha(path))


def test_student_operator_fingerprint_includes_padding_behavior():
    value = model(); value.stem[0].padding_mode = 'reflect'
    with pytest.raises(ValueError):
        api().assert_deployment_contract(value)


def test_improved_supervised_control_is_not_discarded_when_transfer_adds_nothing(tmp_path, monkeypatch):
    from titan_v4.reproducibility.pretrained_transfer.contracts import SEEDS
    module = selection()
    monkeypatch.setattr(api(), 'load_exported_student', lambda *a: None)
    def data(record):
        return {'keys': np.array(['one']), 'labels': np.array([0]), 'mask': np.array([1]),
                'sources': np.array(['source']), 'patients': np.array(['patient']),
                'roles': np.array([record['endpoint'] + '_validation']), 'classes': np.array(['class']),
                'fixture_metrics': record['fixture_metrics']}
    monkeypatch.setattr(module, 'load_predictions', data)
    monkeypatch.setattr(module, 'prediction_metrics', lambda d, endpoint: d['fixture_metrics'])
    results = []
    for seed in SEEDS:
        measure = measures(.82, .52); measure['pathology']['thresholds'] = [.5] * 10
        results.append({'scope': 'validation', 'arm': 'supervised', 'seed': seed,
            'checkpoint': str(tmp_path / ('supervised-' + str(seed) + '.pth')), 'checkpoint_sha256': 'a' * 64,
            'elapsed_seconds': 1, 'validation': {e: {'endpoint': e, 'fixture_metrics': m} for e, m in measure.items()}})
    warm = measures(); warm['pathology']['thresholds'] = [.5] * 10
    selected = module.select_students(results, {'checkpoint': 'warm.pth', 'checkpoint_sha256': 'b' * 64, 'metrics': warm})
    assert selected['selected_arm'] == 'supervised' and selected['promoted']


def test_new_training_receipt_preserves_pre_recalibration_buffers(monkeypatch):
    value = model()
    optimizer = torch.optim.AdamW(value.head_rhythm.parameters(), lr=.001)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
    monkeypatch.setattr(torch.cuda, 'get_rng_state_all', lambda: [])
    snapshot = api().training_snapshot(value, optimizer, scheduler, epoch=2, seed=20261014)
    saved = snapshot['training_buffers']['stem.1.running_mean'].clone()
    value.stem[1].running_mean.add_(1)
    assert torch.equal(snapshot['training_buffers']['stem.1.running_mean'], saved)
    assert snapshot['next_permutation_seed'] == 20261017
