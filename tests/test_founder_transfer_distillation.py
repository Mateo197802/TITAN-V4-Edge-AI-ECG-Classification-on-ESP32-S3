from dataclasses import replace
import importlib

import numpy as np
import pytest
import torch


def api():
    return importlib.import_module('titan_v4.reproducibility.pretrained_transfer.distillation')


def rows():
    from titan_v4.reproducibility.pretrained_transfer.contracts import PairedRecord
    row = PairedRecord('ptb-xl/1.0.3/1', 'patient1', 'train', 'record1', 'record1.dat',
                       'a' * 64, 'b' * 64, 0, 5000, 0, (0.,) * 10, (True,) * 10, 'c' * 64)
    return [row, replace(row, key='ptb-xl/1.0.3/2', patient_token='patient2', rhythm_label=1,
                         record_path='record2', signal_filename='record2.dat', signal_sha256='d' * 64,
                         student_tensor_sha256='e' * 64)]


def qualification(bundle, endpoints=('rhythm', 'pathology')):
    from titan_v4.reproducibility.pretrained_transfer.execution import file_sha
    records = {}
    for endpoint in endpoints:
        checkpoints = []
        for seed in (20261014, 20261015, 20261016):
            path = bundle / (endpoint + str(seed) + '.pth'); path.write_bytes(str(seed).encode())
            checkpoints.append({'seed': seed, 'path': str(path), 'sha256': file_sha(path)})
        records[endpoint] = {'eligible': True, 'contract': 'canonical_unfiltered', 'checkpoints': checkpoints,
                             'ensemble': {'thresholds': [.5] * 10} if endpoint == 'pathology' else {}}
    return {'scope': 'validation', 'seeds': [20261014, 20261015, 20261016],
            'selection_uses_tracking': False, 'eligible': records}


def cache(tmp_path, endpoints=('rhythm', 'pathology')):
    predictions = {e: np.arange(2 * (9 if e == 'rhythm' else 10), dtype=np.float32).reshape(2, -1) / 20
                   for e in endpoints}
    result = api().write_target_cache(tmp_path / 'teacher-targets', rows(), predictions, qualification(tmp_path, endpoints))
    return result, predictions


def test_shuffled_records_are_joined_by_key(tmp_path):
    path, predictions = cache(tmp_path)
    result = api().join_teacher_targets(list(reversed(rows())), path)
    assert result['record_keys'].tolist() == [r.key for r in reversed(rows())]
    assert np.array_equal(result['rhythm_teacher_logits'], predictions['rhythm'][::-1])


@pytest.mark.parametrize('kind', ['duplicate', 'missing', 'extra', 'tracking', 'labels', 'mask', 'signal', 'tensor'])
def test_join_rejects_wrong_population_or_reference(tmp_path, kind):
    path, _ = cache(tmp_path)
    requested = rows()
    if kind == 'duplicate': requested[1] = requested[0]
    if kind == 'missing': requested = requested[:1]
    if kind == 'extra': requested.append(replace(requested[0], key='ptb-xl/1.0.3/3'))
    if kind == 'tracking': requested[0] = replace(requested[0], role='tracking')
    if kind == 'labels': requested[0] = replace(requested[0], rhythm_label=8)
    if kind == 'mask': requested[0] = replace(requested[0], path_mask=(False,) * 10)
    if kind == 'signal': requested[0] = replace(requested[0], signal_sha256='f' * 64)
    if kind == 'tensor': requested[0] = replace(requested[0], student_tensor_sha256='f' * 64)
    with pytest.raises(ValueError):
        api().join_teacher_targets(requested, path)


def test_teacher_checkpoint_hash_is_verified_at_join(tmp_path):
    path, _ = cache(tmp_path)
    (tmp_path / 'rhythm20261014.pth').write_bytes(b'changed')
    with pytest.raises(ValueError, match='checkpoint'):
        api().join_teacher_targets(rows(), path)


def test_pathology_probabilities_are_independent_sigmoids(tmp_path):
    q = qualification(tmp_path, ('pathology',))
    path = api().write_target_cache(tmp_path / 'targets', rows(), {'pathology': np.zeros((2, 10), np.float32)}, q)
    with np.load(path / 'targets.npz', allow_pickle=False) as data:
        assert np.array_equal(data['path_teacher_probabilities'], np.full((2, 10), .5))
        assert np.all(data['path_teacher_probabilities'].sum(1) == 5)


@pytest.mark.parametrize('kind', ['nonfinite', 'shape', 'unqualified', 'seed', 'thresholds'])
def test_target_writer_rejects_invalid_predictions_or_qualification(tmp_path, kind):
    q = qualification(tmp_path)
    predicted = {'rhythm': np.zeros((2, 9), np.float32), 'pathology': np.zeros((2, 10), np.float32)}
    if kind == 'nonfinite': predicted['rhythm'][0, 0] = np.nan
    if kind == 'shape': predicted['rhythm'] = predicted['rhythm'][:1]
    if kind == 'unqualified': q['eligible']['rhythm']['eligible'] = False
    if kind == 'seed': q['eligible']['rhythm']['checkpoints'].pop()
    if kind == 'thresholds': q['eligible']['pathology']['ensemble']['thresholds'][0] = 2.
    with pytest.raises(ValueError):
        api().write_target_cache(tmp_path / 'targets', rows(), predicted, q)


def test_transfer_masks_unknown_and_disagreeing_labels():
    student = {'rhythm': torch.zeros(3, 9, requires_grad=True),
               'pathology': torch.zeros(3, 10, requires_grad=True)}
    rhythm = torch.zeros(3, 9); rhythm[0, 0] = 3; rhythm[1, 2] = 3
    path = torch.full((3, 10), 2.); path[0, 0] = -2
    mask = torch.zeros(3, 10); mask[:2, 0] = 1
    target = {'rhythm_labels': torch.tensor([0, 1, -1]), 'path_targets': torch.zeros(3, 10),
              'path_mask': mask, 'rhythm_teacher_logits': rhythm, 'path_teacher_logits': path,
              'path_thresholds': torch.full((10,), .5)}
    result = api().transfer_losses(student, target)
    (result['rhythm_loss'] + result['pathology_loss']).backward()
    assert result['rhythm_support'] == 1 and result['pathology_support'] == 1
    assert student['rhythm'].grad[0].abs().sum() > 0
    assert student['rhythm'].grad[1:].abs().sum() == 0
    assert student['pathology'].grad[0, 0].abs() > 0
    assert student['pathology'].grad[mask == 0].abs().sum() == 0
    assert student['pathology'].grad[1:].abs().sum() == 0


def test_absent_endpoints_have_differentiable_zero_not_fabricated_targets():
    student = {'rhythm': torch.zeros(2, 9, requires_grad=True),
               'pathology': torch.zeros(2, 10, requires_grad=True)}
    result = api().transfer_losses(student, {'rhythm_teacher_logits': None, 'path_teacher_logits': None})
    (result['rhythm_loss'] + result['pathology_loss']).backward()
    assert result['rhythm_support'] == result['pathology_support'] == 0
    assert student['rhythm'].grad.abs().sum() == student['pathology'].grad.abs().sum() == 0


def test_real_target_export_requires_slurm(tmp_path, monkeypatch):
    monkeypatch.delenv('SLURM_JOB_ID', raising=False)
    with pytest.raises(RuntimeError, match='Slurm'):
        api().export_teacher_targets(tmp_path, {})


def test_noninteger_rhythm_references_cannot_be_silently_cast():
    student = {'rhythm': torch.zeros(2, 9, requires_grad=True),
               'pathology': torch.zeros(2, 10, requires_grad=True)}
    with pytest.raises(ValueError):
        api().transfer_losses(student, {'rhythm_teacher_logits': np.zeros((2, 9)),
                                       'rhythm_labels': np.array([.5, 1.])})


def test_cache_records_cannot_silently_cast_fractional_references(tmp_path):
    records = rows(); records[0] = replace(records[0], rhythm_label=.5)
    with pytest.raises(ValueError):
        api().write_target_cache(tmp_path / 'targets', records,
                                  {'rhythm': np.zeros((2, 9))}, qualification(tmp_path, ('rhythm',)))


def test_probability_outside_unit_interval_is_invalid_even_near_zero(tmp_path):
    import json
    from titan_v4.reproducibility.pretrained_transfer.execution import file_sha
    path = api().write_target_cache(tmp_path / 'targets', rows(),
        {'pathology': np.full((2, 10), -100, np.float32)}, qualification(tmp_path, ('pathology',)))
    with np.load(path / 'targets.npz', allow_pickle=False) as archive:
        arrays = {k: archive[k] for k in archive.files}
    arrays['path_teacher_probabilities'][0, 0] = -1e-8
    np.savez_compressed(path / 'targets.npz', **arrays)
    (path / 'SHA256SUMS.json').write_text(json.dumps({p.name: file_sha(p) for p in path.iterdir()
                                                    if p.name != 'SHA256SUMS.json'}))
    with pytest.raises(ValueError, match='probability'):
        api().join_teacher_targets(rows(), path)


def test_pathology_agreement_preserves_frozen_float64_threshold():
    student = {'rhythm': torch.zeros(1, 9, requires_grad=True),
               'pathology': torch.zeros(1, 10, requires_grad=True)}
    logits = torch.full((1, 10), .1)
    probability = logits.sigmoid()
    threshold = np.full(10, float(probability[0, 0]) + 1e-9, np.float64)
    mask = torch.zeros(1, 10); mask[0, 0] = 1
    target = {'path_teacher_logits': logits, 'path_teacher_probabilities': probability,
              'path_targets': torch.ones(1, 10), 'path_mask': mask, 'path_thresholds': threshold}
    result = api().transfer_losses(student, target)
    assert result['pathology_support'] == 0


def test_cache_records_preprocessing_fingerprint(tmp_path):
    import json
    path, _ = cache(tmp_path)
    metadata = json.loads((path / 'metadata.json').read_text())
    assert len(metadata['teachers']['rhythm']['contract_sha256']) == 64


def test_cache_binds_the_full_qualification_receipt(tmp_path):
    import json
    from titan_v4.reproducibility.pretrained_transfer.execution import file_sha
    path, _ = cache(tmp_path)
    metadata = json.loads((path / 'metadata.json').read_text())
    assert metadata['qualification_sha256'] == file_sha(path / 'qualification.json')
