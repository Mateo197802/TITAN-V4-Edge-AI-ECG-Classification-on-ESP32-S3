from __future__ import annotations

import dataclasses
import hashlib
import importlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
import wfdb

from titan_v4.reproducibility.physionet import preprocess_primary9_signal

ROOT = Path(__file__).resolve().parents[1]
LEADS = ('I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6')


def api():
    return importlib.import_module('titan_v4.reproducibility.pretrained_transfer.inputs')


@pytest.fixture
def waveform():
    t = np.arange(5000, dtype=np.float64) / 500
    return np.stack([np.sin(t * (i + 1)) * (i + 1) / 20 + i / 10 for i in range(12)], axis=1)


@pytest.fixture
def source(tmp_path, waveform):
    folder = tmp_path / 'ptb-xl' / '1.0.3' / 'records500'
    folder.mkdir(parents=True)
    wfdb.wrsamp('fixture', fs=500, units=['mV'] * 12, sig_name=list(LEADS),
                p_signal=waveform, fmt=['16'] * 12, write_dir=str(folder))
    raw = wfdb.rdrecord(str(folder / 'fixture'))
    expected = preprocess_primary9_signal(raw.p_signal, raw.sig_name, raw.fs)
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    return {
        'key': 'ptb-xl/1.0.3/fixture', 'patient_token': 'unit-patient', 'role': 'train',
        'record_path': 'records500/fixture', 'signal_filename': 'fixture.dat',
        'header_sha256': digest(folder / 'fixture.hea'),
        'signal_sha256': digest(folder / 'fixture.dat'),
        'window_start': 0, 'window_samples': 5000, 'rhythm_label': 3,
        'path_targets': (0.,) * 10, 'path_mask': (True,) * 10,
        'student_tensor_sha256': hashlib.sha256(expected.tobytes()).hexdigest(),
        '_expected': expected, '_cache': tmp_path, '_folder': folder,
    }


def record(source):
    cls = importlib.import_module('titan_v4.reproducibility.pretrained_transfer.contracts').PairedRecord
    return cls(**{k: v for k, v in source.items() if not k.startswith('_')})


def test_pair_shapes_and_original_student_bytes(source):
    module = api()
    pair = module.load_pair(record(source), source['_cache'], 'canonical_unfiltered')
    assert pair.teacher.shape == (12, 5000)
    assert pair.student.shape == (6, 1250)
    assert pair.teacher.dtype == pair.student.dtype == np.float32
    assert pair.student.tobytes() == source['_expected'].tobytes()
    assert pair.key == source['key']


@pytest.mark.parametrize('field', ['header_sha256', 'signal_sha256', 'student_tensor_sha256'])
def test_source_or_tensor_hash_mismatch_fails(source, field):
    module = api()
    r = dataclasses.replace(record(source), **{field: '0' * 64})
    with pytest.raises(ValueError, match='hash'):
        module.load_pair(r, source['_cache'], 'canonical_unfiltered')


def test_named_heed_order_and_global_normalization(waveform):
    module = api()
    indices = [0, 1, 2, 3, 5, 4, 6, 7, 8, 9, 10, 11]
    a, _ = module.preprocess_founder(waveform, LEADS, 500, ['mV'] * 12, 'heed_unfiltered')
    b, _ = module.preprocess_founder(waveform, LEADS, 500, ['mV'] * 12, 'canonical_unfiltered')
    np.testing.assert_array_equal(a, b[indices])
    expected = ((waveform - waveform.mean()) / (waveform.std() + 1e-8)).T.astype(np.float32)
    np.testing.assert_array_equal(b, expected)


def test_physical_unit_conversion_is_consistent(waveform):
    module = api()
    a, _ = module.preprocess_founder(waveform, LEADS, 500, ['mV'] * 12, 'canonical_unfiltered')
    b, _ = module.preprocess_founder(waveform * 1000, LEADS, 500, ['uV'] * 12, 'canonical_unfiltered')
    np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize('kind', ['missing', 'duplicate', 'units', 'frequency', 'nonfinite', 'contract'])
def test_invalid_source_contract_is_rejected(waveform, kind):
    module = api()
    names, units, fs, data, contract = list(LEADS), ['mV'] * 12, 500, waveform.copy(), 'canonical_unfiltered'
    if kind == 'missing':
        names[-1] = 'unknown'
    elif kind == 'duplicate':
        names[-1] = 'V5'
    elif kind == 'units':
        units[0] = 'arbitrary'
    elif kind == 'frequency':
        fs = 125
    elif kind == 'nonfinite':
        data[1, 0] = np.nan
    else:
        contract = 'not_registered'
    with pytest.raises(ValueError):
        module.preprocess_founder(data, names, fs, units, contract)


def test_short_and_constant_signals_are_flagged_not_silently_dropped():
    module = api()
    x, flags = module.preprocess_founder(np.zeros((3000, 12)), LEADS, 500, ['mV'] * 12, 'canonical_filtered')
    assert x.shape == (12, 5000)
    assert np.isfinite(x).all() and not x.any()
    assert {'right_padding', 'constant_signal'}.issubset(flags)


def test_filtered_contract_matches_registered_filter(waveform):
    from scipy.signal import butter, filtfilt, iirnotch, medfilt

    module = api()
    x = waveform.T.copy()
    b, a = iirnotch(50, 30, fs=500)
    x = filtfilt(b, a, x, axis=1)
    b, a = butter(4, [0.67, 40], btype='bandpass', fs=500)
    x = filtfilt(b, a, x, axis=1)
    x -= np.stack([medfilt(c, kernel_size=201) for c in x])
    expected = ((x - x.mean()) / (x.std() + 1e-8)).astype(np.float32)
    actual, _ = module.preprocess_founder(waveform, LEADS, 500, ['mV'] * 12, 'canonical_filtered')
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize('identity', ['key', 'patient_token', 'signal_sha256', 'student_tensor_sha256'])
def test_fitting_excludes_each_identity(source, identity):
    module = api()
    train = record(source)
    changed = dict(key='ptb-xl/1.0.3/other', patient_token='other-patient',
                   signal_sha256='a' * 64, student_tensor_sha256='b' * 64, role='validation')
    changed[identity] = getattr(train, identity)
    excluded = dataclasses.replace(train, **changed)
    with pytest.raises(ValueError, match='overlap'):
        module.assert_disjoint([train], [excluded])


def test_duplicates_and_later_windows_are_rejected(source):
    module = api()
    r = record(source)
    with pytest.raises(ValueError, match='duplicate'):
        module.assert_disjoint([r, r], [])
    with pytest.raises(ValueError, match='window'):
        module.assert_disjoint([dataclasses.replace(r, window_start=1)], [])


def test_overlapping_training_rows_are_preserved_but_not_fitted(source):
    module = api()
    r = record(source)
    history = dataclasses.replace(r, role='historical', key='ptb-xl/1.0.3/historical')
    kept, excluded = module.apply_fitting_exclusions([r], [history])
    assert kept == [] and len(excluded) == 1
    assert excluded[0].role == 'train_excluded'
    assert excluded[0].rhythm_label == r.rhythm_label
    assert excluded[0].path_targets == r.path_targets
    assert {'patient_token', 'signal_sha256', 'student_tensor_sha256'}.issubset(excluded[0].exclusion_reasons)


def test_record_paths_cannot_escape_cache(source):
    module = api()
    with pytest.raises(ValueError, match='path'):
        module.load_pair(dataclasses.replace(record(source), record_path='../outside'), source['_cache'], 'canonical_unfiltered')


@pytest.mark.parametrize('role', ['train', 'historical', 'tracking', 'train_excluded'])
def test_legacy_missing_sample_policy_does_not_enter_teacher_fitting(source, role):
    module = api()
    signal = source['_folder'] / 'fixture.dat'
    data = np.frombuffer(signal.read_bytes(), dtype='<i2').copy()
    data[12] = -32768
    signal.write_bytes(data.tobytes())
    raw = wfdb.rdrecord(str(source['_folder'] / 'fixture'))
    assert np.isnan(raw.p_signal).any()
    expected = preprocess_primary9_signal(raw.p_signal, raw.sig_name, raw.fs)
    r = dataclasses.replace(record(source), role=role,
                            signal_sha256=hashlib.sha256(signal.read_bytes()).hexdigest(),
                            student_tensor_sha256=hashlib.sha256(expected.tobytes()).hexdigest())
    if role == 'train':
        with pytest.raises(ValueError, match='nonfinite'):
            module.load_student_tensor(r, source['_cache'])
    else:
        actual, flags = module.load_student_tensor(r, source['_cache'])
        assert actual.tobytes() == expected.tobytes()
        assert 'legacy_nonfinite_zero_fill' in flags


def test_real_manifest_has_frozen_support_and_unknown_masks():
    module = api()
    config = json.loads((ROOT / 'configs/founder_transfer_2026-10-14.json').read_text())
    rows = module.build_pair_manifest(ROOT, config)
    train = [r for r in rows if r.role == 'train']
    removed = [r for r in rows if r.role == 'train_excluded']
    assert len(train) == 22540
    assert len(train) + len(removed) == 22567
    assert len(removed) == 27
    assert sum(any(r.path_mask) for r in train) == 16413
    assert len([r for r in rows if r.role == 'rhythm_validation']) == 1476
    assert len([r for r in rows if r.role == 'pathology_validation']) == 2027
    assert len([r for r in rows if r.role == 'tracking']) == 2171
    assert len([r for r in rows if r.role == 'historical']) == 672
    assert all(not any(r.path_mask) for r in train if not r.key.startswith('ptb-xl/'))
    module.assert_disjoint(train, [r for r in rows if r.role != 'train'])


def test_changed_reference_manifest_is_rejected_before_pairing(tmp_path):
    module = api()
    config = json.loads((ROOT / 'configs/founder_transfer_2026-10-14.json').read_text())
    original = config['manifests']['train'][0]
    payload = (ROOT / original).read_bytes()
    changed = tmp_path / 'changed_references.csv'
    changed.write_bytes(payload.replace(b'NSR', b'AFIB', 1))
    assert changed.read_bytes() != payload
    config['manifests']['train'][0] = str(changed)
    config.setdefault('manifest_sha256', {})[str(changed)] = hashlib.sha256(payload).hexdigest()
    with pytest.raises(ValueError, match='manifest hash'):
        module.build_pair_manifest(ROOT, config)


def test_archived_input_hash_mismatch_is_not_ignored(tmp_path):
    module = api()
    file = tmp_path / 'input.npz'
    file.write_bytes(b'wrong archived bytes')
    with pytest.raises(ValueError, match='archived input hash'):
        module.verify_archived_inputs(ROOT, {'archived_input_sha256': {str(file): '0' * 64}})


def test_explicit_wfdb16_repair_keeps_case_and_original_student_bytes(source):
    module = api()
    signal = source['_folder'] / 'fixture.dat'
    digital = np.frombuffer(signal.read_bytes(), dtype='<i2').copy()
    digital[12] = -32768
    signal.write_bytes(digital.tobytes())
    raw = wfdb.rdrecord(str(source['_folder'] / 'fixture'))
    expected = preprocess_primary9_signal(raw.p_signal, raw.sig_name, raw.fs)
    r = dataclasses.replace(record(source), signal_sha256=hashlib.sha256(signal.read_bytes()).hexdigest(),
                            student_tensor_sha256=hashlib.sha256(expected.tobytes()).hexdigest())
    with pytest.raises(ValueError, match='nonfinite'):
        module.load_pair(r, source['_cache'], 'canonical_unfiltered')
    pair = module.load_pair(r, source['_cache'], 'canonical_unfiltered', missing_policy='wfdb16_zero_fill')
    assert np.isfinite(pair.teacher).all()
    assert pair.student.tobytes() == expected.tobytes()
    assert pair.missing_samples == 1
    assert 'wfdb16_missing_zero_fill' in pair.quality_flags


@pytest.mark.parametrize('problem', ['infinity', 'wrong_sentinel', 'wrong_format'])
def test_verified_repair_rejects_unexplained_numerical_errors(problem):
    module = api()
    physical = np.zeros((5, 12), dtype=np.float64)
    physical[0, 0] = np.nan
    digital = np.zeros((5, 12), dtype=np.int16)
    digital[0, 0] = -32768
    fmt = ['16'] * 12
    if problem == 'infinity':
        physical[1, 0] = np.inf
    elif problem == 'wrong_sentinel':
        digital[0, 0] = 0
    else:
        fmt[0] = '24'
    with pytest.raises(ValueError):
        module.validated_wfdb16_zero_fill(physical, digital, fmt)


def test_parallel_verification_returns_only_ordered_receipts(source):
    module = api()
    r = record(source)
    progress = []
    result = module.verify_source_pairs([r, dataclasses.replace(r, role='rhythm_validation')],
                                       source['_cache'], 'reject', workers=2, progress=progress.append)
    assert [v['role'] for v in result] == ['train', 'rhythm_validation']
    assert all(v['verified'] is True for v in result)
    assert not any(isinstance(value, np.ndarray) for row in result for value in row.values())
    assert progress == [1, 2]


def test_sealed_input_bytes_survive_git_storage_filters():
    relative = 'outputs/reviewer_verification/founder_transfer_2026-10-14/inputs/paired_records.json'
    raw = subprocess.check_output(['git', 'hash-object', '--no-filters', relative], cwd=ROOT)
    stored = subprocess.check_output(['git', 'hash-object', '--path=' + relative, relative], cwd=ROOT)
    assert stored == raw, 'Git storage would change the sealed manifest bytes'
