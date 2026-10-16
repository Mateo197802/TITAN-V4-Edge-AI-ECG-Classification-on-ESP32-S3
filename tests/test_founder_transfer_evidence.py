import importlib
import hashlib
import io
import json
import tarfile

import numpy as np
import pytest


def api():
    return importlib.import_module('titan_v4.reproducibility.pretrained_transfer.evidence')


def rows():
    from titan_v4.reproducibility.pretrained_transfer.contracts import PairedRecord
    return [PairedRecord('ptb-xl/1.0.3/1', 'patient', 'rhythm_validation', 'record', 'record.dat',
                         'a' * 64, 'b' * 64, 0, 5000, 0, (0.,) * 10, (True,) * 10, 'c' * 64)]


@pytest.mark.parametrize('kind', ['missing', 'extra', 'duplicate', 'label', 'patient', 'role'])
def test_prediction_evidence_requires_complete_original_references(kind):
    from titan_v4.reproducibility.pretrained_transfer.contracts import PRIMARY9_CLASSES
    data = {'keys': np.array([rows()[0].key]), 'labels': np.array([0]), 'mask': np.ones(1),
            'sources': np.array(['ptb-xl']), 'patients': np.array(['patient']),
            'roles': np.array(['rhythm_validation']), 'classes': np.asarray(PRIMARY9_CLASSES),
            'logits': np.zeros((1, 9))}
    if kind == 'missing': data['keys'] = np.array([])
    if kind == 'extra': data['keys'] = np.array(['different'])
    if kind == 'duplicate': data['keys'] = np.array([rows()[0].key] * 2)
    if kind == 'label': data['labels'][0] = 1
    if kind == 'patient': data['patients'] = np.array(['other'])
    if kind == 'role': data['roles'] = np.array(['tracking'])
    with pytest.raises(ValueError):
        api().assert_references(rows(), data, 'rhythm')


def test_raw_replay_accepts_only_bounded_probabilities_and_identical_decisions():
    value = np.zeros((2, 9), np.float32); value[:, 0] = 1
    result = api().assert_replay(value + 1e-7, value, endpoint='rhythm')
    assert result['decisions_equal']
    changed = value.copy(); changed[:, 1] += .01
    with pytest.raises(ValueError):
        api().assert_replay(changed, value, endpoint='rhythm')


def test_rhythm_logit_common_offset_is_not_a_probability_replay_error():
    expected = np.zeros((2, 9), np.float32); expected[:, 0] = 1
    result = api().assert_replay(expected + 2., expected, endpoint='rhythm')
    assert result['decisions_equal']
    assert result['maximum_absolute_probability_difference'] < 1e-6


def test_near_tie_cannot_hide_a_changed_class_inside_numerical_tolerance():
    expected = np.zeros((1, 9), np.float32)
    actual = expected.copy(); actual[0, 1] = 1e-7
    with pytest.raises(ValueError, match='decision'):
        api().assert_replay(actual, expected, endpoint='rhythm')


def test_small_logit_difference_cannot_hide_probability_tolerance_failure():
    expected = np.full((1, 9), -10., np.float64); expected[0, :2] = [10., 9.9]
    actual = expected.copy(); actual[0, 0] += .0001
    with pytest.raises(ValueError, match='probability'):
        api().assert_replay(actual, expected, endpoint='rhythm')


def test_pathology_replay_preserves_frozen_threshold_decisions():
    expected = np.zeros((1, 10), np.float32)
    actual = np.full((1, 10), -2e-7, np.float32)
    with pytest.raises(ValueError, match='decision'):
        api().assert_replay(actual, expected, endpoint='pathology', thresholds=np.full(10, .5))


def test_tracking_is_never_presented_as_an_untouched_test():
    with pytest.raises(ValueError, match='scope'):
        api().validate_evaluation_scope({'tracking': {'records': 2171, 'untouched_test': True}})


def test_incomplete_bundle_cannot_claim_review_readiness(tmp_path):
    with pytest.raises(ValueError, match='stage'):
        api().verify_founder_bundle(tmp_path, data_root=tmp_path, offline=True)


def test_report_comparison_cannot_tolerate_changed_selection_or_thresholds():
    with pytest.raises(ValueError):
        api().assert_report_equal({'selected_arm': 'one', 'seed': 20261014}, {'selected_arm': 'two', 'seed': 20261014})
    api().assert_report_equal({'macro_f1': .8 + 1e-14}, {'macro_f1': .8})


def test_source_replay_requires_student_selection_before_tracking(tmp_path, monkeypatch):
    monkeypatch.delenv('SLURM_JOB_ID', raising=False)
    with pytest.raises(RuntimeError, match='Slurm'):
        api().run_source_replay(tmp_path)


def test_cohort_packet_cannot_change_references_even_without_source_inference():
    records = rows()
    data = {'keys': np.asarray([records[0].key]), 'y': np.array([1]), 'original_y': np.array([-1]),
            'path_targets': np.zeros((1, 10)), 'path_mask': np.ones((1, 10)),
            'rhythm': np.zeros((1, 9)), 'pathology': np.zeros((1, 10)),
            'quality': np.zeros((1, 1)), 'biometrics': np.zeros((1, 3)), 'thresholds': np.full(10, .5)}
    with pytest.raises(ValueError, match='reference'):
        api().assert_cohort_references(records, data)


def test_patient_bootstrap_shares_draws_across_overlapping_endpoints():
    r = np.asarray(['a', 'shared']); p = np.asarray(['shared', 'c'])
    samples = list(api().shared_patient_weights(r, p, seed=20261014, replicates=5))
    again = list(api().shared_patient_weights(r, p, seed=20261014, replicates=5))
    assert len(samples) == 5
    for (rw, pw), (r2, p2) in zip(samples, again):
        assert rw[1] == pw[0]
        assert np.array_equal(rw, r2) and np.array_equal(pw, p2)


def test_registered_nonwinning_trials_cannot_be_omitted_from_packet():
    with pytest.raises(ValueError, match='coverage'):
        api().assert_trial_coverage([], [], adaptation={}, eligible={})


def test_historical_challenge_families_keep_their_pinned_source_cache():
    from titan_v4.reproducibility.pretrained_transfer.contracts import PairedRecord
    from titan_v4.reproducibility.pretrained_transfer.execution import verify_frozen_inputs
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    records = [PairedRecord(**r) for r in verify_frozen_inputs(
        root / 'outputs/reviewer_verification/founder_transfer_2026-10-14',
        root / 'configs/founder_transfer_2026-10-14.json')]
    historical = [r for r in records if r.role == 'historical']
    assert len(historical) == 672
    assert {api().validated_source_release(r) for r in historical} == {'challenge-2021'}


def test_historical_source_cannot_change_family_or_release():
    from dataclasses import replace
    record = replace(rows()[0], role='historical', key='ningbo/1.0.3/JS00012',
                     record_path='training/chapman_shaoxing/g1/JS00012', source_release='challenge-2021')
    with pytest.raises(ValueError): api().validated_source_release(record)


def test_replay_source_download_preserves_hash_checks_and_offline_mode(tmp_path, monkeypatch):
    from dataclasses import replace
    from titan_v4.reproducibility import refinement
    header, signal = b'pinned header', b'pinned signal'
    record = replace(rows()[0], record_path='records500/00000/00001_hr', signal_filename='00001_hr.dat',
                     header_sha256=hashlib.sha256(header).hexdigest(), signal_sha256=hashlib.sha256(signal).hexdigest())
    monkeypatch.setattr(refinement, '_http_bytes', lambda url: header if url.endswith('.hea') else signal)
    with pytest.raises(FileNotFoundError): api().ensure_source(record, tmp_path, offline=True)
    api().ensure_source(record, tmp_path, offline=False)
    api().ensure_source(record, tmp_path, offline=True)
    (tmp_path / 'ptb-xl/1.0.3/records500/00000/00001_hr.dat').write_bytes(b'tampered')
    with pytest.raises(ValueError, match='hash'): api().ensure_source(record, tmp_path, offline=False)


@pytest.mark.parametrize('change', ['active', 'missing_job', 'negative', 'bucket', 'total'])
def test_allocation_receipt_cannot_hide_failed_time_or_exceed_budget(change):
    receipt = {'active': [], 'total': 58 / 3600, 'buckets': {'preparation': 58 / 3600},
               'jobs': {'1': {'stage': 'bootstrap', 'state': 'FAILED', 'exit_code': '1:0', 'seconds': 58}}}
    if change == 'active': receipt['active'] = [1]
    if change == 'missing_job': receipt['jobs'] = {}
    if change == 'negative': receipt['jobs']['1']['seconds'] = -1
    if change == 'bucket': receipt['buckets']['preparation'] = 0
    if change == 'total':
        receipt['jobs']['1']['seconds'] = 5 * 3600
        receipt['total'] = receipt['buckets']['preparation'] = 5
    with pytest.raises(ValueError):
        api().verify_allocation_receipt(receipt)


def test_failed_allocations_are_part_of_actual_gpu_accounting():
    receipt = {'active': [], 'total': 58 / 3600, 'buckets': {'preparation': 58 / 3600},
               'jobs': {'1': {'stage': 'bootstrap', 'state': 'FAILED', 'exit_code': '1:0', 'seconds': 58}}}
    assert api().verify_allocation_receipt(receipt)['total_gpu_hours'] == 58 / 3600


@pytest.mark.parametrize('change', ['none', 'source', 'manifest', 'extra', 'escape'])
def test_execution_snapshot_binds_exact_code_inventory(change):
    source = b'print(1)\n'
    manifest = json.dumps({'code/worker.py': hashlib.sha256(source).hexdigest()}).encode()
    contents = {'code/worker.py': source, 'INPUT_SHA256.json': manifest}
    expected = hashlib.sha256(manifest).hexdigest()
    if change == 'source': contents['code/worker.py'] = b'print(2)\n'
    if change == 'manifest': expected = '0' * 64
    if change == 'extra': contents['unlisted.py'] = source
    if change == 'escape': contents['../outside'] = source
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode='w:gz') as archive:
        for name, raw in contents.items():
            info = tarfile.TarInfo(name); info.size = len(raw)
            archive.addfile(info, io.BytesIO(raw))
    if change == 'none':
        assert api().verify_stage_snapshot(buffer.getvalue(), expected)['files'] == 1
    else:
        with pytest.raises(ValueError): api().verify_stage_snapshot(buffer.getvalue(), expected)
