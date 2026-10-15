from __future__ import annotations

import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile

import pytest

ROOT = Path(__file__).resolve().parents[1]


def api():
    spec = importlib.util.spec_from_file_location('founder_manager', ROOT / 'scripts/manage_founder_transfer_cedia.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('name', ['../escape', '/absolute', '.', '..', 'bad name', 'x;touch'])
def test_unsafe_run_names_are_rejected(name):
    module = api()
    with pytest.raises(ValueError):
        module.validate_run_name(name)


@pytest.mark.parametrize('kind', ['traversal', 'absolute', 'symlink', 'hardlink', 'duplicate'])
def test_archive_members_cannot_escape_or_shadow(kind):
    module = api()
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode='w:gz') as archive:
        entry = tarfile.TarInfo('../bad' if kind == 'traversal' else '/bad' if kind == 'absolute' else 'code/file.py')
        if kind in ('symlink', 'hardlink'):
            entry.type = tarfile.SYMTYPE if kind == 'symlink' else tarfile.LNKTYPE
            entry.linkname = '/outside'
        archive.addfile(entry)
        if kind == 'duplicate':
            archive.addfile(entry)
    with pytest.raises(ValueError):
        module.validate_archive(buffer.getvalue())


def test_repeated_submission_is_rejected(tmp_path):
    module = api()
    (tmp_path / 'bootstrap.json').write_text('{}')
    with pytest.raises(FileExistsError):
        module.assert_submission_allowed(tmp_path, 'bootstrap', {'total': 0, 'buckets': {}, 'active': []}, .5)


@pytest.mark.parametrize('usage', [
    {'total': 47.8, 'buckets': {}, 'active': []},
    {'total': 0, 'buckets': {'preparation': 4.4}, 'active': []},
    {'total': 0, 'buckets': {}, 'active': [123]},
])
def test_budget_or_active_run_prevents_submission(tmp_path, usage):
    module = api()
    with pytest.raises(ValueError):
        module.assert_submission_allowed(tmp_path, 'bootstrap', usage, .5)


def test_missing_stage_input_is_rejected(tmp_path):
    module = api()
    manifest = {'missing.py': '0' * 64}
    with pytest.raises(ValueError, match='input'):
        module.verify_files(tmp_path, manifest)


def test_resealed_stage_cannot_replace_the_preparation_identity(tmp_path):
    code = tmp_path / 'source.py'; code.write_text('original')
    seal = tmp_path / 'INPUT_SHA256.json'
    seal.write_text(json.dumps({'source.py': hashlib.sha256(code.read_bytes()).hexdigest()}))
    expected = hashlib.sha256(seal.read_bytes()).hexdigest()
    code.write_text('changed')
    seal.write_text(json.dumps({'source.py': hashlib.sha256(code.read_bytes()).hexdigest()}))
    with pytest.raises(ValueError, match='prepar'):
        execution().verify_prepared_stage(tmp_path, expected)


def test_dependency_requires_a_successful_exact_sealed_producer(tmp_path):
    stage = tmp_path / 'stages/teachers'; stage.mkdir(parents=True)
    identity = {'stage': 'teachers', 'attempt': 1, 'job_id': 451,
                'code_sha256': 'a' * 64, 'paired_manifest_sha256': 'b' * 64}
    (stage / 'SUBMISSION.json').write_text(json.dumps(identity))
    with pytest.raises(ValueError):
        execution().successful_stage(tmp_path, 'teachers', run=lambda c: '451|COMPLETED|0:0|60\n')


def test_public_provenance_paths_are_relative_and_cannot_leak_foreign_roots(tmp_path):
    module = execution()
    assert module.public_paths({'path': str(tmp_path / 'models/x.pth')}, tmp_path) == {'path': 'models/x.pth'}
    with pytest.raises(ValueError):
        module.public_paths({'path': str(tmp_path.parent / 'private.pth')}, tmp_path)


def test_unsealed_result_is_not_success(tmp_path):
    module = api()
    with pytest.raises(ValueError, match='sealed'):
        module.fetch_run({'local_results': str(tmp_path), 'stage': 'pilot',
                          'job_state': 'COMPLETED', 'job_exit_code': '0:0'})


@pytest.mark.parametrize('job_state,code', [('FAILED', '1:0'), ('RUNNING', '0:0'), ('COMPLETED', '1:0')])
def test_nonterminal_or_failed_job_cannot_be_published(tmp_path, job_state, code):
    module = api()
    payload = json.dumps({'stage': 'pilot', 'completed': True}).encode()
    (tmp_path / 'status.json').write_bytes(payload)
    (tmp_path / 'SHA256SUMS.json').write_text(json.dumps({'status.json': hashlib.sha256(payload).hexdigest()}))
    with pytest.raises(ValueError, match='job'):
        module.fetch_run({'local_results': str(tmp_path), 'stage': 'pilot',
                          'job_state': job_state, 'job_exit_code': code})


def test_result_seal_binds_stage_and_file_bytes(tmp_path):
    module = api()
    identity = {'stage': 'pilot', 'job_id': 123, 'attempt': 1,
                'code_sha256': 'a' * 64, 'paired_manifest_sha256': 'b' * 64}
    payload = json.dumps(dict(identity, completed=True)).encode()
    (tmp_path / 'status.json').write_bytes(payload)
    (tmp_path / 'SHA256SUMS.json').write_text(json.dumps({'status.json': hashlib.sha256(payload).hexdigest()}))
    result = module.fetch_run({**identity, 'local_results': str(tmp_path), 'stage': 'pilot',
                               'job_state': 'COMPLETED', 'job_exit_code': '0:0'})
    assert result['verified_files'] == 1
    with pytest.raises(ValueError, match='stage'):
        module.fetch_run({**identity, 'local_results': str(tmp_path), 'stage': 'features',
                          'job_state': 'COMPLETED', 'job_exit_code': '0:0'})
    (tmp_path / 'status.json').write_text('tampered')
    with pytest.raises(ValueError, match='hash'):
        module.fetch_run({**identity, 'local_results': str(tmp_path), 'stage': 'pilot',
                          'job_state': 'COMPLETED', 'job_exit_code': '0:0'})


def test_pilot_asset_scope_is_training_only_and_pinned():
    module = api()
    bundle = ROOT / 'outputs/reviewer_verification/founder_transfer_2026-10-14'
    files, records = module.asset_files(bundle, 'pilot')
    assert len(records) == 64 and all(r.role == 'train' for r in records)
    assert len({r.dataset for r in records}) == 4
    assert 'models/student_start.pth' in files
    assert 'weights/12_lead_ECGFounder.pth' in files
    assert len([name for name in files if name.startswith('raw/')]) == 128


def test_unknown_asset_scope_is_rejected():
    module = api()
    with pytest.raises(ValueError, match='scope'):
        module.asset_files(ROOT, 'arbitrary')


def test_remote_submission_program_compiles_without_interpreting_path_text():
    module = api()
    script = module.submission_program({'remote_root': '/project/research/study',
                                         'stage_root': '/project/research/study/stages/bootstrap'},
                                        'bootstrap', 40)
    compile(script, '<submission>', 'exec')
    assert 'sbatch' in script


def test_bootstrap_installs_modern_resolver_before_cuda_runtime(tmp_path):
    spec = importlib.util.spec_from_file_location('founder_worker', ROOT / 'scripts/run_founder_transfer.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    commands = module.bootstrap_commands(tmp_path / 'python', tmp_path / 'runtime.txt', tmp_path)
    assert 'pip==25.0.1' in commands[0]
    assert 'torch==2.6.0+cu118' in commands[1]
    assert '--extra-index-url' in commands[1]
    assert '--index-url' not in commands[1]


@pytest.mark.parametrize('name', ['C:/escape.txt', 'C:escape.txt', '//host/share/x', 'a/../x', 'x:stream'])
def test_windows_archive_paths_fail_closed(name):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode='w:gz') as archive:
        archive.addfile(tarfile.TarInfo(name))
    with pytest.raises(ValueError):
        api().validate_archive(buffer.getvalue())


def execution():
    from titan_v4.reproducibility.pretrained_transfer import execution
    return execution


def test_frozen_manifest_rejects_changed_labels_even_with_updated_receipt(tmp_path):
    module = execution()
    inputs = tmp_path / 'inputs'; inputs.mkdir()
    (inputs / 'paired_records.json').write_text('[{"rhythm_label": 8}]')
    (inputs / 'input_receipt.json').write_text(json.dumps({
        'paired_manifest_sha256': hashlib.sha256((inputs / 'paired_records.json').read_bytes()).hexdigest()}))
    with pytest.raises(ValueError, match='frozen'):
        module.verify_frozen_inputs(tmp_path, ROOT / 'configs/founder_transfer_2026-10-14.json')


@pytest.mark.parametrize('state', ['SUSPENDED', 'REQUEUED', 'PREEMPTED_PENDING', 'UNKNOWN'])
def test_only_explicit_terminal_states_release_allocation(state):
    assert not execution().terminal(state)


def test_remote_registry_survives_lost_response_and_new_local_directory(tmp_path):
    module = execution()
    stage = tmp_path / 'stages/pilot'; stage.mkdir(parents=True)
    receipt = module.reserve_submission(stage, 'pilot', 1, .5,
                                        {'code_sha256': 'a' * 64, 'paired_manifest_sha256': 'b' * 64})
    assert json.loads((stage / 'SUBMISSION.json').read_text())['job_name'] == receipt['job_name']
    # Simulate accepted sbatch followed by a lost connection, before the job ID is saved.
    calls = []
    def accounting(command):
        calls.append(command)
        return '451|RUNNING|0:0|60\n'
    usage = module.registry_usage(tmp_path, accounting)
    assert usage['active'] == [451] and usage['total'] == 60 / 3600
    assert '--name=' + receipt['job_name'] in calls[0]
    assert json.loads((stage / 'SUBMISSION.json').read_text())['job_id'] == 451
    assert module.registry_usage(tmp_path, accounting)['active'] == [451]


def test_unresolved_submission_blocks_retry(tmp_path):
    module = execution()
    stage = tmp_path / 'stages/pilot'; stage.mkdir(parents=True)
    module.reserve_submission(stage, 'pilot', 1, .5, {})
    with pytest.raises(ValueError, match='unresolved'):
        module.registry_usage(tmp_path, lambda command: '')


def test_fresh_state_cannot_omit_legacy_allocation_logs(tmp_path):
    stage = tmp_path / 'stages/bootstrap'; stage.mkdir(parents=True)
    (stage / 'slurm-451.out').write_text('previous allocation')
    with pytest.raises(ValueError, match='legacy'):
        execution().registry_usage(tmp_path, lambda command: '')


def test_reservation_reconciliation_cannot_hide_another_logged_job(tmp_path):
    module = execution()
    stage = tmp_path / 'stages/pilot'; stage.mkdir(parents=True)
    module.reserve_submission(stage, 'pilot', 2, .5, {})
    for job in (451, 999):
        (stage / ('slurm-' + str(job) + '.out')).write_text('allocation')
    with pytest.raises(ValueError, match='legacy'):
        module.registry_usage(tmp_path, lambda command: '451|COMPLETED|0:0|39\n')


def test_worker_recovers_reserved_submission_after_lost_submit_response(tmp_path):
    module = execution()
    stage = tmp_path / 'stages/pilot'; stage.mkdir(parents=True)
    receipt = module.reserve_submission(stage, 'pilot', 2, .5, {})
    found = module.worker_receipt(tmp_path, stage, 451, receipt['job_name'],
                                  run=lambda command: '451|RUNNING|0:0|10\n')
    assert found['job_id'] == 451
    with pytest.raises(ValueError):
        module.worker_receipt(tmp_path, stage, 452, receipt['job_name'])


def test_unimplemented_stage_is_rejected_before_remote_allocation(monkeypatch):
    module = api()
    monkeypatch.setattr(module, 'IMPLEMENTED_STAGES', ('bootstrap', 'pilot'))
    with pytest.raises(NotImplementedError):
        module.submission_program({'remote_root': '/study', 'stage_root': '/study/stages/students'}, 'students', 40)


@pytest.mark.parametrize('change', ['attempt', 'job_id', 'code_sha256', 'extra'])
def test_result_seal_rejects_stale_submission_and_unlisted_files(tmp_path, change):
    module = execution()
    expected = {'stage': 'pilot', 'attempt': 2, 'job_id': 451,
                'code_sha256': 'a' * 64, 'paired_manifest_sha256': 'b' * 64}
    status = dict(expected, completed=True)
    if change != 'extra':
        status[change] = 'stale'
    raw = json.dumps(status).encode(); (tmp_path / 'status.json').write_bytes(raw)
    (tmp_path / 'SHA256SUMS.json').write_text(json.dumps({'status.json': hashlib.sha256(raw).hexdigest()}))
    if change == 'extra':
        (tmp_path / 'unexpected.csv').write_text('unsealed')
    with pytest.raises(ValueError):
        module.verify_result(tmp_path, expected)


def test_interrupted_trial_is_preserved_and_can_restart(tmp_path):
    output = tmp_path / 'fits' / 'trial'; output.mkdir(parents=True)
    (output / 'partial.pth').write_bytes(b'original partial evidence')
    assert execution().prepare_fit_trial(output, {'seed': 1}) is None
    assert not output.exists()
    archived = list((tmp_path / 'fits/.interrupted').glob('*/partial.pth'))
    assert len(archived) == 1 and archived[0].read_bytes() == b'original partial evidence'


def test_completed_trial_is_reused_only_when_sealed_identity_matches(tmp_path):
    output = tmp_path / 'trial'; output.mkdir()
    record = {'fit_identity': {'seed': 1}}
    (output / 'result.json').write_text(json.dumps(record))
    inventory = {'result.json': hashlib.sha256((output / 'result.json').read_bytes()).hexdigest()}
    (output / 'SHA256SUMS.json').write_text(json.dumps(inventory))
    assert execution().prepare_fit_trial(output, {'seed': 1}) == record
    with pytest.raises(ValueError, match='identity'):
        execution().prepare_fit_trial(output, {'seed': 2})
    (output / 'result.json').write_text('{}')
    with pytest.raises(ValueError): execution().prepare_fit_trial(output, {'seed': 1})
