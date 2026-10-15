"""Standard-library execution receipts, shared by the client and allocation worker."""
from collections import Counter
from contextlib import contextmanager
import hashlib
import json
import re
from pathlib import Path, PurePosixPath, PureWindowsPath
import subprocess
import uuid

MANIFEST_SHA = 'b27acd413edf89bda192c7a03b38bd6ba9e6c799da34e490dab832bd25c4e3b5'
CONFIG_SHA = '609d7a3e0bc4bcd68133f0a7449fb5e81f839dbe5b70f8623123e1cfba6f3009'
BUCKETS = {'bootstrap': 'preparation', 'pilot': 'preparation', 'features': 'preparation',
           'teachers': 'teachers', 'students': 'students', 'replay': 'students'}
CEILINGS = {'preparation': 4.5, 'teachers': 21., 'students': 22.5}
IDENTITY_FIELDS = ('stage', 'attempt', 'job_id', 'code_sha256', 'paired_manifest_sha256')


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def safe_relative(name):
    posix, windows = PurePosixPath(name), PureWindowsPath(name)
    if (not posix.parts or str(posix) != name or posix.is_absolute() or windows.drive or windows.root
            or '..' in posix.parts or '\\' in name or ':' in name):
        raise ValueError('Unsafe relative archive/input path')
    return posix


def verify_inventory(root, manifest, *, exact=False, seal_name=None):
    root = Path(root).resolve()
    if not manifest:
        raise ValueError('Empty input inventory')
    for name, expected in manifest.items():
        path = root / safe_relative(name)
        if not path.resolve().is_relative_to(root) or path.is_symlink() or not path.is_file():
            raise ValueError('Missing or unsafe stage input file')
        if file_sha(path) != expected:
            raise ValueError('Stage input hash differs')
    if exact:
        actual = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
        if actual != set(manifest) | ({seal_name} if seal_name else set()):
            raise ValueError('Result inventory contains unsealed or missing files')


def verify_frozen_inputs(bundle, configuration):
    bundle = Path(bundle)
    manifest = bundle / 'inputs/paired_records.json'
    if file_sha(manifest) != MANIFEST_SHA or file_sha(configuration) != CONFIG_SHA:
        raise ValueError('Input identity differs from frozen protocol')
    receipt = json.loads((bundle / 'inputs/input_receipt.json').read_text())
    if (receipt.get('paired_manifest_sha256') != MANIFEST_SHA
            or receipt.get('configuration_sha256') != CONFIG_SHA
            or receipt.get('all_source_bytes_and_student_tensors_verified') is not True):
        raise ValueError('Input receipt differs from frozen protocol')
    rows = json.loads(manifest.read_text())
    config = json.loads(Path(configuration).read_text())
    if dict(Counter(r['role'] for r in rows)) != config['expected_role_counts']:
        raise ValueError('Frozen role counts differ')
    train = [r for r in rows if r['role'] == 'train']
    excluded = [r for r in rows if r['role'] not in ('train', 'train_excluded')]
    for field in ('key', 'patient_token', 'signal_sha256', 'student_tensor_sha256'):
        fit = [str(r[field]).lower() for r in train if r.get(field)]
        other = {str(r[field]).lower() for r in excluded if r.get(field)}
        if (field != 'patient_token' and len(fit) != len(set(fit))) or set(fit) & other:
            raise ValueError('Frozen fitting identity overlap')
    return rows


def verify_prepared_stage(stage, expected):
    stage = Path(stage)
    seal = stage / 'INPUT_SHA256.json'
    if file_sha(seal) != expected:
        raise ValueError('Stage differs from immutable preparation identity')
    verify_inventory(stage, json.loads(seal.read_text()))


def verify_result(root, expected):
    root = Path(root)
    seal = root / 'SHA256SUMS.json'
    if not seal.is_file():
        raise ValueError('Results must be sealed')
    manifest = json.loads(seal.read_text())
    if 'status.json' not in manifest:
        raise ValueError('Incomplete result seal')
    verify_inventory(root, manifest, exact=True, seal_name='SHA256SUMS.json')
    status = json.loads((root / 'status.json').read_text())
    if status.get('completed') is not True or any(
            field not in expected or status.get(field) != expected[field] for field in IDENTITY_FIELDS):
        raise ValueError('Result stage/submission/input identity differs')
    return {'stage': expected['stage'], 'verified_files': len(manifest), 'status': status}


def terminal(state):
    return state in {'COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY',
                     'NODE_FAIL', 'PREEMPTED', 'BOOT_FAIL', 'DEADLINE', 'REVOKED'}


def _write_json(path, value):
    temporary = path.with_suffix('.tmp-' + uuid.uuid4().hex)
    temporary.write_text(json.dumps(value, sort_keys=True) + '\n')
    temporary.replace(path)


def reserve_submission(stage_root, stage, attempt, hours, identity):
    if stage not in BUCKETS or not 1 <= attempt <= 9:
        raise ValueError('Invalid submission identity')
    receipt = dict(identity, stage=stage, attempt=attempt, requested_hours=hours,
                   job_name='tf-' + uuid.uuid4().hex[:24])
    with (Path(stage_root) / 'SUBMISSION.json').open('x') as stream:
        json.dump(receipt, stream)
    return receipt


def save_submission(stage_root, receipt, job_id):
    receipt = dict(receipt, job_id=int(job_id))
    _write_json(Path(stage_root) / 'SUBMISSION.json', receipt)
    return receipt


def _run(command):
    return subprocess.check_output(command, text=True, timeout=45)


def registry_usage(root, run=_run):
    result = {'total': 0., 'buckets': {}, 'active': [], 'jobs': {}}
    seen = set()
    stages = Path(root) / 'stages'
    paths = sorted(stages.glob('*/SUBMISSION.json'))
    for stage in stages.iterdir() if stages.exists() else []:
        logged = {int(m.group(1)) for p in stage.glob('slurm-*')
                  if (m := re.fullmatch(r'slurm-(\d+)\.(out|err)', p.name))}
        receipt = stage / 'SUBMISSION.json'
        registered = json.loads(receipt.read_text()).get('job_id') if receipt.exists() else None
        if logged and not receipt.exists():
            raise ValueError('Unmigrated legacy allocation; accounting completeness required')
        if registered is not None and logged - {int(registered)}:
            raise ValueError('Unregistered legacy job in stage logs')
    for path in paths:
        receipt = json.loads(path.read_text())
        if receipt.get('stage') not in BUCKETS:
            raise ValueError('Invalid registry stage')
        selector = ['-j', str(receipt['job_id'])] if 'job_id' in receipt else ['--name=' + receipt['job_name']]
        text = run(['sacct', *selector, '--starttime=2026-10-14', '--noheader', '--parsable2',
                    '--format=JobIDRaw,State,ExitCode,ElapsedRaw'])
        jobs = {}
        for line in text.splitlines():
            fields = line.split('|')
            if len(fields) >= 4 and fields[0].isdigit():
                jobs[int(fields[0])] = {'state': fields[1].split()[0].rstrip('+'),
                                       'exit_code': fields[2], 'seconds': int(fields[3])}
        if len(jobs) != 1:
            raise ValueError('Missing/unresolved submission accounting; retry is blocked')
        job_id, job = next(iter(jobs.items()))
        logged = {int(m.group(1)) for p in path.parent.glob('slurm-*')
                  if (m := re.fullmatch(r'slurm-(\d+)\.(out|err)', p.name))}
        if logged - {job_id}:
            raise ValueError('Unregistered legacy job after reservation reconciliation')
        if 'job_id' not in receipt:
            save_submission(path.parent, receipt, job_id)
        if job_id in seen or ('job_id' in receipt and job_id != int(receipt['job_id'])):
            raise ValueError('Duplicate or mismatched submission accounting')
        seen.add(job_id); result['jobs'][str(job_id)] = job
        if not terminal(job['state']):
            result['active'].append(job_id)
        hours = job['seconds'] / 3600
        result['total'] += hours
        bucket = BUCKETS[receipt['stage']]
        result['buckets'][bucket] = result['buckets'].get(bucket, 0) + hours
    return result


def worker_receipt(root, stage_root, job_id, job_name, run=_run):
    path = Path(stage_root) / 'SUBMISSION.json'
    receipt = json.loads(path.read_text())
    if 'job_id' not in receipt:
        if receipt.get('job_name') != job_name:
            raise ValueError('Worker reservation name differs')
        registry_usage(root, run)
        receipt = json.loads(path.read_text())
    if receipt.get('job_id') != job_id:
        raise ValueError('Worker job identity differs')
    return receipt


@contextmanager
def allocation_lock(root):
    import fcntl
    with (Path(root) / 'SUBMISSIONS.lock').open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)
