"""Allocation-only worker for the pinned pretrained-transfer stages."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from titan_v4.reproducibility.pretrained_transfer.execution import (
    verify_frozen_inputs, verify_inventory, file_sha, IDENTITY_FIELDS, allocation_lock, worker_receipt)


def submission_identity(bundle, stage_root, stage, attempt):
    # A job can start before sbatch returns; share the submitter's receipt lock.
    with allocation_lock(bundle):
        receipt = worker_receipt(bundle, stage_root, int(os.environ['SLURM_JOB_ID']),
                                 os.environ.get('SLURM_JOB_NAME'))
    expected = dict(receipt, stage=stage, attempt=attempt, job_id=int(os.environ['SLURM_JOB_ID']),
                    code_sha256=file_sha(stage_root / 'INPUT_SHA256.json'))
    if any(receipt.get(k) != expected[k] for k in IDENTITY_FIELDS):
        raise ValueError('Worker submission identity differs')
    verify_inventory(stage_root, json.loads((stage_root / 'INPUT_SHA256.json').read_text()))
    verify_frozen_inputs(bundle, stage_root / 'code/configs/founder_transfer_2026-10-14.json')
    return {k: expected[k] for k in IDENTITY_FIELDS}


IDENTITY = {}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()


def seal(root, stage, data):
    root.mkdir(parents=True, exist_ok=False)
    if not IDENTITY or IDENTITY['stage'] != stage:
        raise ValueError('Cannot seal results without verified submission identity')
    (root / 'status.json').write_text(json.dumps({**data, **IDENTITY, 'completed': True}, indent=2) + '\n')
    manifest = {f.relative_to(root).as_posix(): digest(f) for f in root.rglob('*') if f.is_file()}
    (root / 'SHA256SUMS.json').write_text(json.dumps(manifest, indent=2) + '\n')


def bootstrap_commands(python, requirements, reports):
    common = [str(python), '-m', 'pip', 'install']
    return [common + ['--upgrade', 'pip==25.0.1'],
            common + ['torch==2.6.0+cu118', '--extra-index-url', 'https://download.pytorch.org/whl/cu118',
                      '--report', str(reports / 'torch.json')],
            common + ['-r', str(requirements), '--report', str(reports / 'runtime.json')]]


def bootstrap(bundle, stage_root, attempt=1):
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Bootstrap requires a Slurm allocation')
    env = bundle / ('.venv_founder' if attempt == 1 else '.venv_founder-a' + str(attempt))
    if env.exists():
        raise FileExistsError('Preserve the isolated environment')
    subprocess.run([sys.executable, '-m', 'venv', str(env)], check=True)
    python = env / 'bin/python'
    reports = stage_root / 'bootstrap-receipts'; reports.mkdir()
    environment = dict(os.environ, PIP_CACHE_DIR=str(bundle / 'pip-cache'))
    for command in bootstrap_commands(python, stage_root / 'code/requirements/founder-transfer.txt', reports):
        subprocess.run(command, env=environment, check=True)
    script = '''import torch,numpy,scipy,wfdb,pandas,sys,json
assert torch.__version__.startswith('2.6.0')
assert numpy.__version__=='1.26.4'
assert torch.cuda.is_available()
name=torch.cuda.get_device_name(0)
assert 'A100' in name
print(json.dumps({'python':sys.version.split()[0],'torch':torch.__version__,'numpy':numpy.__version__,'scipy':scipy.__version__,'wfdb':wfdb.__version__,'pandas':pandas.__version__,'cuda':torch.version.cuda,'gpu':name,'device_memory_bytes':torch.cuda.get_device_properties(0).total_memory}))
'''
    runtime = json.loads(subprocess.check_output([str(python), '-c', script], text=True))
    wheels = []
    for path in reports.glob('*.json'):
        for item in json.loads(path.read_text()).get('install', []):
            info = item.get('download_info', {}).get('archive_info', {})
            wheels.append({'name': item['metadata']['name'], 'version': item['metadata']['version'],
                           'sha256': info.get('hashes', {}).get('sha256')})
    pointer = bundle / 'ENVIRONMENT.json'
    if pointer.exists():
        raise FileExistsError('Preserve previously selected runtime')
    pointer.write_text(json.dumps({'relative_env': env.name, 'runtime': runtime, 'attempt': attempt}) + '\n')
    seal(stage_root / 'results', 'bootstrap', {'runtime': runtime, 'wheel_receipts': wheels, 'attempt': attempt,
                                             'audit_environment_modified': False})


def pilot(bundle, stage_root):
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Pilot requires a Slurm allocation')
    code = stage_root / 'code'; sys.path.insert(0, str(code / 'src'))
    import numpy as np
    import torch
    from titan_v4.reproducibility.pretrained_transfer.contracts import PairedRecord
    from titan_v4.reproducibility.pretrained_transfer.inputs import load_pair
    from titan_v4.reproducibility.pretrained_transfer.founder import load_founder, founder_features
    from titan_v4.models.titan_v4_net import TitanV4Lite
    config = json.loads((code / 'configs/founder_transfer_2026-10-14.json').read_text())
    all_rows = [PairedRecord(**r) for r in verify_frozen_inputs(bundle, code / 'configs/founder_transfer_2026-10-14.json')]
    train = [r for r in all_rows if r.role == 'train']
    selected = [r for dataset in sorted({r.dataset for r in train}) for r in [v for v in train if v.dataset == dataset][:16]]
    assert len(selected) == 64
    teacher_path = bundle / 'weights/12_lead_ECGFounder.pth'
    if digest(bundle / 'models/student_start.pth') != config['student_sha256']:
        raise ValueError('Student warm-start hash differs')
    torch.manual_seed(20261014); np.random.seed(20261014)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    device = torch.device('cuda'); torch.cuda.reset_peak_memory_stats()
    model = load_founder(teacher_path, outputs=150, seed=20261014).to(device).eval()
    model.requires_grad_(False)
    pairs = [load_pair(r, bundle / 'raw', 'canonical_unfiltered', missing_policy=config['missing_sample_policy']) for r in selected]
    x = torch.from_numpy(np.stack([p.teacher for p in pairs[:8]])).to(device)
    initial = x.clone()
    with torch.no_grad():
        logits, features = founder_features(model, x)
        assert logits.shape == (8, 150) and features.shape == (8, 1024)
    model.dense = torch.nn.Linear(1024, 9).to(device)
    adapted, _ = founder_features(model, x)
    loss = adapted.square().mean(); loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.dense.parameters())
    assert torch.equal(x, initial)
    student = TitanV4Lite(in_channels=6, num_rhythm=9, num_pathology=10).to(device).eval()
    checkpoint = torch.load(bundle / 'models/student_start.pth', map_location='cpu', weights_only=True)
    state = checkpoint.get('state_dict', checkpoint.get('model_state_dict', checkpoint))
    student.load_state_dict(state, strict=True)
    assert sum(p.numel() for p in student.parameters()) == 1143751
    sx = torch.from_numpy(np.stack([p.student for p in pairs[:8]])).to(device)
    with torch.no_grad():
        a, b = student(sx), student(sx)
    assert len(a) == 4 and all(torch.equal(u, v) for u, v in zip(a, b))
    peak = torch.cuda.max_memory_allocated()
    assert peak < 36 * 1024**3
    seal(stage_root / 'results', 'pilot', {'gpu': torch.cuda.get_device_name(0), 'peak_bytes': peak,
         'records_checked': 64, 'forward_backward_batch': 8, 'teacher_logits_shape': [8, 150],
         'adapted_head_gradient_finite': True, 'student_parameters': 1143751,
         'student_repeated_outputs_equal': True, 'input_tensor_mutated': False, 'optimizer_steps': 0})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--stage', required=True)
    parser.add_argument('--stage-root', type=Path, required=True)
    parser.add_argument('--attempt', type=int, default=1)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Execution requires a Slurm allocation')
    global IDENTITY
    IDENTITY = submission_identity(args.bundle, args.stage_root, args.stage, args.attempt)
    if args.stage == 'bootstrap':
        bootstrap(args.bundle, args.stage_root, args.attempt)
    elif args.stage == 'pilot':
        pilot(args.bundle, args.stage_root)
    else:
        raise NotImplementedError('This stage must have its tested implementation before submission')


if __name__ == '__main__':
    main()
