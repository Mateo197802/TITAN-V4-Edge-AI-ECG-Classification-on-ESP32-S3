"""Stage immutable teacher-transfer snapshots and audit bounded CEDIA allocations."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import re
import shlex
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
from manage_joint_cedia_run import rpc, SSH
from titan_v4.reproducibility.refinement import sha256
from titan_v4.reproducibility.pretrained_transfer.execution import (
    safe_relative, verify_inventory, verify_frozen_inputs, verify_result, MANIFEST_SHA)

STAGES = ('bootstrap', 'pilot', 'features', 'teachers', 'students', 'replay')
IMPLEMENTED_STAGES = ('bootstrap', 'pilot')
BUCKET = {s: 'preparation' if s in STAGES[:3] else 'teachers' if s == 'teachers' else 'students' for s in STAGES}
LIMITS = {'preparation': 4.5, 'teachers': 21., 'students': 22.5}
RUN_NAME = 'TITAN-V4-Founder-Transfer-2026-10-14'


def validate_run_name(name):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,95}', name) or '..' in name:
        raise ValueError('Unsafe run name')
    return name


def stage_key(stage, attempt=1):
    if stage not in STAGES or not isinstance(attempt, int) or not 1 <= attempt <= 9:
        raise ValueError('Invalid stage attempt')
    return stage if attempt == 1 else stage + '-a' + str(attempt)


def validate_archive(blob):
    names = set()
    with tarfile.open(fileobj=io.BytesIO(blob), mode='r:gz') as archive:
        for item in archive.getmembers():
            path = safe_relative(item.name)
            if (not item.isfile() or path.is_absolute() or '..' in path.parts or not path.parts
                    or '\\' in item.name or item.name in names):
                raise ValueError('Unsafe archive member')
            names.add(item.name)
    return names


def verify_files(root, manifest):
    verify_inventory(root, manifest)


def assert_submission_allowed(state_dir, stage, usage, request_hours, *, attempt=1):
    if stage not in STAGES or not math.isfinite(request_hours) or request_hours <= 0:
        raise ValueError('Invalid stage/resource request')
    if (Path(state_dir) / (stage_key(stage, attempt) + '.json')).exists():
        raise FileExistsError('Preserve immutable stage submission')
    if usage['active'] or usage['total'] + request_hours > 48:
        raise ValueError('Active allocation or total GPU budget prevents submission')
    bucket = BUCKET[stage]
    if usage['buckets'].get(bucket, 0) + request_hours > LIMITS[bucket]:
        raise ValueError('Stage GPU bucket exhausted')


def fetch_run(state):
    root = Path(state['local_results'])
    seal = root / 'SHA256SUMS.json'
    if not seal.is_file():
        raise ValueError('Results must be sealed')
    if state['job_state'] != 'COMPLETED' or state['job_exit_code'] != '0:0':
        raise ValueError('Slurm job is not successfully complete')
    return verify_result(root, state)


def _remote_helpers():
    return (ROOT / 'src/titan_v4/reproducibility/pretrained_transfer/execution.py').read_text()


def _accounting(state_dir):
    documents = [json.loads(p.read_text()) for p in Path(state_dir).glob('*.json')]
    records = [r for r in documents if r.get('stage') in STAGES and 'job_id' in r]
    script = _remote_helpers() + f'''
root=(Path.home()/'V4_CEDIA_LINK').resolve(strict=True)/'99_RESEARCH_RUNS'/{RUN_NAME!r}
if root.exists():
 with allocation_lock(root):
  for r in {records!r}:
   stage=root/'stages'/({{True:r['stage'],False:r['stage']+'-a'+str(r.get('attempt',1))}}[r.get('attempt',1)==1])
   if str(stage)!=r['stage_root']:raise ValueError('Foreign legacy submission')
   receipt=stage/'SUBMISSION.json'
   if not receipt.exists():
    with receipt.open('x') as stream:json.dump(r,stream)
   elif json.loads(receipt.read_text()).get('job_id')!=r['job_id']:raise ValueError('Submission identity differs')
  result=registry_usage(root)
else:result={{'total':0.,'buckets':{{}},'active':[],'jobs':{{}}}}
print(json.dumps(result))
'''
    return json.loads(rpc(script))


def _snapshot(bundle, stage):
    files = {p.relative_to(ROOT).as_posix(): p for p in (ROOT / 'src').rglob('*')
             if p.is_file() and p.suffix == '.py'}
    for rel in ('scripts/run_founder_transfer.py', 'requirements/founder-transfer.txt',
                'configs/founder_transfer_2026-10-14.json',
                'src/titan_v4/reproducibility/pretrained_transfer/vendor/ecgfounder/LICENSE'):
        files[rel] = ROOT / rel
    manifest = {name: sha256(path) for name, path in files.items()}
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode='w:gz') as archive:
        for name, path in files.items():
            archive.add(path, arcname='code/' + name, recursive=False)
        raw = (json.dumps({'code/' + k: v for k, v in manifest.items()}, sort_keys=True) + '\n').encode()
        info = tarfile.TarInfo('INPUT_SHA256.json'); info.size = len(raw)
        archive.addfile(info, io.BytesIO(raw))
    validate_archive(buffer.getvalue())
    return buffer.getvalue()


def _prepare(args):
    stage = args.stage
    key = stage_key(stage, args.attempt)
    verify_frozen_inputs(args.bundle, ROOT / 'configs/founder_transfer_2026-10-14.json')
    payload = _snapshot(args.bundle, stage)
    script = f'''import pathlib,io,tarfile,json,hashlib,sys
p=(pathlib.Path.home()/'V4_CEDIA_LINK').resolve(strict=True)
root=p/'99_RESEARCH_RUNS'/{RUN_NAME!r}
if not root.resolve().is_relative_to(p):raise ValueError('Target escaped project')
if not root.exists():root.mkdir(mode=0o700)
marker=root/'STUDY.json'
if marker.exists() and json.loads(marker.read_text()).get('name')!={RUN_NAME!r}:raise ValueError('Foreign study root')
if not marker.exists():marker.write_text(json.dumps({{'name':{RUN_NAME!r}}}))
dest=root/'stages'/{key!r}
dest.mkdir(parents=True,exist_ok=False,mode=0o700)
with tarfile.open(fileobj=io.BytesIO(sys.stdin.buffer.read()),mode='r:gz') as a:
 names=set()
 for m in a.getmembers():
  path=pathlib.PurePosixPath(m.name)
  if not m.isfile() or path.is_absolute() or '..' in path.parts or '\\\\' in m.name or m.name in names:raise ValueError('Unsafe stage archive')
  names.add(m.name); target=dest/path
  if not target.resolve().is_relative_to(dest):raise ValueError('Stage path escaped')
  target.parent.mkdir(parents=True,exist_ok=True); target.write_bytes(a.extractfile(m).read())
manifest=json.loads((dest/'INPUT_SHA256.json').read_text())
for n,h in manifest.items():
 if hashlib.sha256((dest/n).read_bytes()).hexdigest()!=h:raise ValueError('Stage input hash differs')
print(json.dumps({{'remote_root':str(root),'stage_root':str(dest),'stage':{stage!r},'attempt':{args.attempt},'code_sha256':hashlib.sha256((dest/'INPUT_SHA256.json').read_bytes()).hexdigest(),'paired_manifest_sha256':{MANIFEST_SHA!r}}}))
'''
    state = json.loads(rpc(script, payload))
    args.state_dir.mkdir(parents=True, exist_ok=True)
    file = args.state_dir / ('prepared-' + key + '.json')
    if file.exists():
        raise FileExistsError('Preserve preparation receipt')
    file.write_text(json.dumps(state, indent=2) + '\n')
    return state


def submission_program(state, stage_name, minutes):
    if stage_name not in STAGES or not 0 < minutes <= 180:
        raise ValueError('Invalid submission stage/time')
    if stage_name not in IMPLEMENTED_STAGES:
        raise NotImplementedError('Stage implementation must be tested before allocating a GPU')
    root, stage_root = state['remote_root'], state['stage_root']
    script = _remote_helpers() + f'''
import pathlib,subprocess,json
root=pathlib.Path({root!r}); stage=pathlib.Path({stage_root!r})
if not stage.resolve().is_relative_to(root.resolve()):raise ValueError('Foreign stage')
identity={{'code_sha256':file_sha(stage/'INPUT_SHA256.json'),'paired_manifest_sha256':MANIFEST_SHA}}
verify_inventory(stage,json.loads((stage/'INPUT_SHA256.json').read_text()))
verify_frozen_inputs(root,stage/'code/configs/founder_transfer_2026-10-14.json')
if {stage_name!r} not in ('bootstrap','pilot'):
 pilot=root/'stages/pilot-a2/results'
 expected=json.loads((root/'stages/pilot-a2/SUBMISSION.json').read_text())
 verify_result(pilot,expected)
 if registry_usage(root)['jobs'][str(expected['job_id'])]['state']!='COMPLETED':raise ValueError('Pilot must complete before fitting')
if {stage_name!r}=='bootstrap':python=pathlib.Path.home()/'V4_CEDIA_LINK/.venv_audit/bin/python'
else:
 env=json.loads((root/'ENVIRONMENT.json').read_text())['relative_env']
 if pathlib.Path(env).name!=env or not env.startswith('.venv_founder'):raise ValueError('Unsafe runtime pointer')
 python=root/env/'bin/python'
command=[str(python),'-u',str(stage/'code/scripts/run_founder_transfer.py'),'--bundle',str(root),'--stage',{stage_name!r},'--stage-root',str(stage),'--attempt',{str(state.get('attempt',1))!r}]
import shlex
job=chr(10).join(['#!/bin/bash','set -euo pipefail','export CUBLAS_WORKSPACE_CONFIG=:4096:8','export OMP_NUM_THREADS=4','export MPLCONFIGDIR='+shlex.quote(str(root/'mpl-cache')),'exec '+shlex.join(command)])+chr(10)
with allocation_lock(root):
 usage=registry_usage(root); bucket=BUCKETS[{stage_name!r}]; hours={minutes}/60
 if usage['active'] or usage['total']+hours>48 or usage['buckets'].get(bucket,0)+hours>CEILINGS[bucket]:raise ValueError('Allocation/budget guard')
 receipt=reserve_submission(stage,{stage_name!r},{state.get('attempt',1)},hours,identity)
 r=subprocess.run(['sbatch','--parsable','--partition=gpu-dev','--exclude=compute-0-1','--gres=gpu:1','--cpus-per-task=8','--mem=48G','--time={minutes}','--job-name='+receipt['job_name'],'--chdir='+str(stage),'--output='+str(stage/'slurm-%j.out'),'--error='+str(stage/'slurm-%j.err')],input=job,capture_output=True,text=True,timeout=45)
 if r.returncode:raise RuntimeError(r.stderr)
 receipt=save_submission(stage,receipt,int(r.stdout.strip().split(';')[0]))
print(json.dumps(receipt))
'''
    return script


def _submit(args):
    usage = _accounting(args.state_dir)
    assert_submission_allowed(args.state_dir, args.stage, usage, args.hours, attempt=args.attempt)
    key = stage_key(args.stage, args.attempt)
    state = json.loads((args.state_dir / ('prepared-' + key + '.json')).read_text())
    script = submission_program(state, args.stage, math.ceil(args.hours * 60))
    state.update(json.loads(rpc(script)))
    state.update({'stage': args.stage, 'requested_hours': args.hours})
    (args.state_dir / (key + '.json')).write_text(json.dumps(state, indent=2) + '\n')
    return {'stage': args.stage, 'job_id': state['job_id'], 'requested_hours': args.hours}


def asset_files(bundle, scope):
    if scope not in ('pilot', 'development', 'tracking'):
        raise ValueError('Unknown asset scope')
    from titan_v4.reproducibility.pretrained_transfer.contracts import PairedRecord
    from titan_v4.reproducibility.pretrained_transfer.inputs import _paths
    config = json.loads((ROOT / 'configs/founder_transfer_2026-10-14.json').read_text())
    rows = [PairedRecord(**r) for r in verify_frozen_inputs(bundle, ROOT / 'configs/founder_transfer_2026-10-14.json')]
    if scope == 'pilot':
        train = [r for r in rows if r.role == 'train']
        rows = [r for d in sorted({v.dataset for v in train}) for r in [v for v in train if v.dataset == d][:16]]
    elif scope == 'development':
        rows = [r for r in rows if r.role in ('train', 'rhythm_validation', 'pathology_validation')]
    else:
        rows = [r for r in rows if r.role in ('historical', 'tracking')]
    cache = ROOT / config['data_root']
    files = {'inputs/' + n: Path(bundle) / 'inputs' / n for n in ('paired_records.json', 'input_receipt.json', 'source_quality.json')}
    files['models/student_start.pth'] = ROOT / config['student_checkpoint']
    files['weights/12_lead_ECGFounder.pth'] = ROOT / 'data/cache/external_teachers/ecgfounder/12_lead_ECGFounder.pth'
    for row in rows:
        for file in _paths(row, cache):
            files['raw/' + file.relative_to(cache.resolve()).as_posix()] = file
    if sha256(files['models/student_start.pth']) != config['student_sha256'] or sha256(files['weights/12_lead_ECGFounder.pth']) != config['teacher_sha256']:
        raise ValueError('Pinned model asset hash differs')
    return files, rows


def _assets(args):
    files, rows = asset_files(args.bundle, args.asset_scope)
    archive_path = args.state_dir / ('assets-' + args.asset_scope + '.tar.gz')
    if archive_path.exists():
        raise FileExistsError('Preserve existing asset package')
    with tarfile.open(archive_path, 'w:gz') as archive:
        for name, path in files.items():
            archive.add(path, arcname=name, recursive=False)
        raw = json.dumps({name: sha256(path) for name, path in files.items()}, sort_keys=True).encode()
        info = tarfile.TarInfo('ASSET_SHA256.json'); info.size = len(raw)
        archive.addfile(info, io.BytesIO(raw))
    state = json.loads((args.state_dir / 'prepared-bootstrap.json').read_text())
    script = f'''import pathlib,io,tarfile,json,hashlib,sys
root=pathlib.Path({state['remote_root']!r})
if json.loads((root/'STUDY.json').read_text()).get('name')!={RUN_NAME!r}:raise ValueError('Foreign study')
incoming=root/'asset-incoming'/{args.asset_scope!r}; incoming.mkdir(parents=True,exist_ok=False)
with tarfile.open(fileobj=sys.stdin.buffer,mode='r|gz') as a:
 names=set()
 for m in a:
  p=pathlib.PurePosixPath(m.name)
  if not m.isfile() or p.is_absolute() or '..' in p.parts or m.name in names:raise ValueError('Unsafe asset archive')
  if p.parts[0] not in ('raw','inputs','models','weights','ASSET_SHA256.json'):raise ValueError('Unexpected asset namespace')
  names.add(m.name); dst=incoming/p; dst.parent.mkdir(parents=True,exist_ok=True); dst.write_bytes(a.extractfile(m).read())
manifest=json.loads((incoming/'ASSET_SHA256.json').read_text())
for name,h in manifest.items():
 src=incoming/name; dst=root/name
 if hashlib.sha256(src.read_bytes()).hexdigest()!=h:raise ValueError('Asset hash differs')
 if dst.exists():
  if hashlib.sha256(dst.read_bytes()).hexdigest()!=h:raise ValueError('Preserve different existing asset')
 else:
  dst.parent.mkdir(parents=True,exist_ok=True); src.rename(dst)
print(json.dumps({{'scope':{args.asset_scope!r},'verified_asset_files':len(manifest),'records_requested':{len(rows)}}}))
'''
    # Stream the large source archive; keep the shared SSH host-key policy.
    with archive_path.open('rb') as stream:
        response = subprocess.run(SSH + ['python3 -c ' + shlex.quote(script)], stdin=stream,
                                  capture_output=True, timeout=3600, check=True)
    return json.loads(response.stdout)


def _fetch(args, usage):
    key = stage_key(args.stage, args.attempt)
    state = json.loads((args.state_dir / (key + '.json')).read_text())
    job = usage['jobs'][str(state['job_id'])]
    if job['state'] != 'COMPLETED' or job['exit_code'] != '0:0':
        raise ValueError('Cannot retrieve an incomplete or failed job as successful evidence')
    local = args.state_dir / (key + '-results')
    if local.exists():
        raise FileExistsError('Preserve retrieved stage results')
    script = f'''import pathlib,io,tarfile,sys
p=pathlib.Path({state['stage_root']!r})/'results'
if not (p/'SHA256SUMS.json').is_file():raise ValueError('Unsealed results')
b=io.BytesIO()
with tarfile.open(fileobj=b,mode='w:gz') as a:
 for f in p.rglob('*'):
  if f.is_file():a.add(f,arcname=f.relative_to(p).as_posix(),recursive=False)
sys.stdout.buffer.write(b.getvalue())
'''
    blob = rpc(script); validate_archive(blob); local.mkdir()
    with tarfile.open(fileobj=io.BytesIO(blob), mode='r:gz') as archive:
        for item in archive.getmembers():
            target = local / safe_relative(item.name)
            if not target.resolve().is_relative_to(local.resolve()):
                raise ValueError('Result path escaped extraction root')
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.extractfile(item).read())
    return fetch_run({**state, 'local_results': str(local), 'stage': args.stage,
                      'job_state': job['state'], 'job_exit_code': job['exit_code']})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'submit', 'status', 'fetch', 'assets'))
    parser.add_argument('--state-dir', type=Path, required=True)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--stage', choices=STAGES, default='bootstrap')
    parser.add_argument('--hours', type=float, default=40 / 60)
    parser.add_argument('--asset-scope', choices=('pilot', 'development', 'tracking'), default='pilot')
    parser.add_argument('--attempt', type=int, default=1)
    args = parser.parse_args()
    validate_run_name(RUN_NAME)
    if args.action == 'assets':
        result = _assets(args)
    elif args.action == 'prepare':
        result = _prepare(args)
        result = {'prepared_stage': result['stage']}
    elif args.action == 'submit':
        result = _submit(args)
    elif args.action == 'status':
        result = _accounting(args.state_dir)
    else:
        result = _fetch(args, _accounting(args.state_dir))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
