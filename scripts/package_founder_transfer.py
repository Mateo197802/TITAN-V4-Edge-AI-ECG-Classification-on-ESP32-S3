"""Collect completed immutable stage packets, code snapshots and sanitized logs."""
import argparse
import io
import json
from pathlib import Path
import shutil
import sys
import tarfile
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))

from manage_founder_transfer_cedia import _accounting, _fetch, stage_key, validate_archive
from manage_joint_cedia_run import rpc
from titan_v4.reproducibility.pretrained_transfer.execution import file_sha, safe_relative, verify_result
from titan_v4.reproducibility.pretrained_transfer.evidence import verify_allocation_receipt, verify_stage_snapshot

REQUIRED = ('features', 'teachers', 'targets', 'students', 'replay')


def copy_exact(source, target):
    source, target = Path(source), Path(target)
    files = {p.relative_to(source).as_posix(): p for p in source.rglob('*') if p.is_file()}
    existing = {p.relative_to(target).as_posix(): p for p in target.rglob('*') if p.is_file()}
    if set(existing) - set(files) or any(file_sha(path) != file_sha(files[name]) for name, path in existing.items()):
        raise ValueError('Refuse to overwrite different published packet bytes')
    for name, path in files.items():
        destination = target / safe_relative(name)
        if not destination.resolve().is_relative_to(target.resolve()): raise ValueError('Unsafe publication target')
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists(): shutil.copyfile(path, destination)


def collect_execution_assets(states):
    # Only fixed stage code and logs are read; private training caches are not archived.
    program = '''import pathlib,io,tarfile,json,hashlib,sys
states=STATES
files={}
for state in states:
 root=pathlib.Path(state['remote_root']); stage=pathlib.Path(state['stage_root'])
 if not stage.resolve().is_relative_to((root/'stages').resolve()):raise ValueError('Foreign stage')
 raw=(stage/'INPUT_SHA256.json').read_bytes()
 if hashlib.sha256(raw).hexdigest()!=state['code_sha256']:raise ValueError('Changed stage identity')
 manifest=json.loads(raw); archive=io.BytesIO()
 with tarfile.open(fileobj=archive,mode='w:gz') as target:
  for name,expected in manifest.items():
   source=stage/name
   if not source.resolve().is_relative_to(stage.resolve()) or source.is_symlink():raise ValueError('Foreign code')
   if hashlib.sha256(source.read_bytes()).hexdigest()!=expected:raise ValueError('Changed source')
   target.add(source,arcname=name,recursive=False)
  info=tarfile.TarInfo('INPUT_SHA256.json');info.size=len(raw);target.addfile(info,io.BytesIO(raw))
 key=state['stage']+('' if state.get('attempt',1)==1 else '-a'+str(state['attempt']))
 files['snapshots/'+key+'-code.tar.gz']=archive.getvalue()
 receipts={}
 for suffix,label in (('out','stdout'),('err','stderr')):
  source=stage/('slurm-'+str(state['job_id'])+'.'+suffix)
  original=source.read_bytes();text=original.decode('utf-8',errors='replace')
  for path,token in ((root,'<STUDY>'),(root.parent.parent,'<PROJECT>'),(pathlib.Path.home(),'<HOME>')):
   text=text.replace(str(path),token)
  published=text.encode('utf-8');name='logs/'+key+'/'+label+'.txt';files[name]=published
  receipts[label]={'original_sha256':hashlib.sha256(original).hexdigest(),'published_sha256':hashlib.sha256(published).hexdigest(),'transformation':'UTF-8 and private path replacement'}
 files['logs/'+key+'/receipt.json']=(json.dumps({'job_id':state['job_id'],'stage':state['stage'],'attempt':state.get('attempt',1),'streams':receipts},sort_keys=True,indent=2)+'\\n').encode()
outer=io.BytesIO()
with tarfile.open(fileobj=outer,mode='w:gz') as archive:
 for name,raw in files.items():
  info=tarfile.TarInfo(name);info.size=len(raw);archive.addfile(info,io.BytesIO(raw))
sys.stdout.buffer.write(outer.getvalue())
'''.replace('states=STATES', 'states=' + repr(states))
    blob = rpc(program); validate_archive(blob)
    files = {}
    with tarfile.open(fileobj=io.BytesIO(blob), mode='r:gz') as archive:
        for member in archive.getmembers(): files[member.name] = archive.extractfile(member).read()
    for state in states:
        key = stage_key(state['stage'], state.get('attempt', 1))
        verify_stage_snapshot(files['snapshots/' + key + '-code.tar.gz'], state['code_sha256'])
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state-dir', type=Path, required=True)
    parser.add_argument('--bundle', type=Path, required=True)
    args = parser.parse_args()
    states = []
    for path in sorted(args.state_dir.glob('*.json')):
        state = json.loads(path.read_text())
        if 'job_id' in state and state.get('stage') and not path.name.startswith('prepared-'): states.append(state)
    accounting = _accounting(args.state_dir)
    by_job = {str(s['job_id']): s for s in states}
    if set(accounting['jobs']) != set(by_job): raise ValueError('Missing local allocation provenance')
    for job, record in accounting['jobs'].items():
        record.update(stage=by_job[job]['stage'], attempt=by_job[job].get('attempt', 1))
    verify_allocation_receipt(accounting)
    chosen = {}
    for stage in REQUIRED:
        successful = [s for s in states if s['stage'] == stage and accounting['jobs'][str(s['job_id'])]['state'] == 'COMPLETED'
                      and accounting['jobs'][str(s['job_id'])]['exit_code'] == '0:0']
        if not successful: raise ValueError('Missing successful stage: ' + stage)
        chosen[stage] = max(successful, key=lambda s: s.get('attempt', 1))
    for stage, state in chosen.items():
        key = stage_key(stage, state.get('attempt', 1)); source = args.state_dir / (key + '-results')
        if not source.exists():
            _fetch(SimpleNamespace(state_dir=args.state_dir, stage=stage, attempt=state.get('attempt', 1)), accounting)
        verify_result(source, state)
        copy_exact(source, args.bundle / stage)
    assets = args.state_dir / 'publication-assets'
    if not assets.exists():
        files = collect_execution_assets(states)
        assets.mkdir()
        for name, raw in files.items():
            path = assets / safe_relative(name); path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(raw)
    snapshot_identities = {stage_key(s['stage'], s.get('attempt', 1)) + '-code.tar.gz': s['code_sha256'] for s in states}
    for path in sorted(assets.rglob('*')):
        if not path.is_file(): continue
        relative = path.relative_to(assets); target = args.bundle / relative
        if target.exists():
            if relative.parts[0] == 'snapshots':
                verify_stage_snapshot(target.read_bytes(), snapshot_identities[path.name])
            elif file_sha(target) != file_sha(path):
                raise ValueError('Refuse to overwrite different execution evidence')
            continue
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(path, target)
    (args.bundle / 'allocations.json').write_text(json.dumps(accounting, indent=2, sort_keys=True) + '\n')
    inventory = {p.relative_to(args.bundle).as_posix(): file_sha(p) for p in args.bundle.rglob('*')
                 if p.is_file() and p.name != 'BUNDLE_SHA256.json'}
    (args.bundle / 'BUNDLE_SHA256.json').write_text(json.dumps(inventory, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'stages': list(chosen), 'files': len(inventory), 'gpu_hours': accounting['total']}))


if __name__ == '__main__': main()
