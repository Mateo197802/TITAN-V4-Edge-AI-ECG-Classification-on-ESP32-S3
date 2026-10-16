"""Stage, inspect, or retrieve an isolated CEDIA joint-refinement run."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import shlex
import subprocess
import tarfile

ROOT=Path(__file__).resolve().parents[1]
TRAINING=ROOT/'outputs/reviewer_verification/joint_refinement_2026-10-12/training'
SSH=['ssh','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=8',
     '-o','ServerAliveInterval=30','-o','ServerAliveCountMax=3','cedia-kevin']


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def rpc(script,payload=None):
    r=subprocess.run(SSH+['python3 -c '+shlex.quote(script)],input=payload,capture_output=True,timeout=900)
    if r.returncode:raise RuntimeError(r.stderr.decode(errors='replace'))
    return r.stdout


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['submit','status','fetch'])
    parser.add_argument('--state-dir',type=Path,default=ROOT/'data/cache/joint_cedia_state')
    parser.add_argument('--path-inputs',type=Path)
    parser.add_argument('--raw-inputs',type=Path)
    parser.add_argument('--training-dir',type=Path,default=TRAINING)
    parser.add_argument('--reuse-remote-path-inputs',action='store_true',help='Copy exact pathology NPZs from source run and verify local hashes')
    parser.add_argument('--source-run',default='TITAN-V4-Same-Architecture-Partial-Backbone-2026-10-12')
    parser.add_argument('--run-name',default='TITAN-V4-Joint-Refinement-2026-10-12')
    args=parser.parse_args();args.state_dir.mkdir(parents=True,exist_ok=True)
    state_file=args.state_dir/'submission.json'
    if args.action=='submit':
        if state_file.exists():raise FileExistsError('Run already submitted')
        if args.path_inputs is None or args.raw_inputs is None:parser.error('submit requires both input directories')
        if Path(args.run_name).name!=args.run_name or Path(args.source_run).name!=args.source_run:parser.error('Run names must be simple directory names')
        files={p.name:p for p in sorted(args.training_dir.iterdir()) if p.is_file() and p.suffix!='.pyc' and p.name!='INPUT_SHA256.json'}
        files.update({n:args.path_inputs/n for n in ['path_targets.npz','path_extra.npz']})
        manifest={n:sha(p) for n,p in files.items()}
        manifest.update({n:sha(args.raw_inputs/n) for n in ['train.npz','validation.npz','evaluation.npz']})
        buffer=io.BytesIO()
        with tarfile.open(fileobj=buffer,mode='w:gz') as archive:
            for name,path in files.items():
                if args.reuse_remote_path_inputs and name in ('path_targets.npz','path_extra.npz'):continue
                archive.add(path,arcname=name,recursive=False)
            raw=(json.dumps(manifest,indent=2)+'\n').encode();info=tarfile.TarInfo('INPUT_SHA256.json');info.size=len(raw);archive.addfile(info,io.BytesIO(raw))
        copy_names=('train.npz','validation.npz','evaluation.npz')+(('path_targets.npz','path_extra.npz') if args.reuse_remote_path_inputs else ())
        script=f'''
import hashlib,io,json,pathlib,shutil,subprocess,sys,tarfile
project=(pathlib.Path.home()/'V4_CEDIA_LINK').resolve(strict=True)
source=project/'99_RESEARCH_RUNS'/{args.source_run!r}
root=project/'99_RESEARCH_RUNS'/{args.run_name!r}
if root.exists():raise FileExistsError('Run already exists')
if not source.resolve(strict=True).is_relative_to(project):raise ValueError('Source escaped project')
if not root.resolve().is_relative_to(project):raise ValueError('Target escaped project')
root.mkdir()
with tarfile.open(fileobj=io.BytesIO(sys.stdin.buffer.read()),mode='r:gz') as archive:
 for member in archive.getmembers():
  if not member.isfile() or pathlib.PurePosixPath(member.name).name!=member.name:raise ValueError('Unsafe archive member')
  (root/member.name).write_bytes(archive.extractfile(member).read())
for name in {copy_names!r}:shutil.copy2(source/name,root/name)
manifest=json.loads((root/'INPUT_SHA256.json').read_text())
for name,h in manifest.items():
 if hashlib.sha256((root/name).read_bytes()).hexdigest()!=h:raise ValueError('Input hash differs: '+name)
r=subprocess.run(['sbatch','--parsable','--chdir='+str(root),'job.sbatch'],cwd=root,capture_output=True,text=True,timeout=60)
print(json.dumps({{'remote_root':str(root),'exit_code':r.returncode,'stdout':r.stdout,'stderr':r.stderr,'input_manifest_sha256':hashlib.sha256((root/'INPUT_SHA256.json').read_bytes()).hexdigest()}}))
'''
        state=json.loads(rpc(script,buffer.getvalue()))
        state_file.write_text(json.dumps(state,indent=2)+'\n',encoding='utf-8')
        (args.state_dir/'INPUT_SHA256.json').write_bytes(raw)
        print(json.dumps(state,indent=2))
        if state['exit_code']:raise RuntimeError('Slurm submission failed')
        return
    state=json.loads(state_file.read_text());job=state['stdout'].strip().split(';')[0]
    if args.action=='status':
        script=f'''
import json,pathlib,subprocess
root=pathlib.Path({state['remote_root']!r})
r=subprocess.run(['sacct','-j',{job!r},'--noheader','--parsable2','--format=JobID,State,ExitCode,Elapsed'],capture_output=True,text=True,timeout=30)
result={{'accounting':r.stdout,'sealed':(root/'results/SHA256SUMS.json').exists()}}
for name in ['slurm-{job}.out','slurm-{job}.err']:
 p=root/name;result[name]=p.read_text(errors='replace')[-2500:] if p.exists() else None
print(json.dumps(result))
'''
        print(rpc(script).decode());return
    if (args.state_dir/'results').exists():raise FileExistsError('Preserve retrieved results')
    script=f'''
import io,pathlib,sys,tarfile
root=pathlib.Path({state['remote_root']!r})
if not (root/'results/SHA256SUMS.json').is_file():raise ValueError('Unsealed results')
b=io.BytesIO()
with tarfile.open(fileobj=b,mode='w:gz') as archive:
 archive.add(root/'results',arcname='results')
 for name in ['slurm-{job}.out','slurm-{job}.err']:
  if (root/name).is_file():archive.add(root/name,arcname=name)
sys.stdout.buffer.write(b.getvalue())
'''
    with tarfile.open(fileobj=io.BytesIO(rpc(script)),mode='r:gz') as archive:
        for member in archive.getmembers():
            if Path(member.name).is_absolute() or '..' in Path(member.name).parts or member.issym() or member.islnk():raise ValueError('Unsafe result archive')
        archive.extractall(args.state_dir,filter='data')
    inventory=json.loads((args.state_dir/'results/SHA256SUMS.json').read_text())
    for name,h in inventory.items():
        if Path(name).name!=name or sha(args.state_dir/'results'/name)!=h:raise ValueError('Result hash differs')
    print(json.dumps({'job_id':job,'verified_results':len(inventory)}))


if __name__=='__main__':main()
