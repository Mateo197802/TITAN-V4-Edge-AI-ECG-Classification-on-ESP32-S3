"""Preserve the cancelled source-mapping preflight run with sanitized receipts."""
import argparse
import hashlib
import json
from pathlib import Path
import re

from manage_joint_cedia_run import ROOT, rpc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state', type=Path, default=ROOT / 'data/cache/directed_pcgrad_cedia')
    parser.add_argument('--reason', default='Source headers contain codes absent from the published taxonomy; reject unsupported and all unrecognized codes before final fitting')
    args = parser.parse_args()
    state = json.loads((args.state / 'submission.json').read_text())
    job = state['stdout'].strip().split(';')[0]
    if not job.isdigit():
        raise ValueError('Invalid Slurm job identifier')
    script = f'''
import json,pathlib,subprocess
root=pathlib.Path({state['remote_root']!r})
project=(pathlib.Path.home()/'V4_CEDIA_LINK').resolve(strict=True)
if not root.resolve(strict=True).is_relative_to(project):raise ValueError('Unsafe run root')
r=subprocess.run(['sacct','-j',{job!r},'--noheader','--parsable2','--format=JobID,State,ExitCode,Elapsed'],capture_output=True,text=True,timeout=30)
print(json.dumps({{'accounting':r.stdout,'sealed_results':(root/'results/SHA256SUMS.json').exists(),'logs':{{s:(root/('slurm-'+{job!r}+s)).read_text(errors='replace') for s in ('.out','.err')}}}}))
'''
    payload = json.loads(rpc(script))
    if payload['sealed_results'] or '|CANCELLED' not in payload['accounting']:
        raise ValueError('Preliminary run cancellation not confirmed')
    target = ROOT / 'outputs/reviewer_verification/directed_pcgrad_2026-10-13/attempts' / job
    receipt = {'job_id': job, 'accounting': payload['accounting'], 'sealed_results': False,
               'input_manifest_sha256': state['input_manifest_sha256'],
               'reason': args.reason,
               'used_for_model_selection': False, 'logs': {}}
    for suffix, content in payload['logs'].items():
        safe = re.sub(r'/home/[^/\s]+', '/home/REDACTED_USER', content).replace('/Mateo Gavilanes/', '/REDACTED_PROJECT/')
        destination = target / ('slurm-' + job + suffix)
        destination.write_text(safe, encoding='utf-8')
        receipt['logs'][destination.name] = {'original_sha256': hashlib.sha256(content.encode()).hexdigest(),
                                           'published_sha256': hashlib.sha256(destination.read_bytes()).hexdigest()}
    (target / 'cancellation.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'job_id': job, 'cancelled': True, 'used_for_model_selection': False}))


if __name__ == '__main__':
    main()
