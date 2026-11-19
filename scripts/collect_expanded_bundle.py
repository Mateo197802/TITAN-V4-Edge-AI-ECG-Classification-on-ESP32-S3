"""Collect exact phase10 inputs/results without publishing source waveforms."""
import argparse
import json
from pathlib import Path
import re
import shutil

from manage_joint_cedia_run import ROOT, sha


def copy_once(source, target):
    if target.exists():
        raise FileExistsError('Preserve existing collected artifact: ' + target.name)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('inputs', 'results'))
    parser.add_argument('--inputs', type=Path, default=ROOT / 'data/cache/expanded_pathology_inputs_s3')
    parser.add_argument('--state', type=Path, default=ROOT / 'data/cache/expanded_population_cedia')
    args = parser.parse_args()
    bundle = ROOT / 'outputs/reviewer_verification/expanded_population_2026-10-13'
    data = bundle / 'data'
    if args.phase == 'inputs':
        for name in ('path_extra_records.csv', 'path_validation_records.csv', 'preparation.json'):
            copy_once(args.inputs / name, data / name)
        previous = ROOT / 'outputs/reviewer_verification/compatible_heads_2026-10-13/data'
        for name in ('path_targets.csv', 'path_targets.npz'):
            copy_once(previous / name, data / name)
        upstream = ROOT.parent / 'TITAN-V4-New-Cohort-2026-10-10/upstream'
        for name in ('ptbxl_database.csv', 'scp_statements.csv'):
            copy_once(upstream / name, data / name)
        copy_once(ROOT / 'data/cache/refinement_source_indexes/ptb-xl/1.0.3/SHA256SUMS.txt', data / 'official_SHA256SUMS.txt')
        # This large input is staged for CEDIA only, and moved back to ignored cache after submission.
        copy_once(args.inputs / 'path_validation.npz', bundle / 'training/path_validation.npz')
        print('Collected official manifests and staged calibration tensors')
        return
    source = args.state / 'results'
    inventory = json.loads((source / 'SHA256SUMS.json').read_text())
    for name, digest in inventory.items():
        if Path(name).name != name or sha(source / name) != digest:
            raise ValueError('Fetched result hash differs')
    for p in sorted(source.iterdir()):
        if p.is_file():
            copy_once(p, bundle / 'results' / p.name)
    for name in ('joint_selected.pth', 'rhythm_selected.pth'):
        copy_once(source / name, bundle / 'model' / name)
    state = json.loads((args.state / 'submission.json').read_text())
    job = state['stdout'].strip().split(';')[0]
    receipt = {'job_id': job, 'remote_run': 'V4_CEDIA/99_RESEARCH_RUNS/TITAN-V4-Expanded-Population-2026-10-13',
               'exit_code': state['exit_code'], 'input_manifest_sha256': state['input_manifest_sha256']}
    copy_once(args.state / 'INPUT_SHA256.json', bundle / 'training/INPUT_SHA256.json')
    (data / 'submission.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
    log_receipt = {}
    for suffix in ('out', 'err'):
        name = f'slurm-{job}.{suffix}'
        original = args.state / name
        text = original.read_text(errors='replace')
        text = re.sub(r'/home/[^/\s]+', '/home/REDACTED_USER', text).replace('/Mateo Gavilanes/', '/REDACTED_PROJECT/')
        target = bundle / 'logs' / name; target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding='utf-8')
        log_receipt[name] = {'original_sha256': sha(original), 'published_sha256': sha(target)}
    (bundle / 'logs/redaction_receipt.json').write_text(json.dumps(log_receipt, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'job_id': job, 'verified_result_files': len(inventory)}))


if __name__ == '__main__':
    main()
