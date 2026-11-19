"""Freeze paired fitting/validation inputs, verifying original source and tensor hashes."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from titan_v4.reproducibility.pretrained_transfer.inputs import build_pair_manifest, verify_source_pairs, verify_archived_inputs
from titan_v4.reproducibility.refinement import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, default=ROOT / 'data/cache/physionet')
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/founder_transfer_2026-10-14.json')
    parser.add_argument('--offline', action='store_true', help='Use only existing pinned source files')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve existing input bundles')
    config = json.loads(args.config.read_text(encoding='utf-8'))
    config['data_root'] = str(args.data_root.resolve())
    archived = verify_archived_inputs(ROOT, config)
    rows = build_pair_manifest(ROOT, config)
    def progress(index):
        if index % 1000 == 0:
            print(json.dumps({'verified_source_pairs': index}), flush=True)
    receipts = verify_source_pairs(rows, args.data_root, config['missing_sample_policy'],
                                   workers=args.workers, progress=progress)
    flags = Counter(flag for row in receipts for flag in row['flags'])
    quality_records = [row for row in receipts if row['flags']]
    args.output.mkdir(parents=True)
    (args.output / 'paired_records.json').write_text(json.dumps([asdict(r) for r in rows], indent=2) + '\n', encoding='utf-8')
    (args.output / 'source_quality.json').write_text(json.dumps(quality_records, indent=2) + '\n', encoding='utf-8')
    receipt = {'scope': 'input preparation, no neural inference', 'records': len(rows),
               'roles': dict(Counter(r.role for r in rows)), 'quality_flags': dict(flags),
               'all_source_bytes_and_student_tensors_verified': True,
               'archived_inputs_sha256': archived,
               'configuration_sha256': sha256(args.config),
               'paired_manifest_sha256': sha256(args.output / 'paired_records.json')}
    (args.output / 'input_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
