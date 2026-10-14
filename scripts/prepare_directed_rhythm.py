"""Census official source headers, then add capped difficult-class rhythm signals."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import threading

import numpy as np
import requests
import wfdb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from prepare_expanded_pathology import fetch_s3_file
from prepare_joint_pathology_inputs import write_csv
from titan_v4.reproducibility.directed_refinement import rhythm_reference, select_records
from titan_v4.reproducibility.refinement import read_manifest, sha256
from titan_v4.reproducibility.physionet import preprocess_primary9_signal
from titan_v4.data.primary9_registry import PRIMARY9_CLASSES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=24)
    parser.add_argument('--cap', type=int, default=500)
    parser.add_argument('--source-index', type=Path, default=ROOT / 'outputs/reviewer_verification/directed_pcgrad_2026-10-13/data/directed_official_SHA256SUMS.txt')
    parser.add_argument('--offline', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve previous preparation')
    index = args.source_index
    if sha256(index) != 'dfad0c26a276ce7e450e5002051f4c749304552d9add4d592928bbdcb6e4f31c':
        raise ValueError('Official source index changed')
    official = {}
    for line in index.read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        official[name.lstrip('*').removeprefix('./')] = digest
    cache = ROOT / 'data/cache/physionet/ecg-arrhythmia/1.0.0'
    url = 'https://physionet-open.s3.amazonaws.com/ecg-arrhythmia/1.0.0/'
    tls = threading.local()

    def fetch(name):
        if not hasattr(tls, 'session'):
            tls.session = requests.Session()
        if args.offline:
            if not (cache / name).exists() or sha256(cache / name) != official[name]:
                raise ValueError('Missing or modified pinned source')
        else:
            fetch_s3_file(cache / name, url + name, official[name], tls.session)

    fetch('ConditionNames_SNOMED-CT.csv')
    paths = sorted(n[:-4] for n in official if n.endswith('.hea'))
    blocked = []
    base = ROOT / 'outputs/reviewer_verification/partial_backbone_refinement_2026-10-12/manifests'
    for role in ('train', 'validation', 'followup'):
        blocked.extend(read_manifest(base / (role + '_records.csv')))
    history = read_manifest(ROOT / 'outputs/reviewer_verification/arrhythmia_primary9/cedia_gold_2026-05-26/primary9_label_history_672.csv')
    excluded_ids = {r['ecg_id'] for r in blocked} | {r['record_id'] for r in history}
    excluded_signals = {r['signal_sha256'] for r in blocked}

    def header(path):
        fetch(path + '.hea')
        raw = (cache / (path + '.hea')).read_text()
        codes = set()
        for line in raw.splitlines():
            if line.split(':', 1)[0].lstrip('# ').strip().lower() == 'dx':
                codes = {c.strip() for c in line.split(':', 1)[1].split(',')}
        return {'ecg_id': Path(path).name, 'record_path': path, 'rhythm_target': rhythm_reference(codes),
                'source_snomed_codes': json.dumps(sorted(codes)), 'header_sha256': official[path + '.hea'],
                'signal_sha256': official[path + '.mat']}

    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for i, row in enumerate(executor.map(header, paths), 1):
            rows.append(row)
            if i % 1000 == 0 or i == len(paths):
                print(f'Pinned source headers: {i}/{len(paths)}', flush=True)
    selected = select_records(rows, excluded_ids, excluded_signals, args.cap)
    if not selected:
        raise ValueError('No eligible new difficult-class signals')
    args.output.mkdir(parents=True)
    write_csv(args.output / 'header_census.csv', rows)

    def signal(row):
        fetch(row['record_path'] + '.mat')
        record = wfdb.rdrecord(str(cache / row['record_path']), physical=True)
        if record.fs != 500 or record.sig_len != 5000:
            raise ValueError('Unexpected original sampling/length')
        x = preprocess_primary9_signal(record.p_signal, record.sig_name, record.fs)
        row = dict(row, dataset='ecg-arrhythmia', version='1.0.0', doi='10.13026/wgex-er52',
                   source_release='ecg-arrhythmia', signal_filename=row['ecg_id'] + '.mat',
                   reference_label=PRIMARY9_CLASSES[row['rhythm_target']],
                   patient_token=hashlib.sha256(('ecg-arrhythmia:' + row['ecg_id']).encode()).hexdigest(),
                   tensor_sha256=hashlib.sha256(x.astype('<f4', copy=False).tobytes()).hexdigest())
        return x, row

    xs, entries = [], []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for i, (x, row) in enumerate(executor.map(signal, selected), 1):
            xs.append(x); entries.append(row)
            if i % 100 == 0 or i == len(selected):
                print(f'Pinned added signals: {i}/{len(selected)}', flush=True)
    np.savez_compressed(args.output / 'rhythm_extra.npz', x=np.stack(xs), y=np.asarray([r['rhythm_target'] for r in entries], dtype=np.int64),
                        ecg_id=np.asarray([r['ecg_id'] for r in entries]), tensor_sha256=np.asarray([r['tensor_sha256'] for r in entries]))
    write_csv(args.output / 'rhythm_extra_records.csv', entries)
    receipt = {'source_index_sha256': sha256(index), 'taxonomy_sha256': official['ConditionNames_SNOMED-CT.csv'],
               'census_records': len(rows), 'new_records': len(entries), 'cap_per_class': args.cap,
               'support': dict(Counter(r['reference_label'] for r in entries)),
               'selection': 'Complete source census; fixed hash order; five difficult classes; exclude known record IDs and waveform hashes',
               'pathology': 'All added-source pathology targets masked; no MI localization inferred',
               'files': {p.name: sha256(p) for p in sorted(args.output.iterdir()) if p.is_file()}}
    (args.output / 'preparation.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == '__main__':
    main()
