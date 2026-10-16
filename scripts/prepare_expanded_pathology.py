"""Materialize all eligible PTB-XL training additions and fold-9 pathology validation."""
import argparse
import ast
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time

import numpy as np
import requests
import wfdb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from prepare_joint_pathology_inputs import META_SHA, SCP_SHA, INDEX_SHA, BASE, fetch_pinned_file, write_csv
from titan_v4.reproducibility.expanded_pathology import select_expanded_population
from titan_v4.reproducibility.joint_refinement import pathology_targets, unambiguous_rhythm_target, resolve_historical_ptb_ids, PATHOLOGY_CLASSES
from titan_v4.reproducibility.refinement import read_manifest, sha256
from titan_v4.reproducibility.physionet import preprocess_primary9_signal
from titan_v4.data.label_registry import SCP_TO_RHYTHM
from titan_v4.data.primary9_registry import PRIMARY9_CLASSES


def fetch_s3_file(target, url, expected, session):
    if target.exists():
        if sha256(target) != expected:
            raise ValueError('Existing source hash differs')
        return
    for attempt in range(3):
        try:
            response = session.get(url, timeout=(10, 60))
            response.raise_for_status()
            payload = response.content
            break
        except requests.RequestException:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)
    if hashlib.sha256(payload).hexdigest() != expected:
        raise ValueError('Downloaded source hash differs')
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as f:
        temporary = Path(f.name); f.write(payload)
    try:
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--metadata', type=Path, required=True)
    parser.add_argument('--scp-statements', type=Path, required=True)
    parser.add_argument('--source-index', type=Path, default=ROOT / 'data/cache/refinement_source_indexes/ptb-xl/1.0.3/SHA256SUMS.txt')
    parser.add_argument('--data-root', type=Path, default=ROOT / 'data/cache/physionet')
    parser.add_argument('--workers', type=int, default=12)
    parser.add_argument('--transport', choices=('physionet', 's3'), default='physionet')
    parser.add_argument('--offline', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a fresh output directory')
    if args.workers < 1:
        parser.error('workers must be positive')
    for path, expected in ((args.metadata, META_SHA), (args.scp_statements, SCP_SHA), (args.source_index, INDEX_SHA)):
        if sha256(path) != expected:
            raise ValueError('Pinned upstream metadata/index differs')
    official = {}
    for line in args.source_index.read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        official[name.lstrip('*').removeprefix('./')] = digest
    meta = read_manifest(args.metadata)
    byid = {str(int(r['ecg_id'])): r for r in meta}
    roles = {role: read_manifest(BASE / (role + '_records.csv')) for role in ('train', 'validation', 'followup')}
    patients = {role: {byid[str(int(r['ecg_id']))]['patient_id'] for r in rows if r['dataset'] == 'ptb-xl'} for role, rows in roles.items()}
    historical = read_manifest(ROOT / 'outputs/reviewer_verification/arrhythmia_primary9/cedia_gold_2026-05-26/primary9_label_history_672.csv')
    aliases = resolve_historical_ptb_ids([str(int(r['record_id'][2:])) for r in historical if r['record_id'].upper().startswith('HR')], set(byid))
    historical_patients = {byid[rid]['patient_id'] for rid in aliases}
    existing_ids = {str(int(r['ecg_id'])) for r in roles['train'] if r['dataset'] == 'ptb-xl'}
    additions, validation = select_expanded_population(meta, patients['followup'] | historical_patients, patients['train'], existing_ids)
    if {r['patient_id'] for r in additions} & patients['validation']:
        raise ValueError('New training overlaps original validation patients')
    blocked_hashes = {r['signal_sha256'] for r in roles['validation'] + roles['followup']}
    if any(official[r['filename_hr'] + '.dat'] in blocked_hashes for r in additions):
        raise ValueError('Cross-partition waveform duplicate')
    args.output.mkdir(parents=True)
    packed = {}
    for role, rows in roles.items():
        values = [pathology_targets(row['dataset'], ast.literal_eval(byid[str(int(row['ecg_id']))]['scp_codes']) if row['dataset'] == 'ptb-xl' else {}) for row in rows]
        packed[role + '_targets'] = np.stack([v[0] for v in values])
        packed[role + '_mask'] = np.stack([v[1] for v in values])
    np.savez_compressed(args.output / 'path_targets.npz', **packed)
    sessions = threading.local()

    def materialize(row):
        record_path = row['filename_hr']
        base = args.data_root / 'ptb-xl/1.0.3'
        url = ('https://physionet.org/files/' if args.transport == 'physionet' else 'https://physionet-open.s3.amazonaws.com/') + 'ptb-xl/1.0.3/'
        if not hasattr(sessions, 'session'):
            sessions.session = requests.Session()
        for suffix in ('.hea', '.dat'):
            if args.transport == 's3' and not args.offline:
                fetch_s3_file(base / (record_path + suffix), url + record_path + suffix, official[record_path + suffix], sessions.session)
            else:
                fetch_pinned_file(base / (record_path + suffix), url + record_path + suffix, official[record_path + suffix], not args.offline)
        record = wfdb.rdrecord(str(base / record_path), physical=True)
        if record.fs != 500:
            raise ValueError('Unexpected sampling frequency')
        x = preprocess_primary9_signal(record.p_signal, record.sig_name, record.fs)
        scores = ast.literal_eval(row['scp_codes'])
        y, mask = pathology_targets('ptb-xl', scores)
        rhythm = unambiguous_rhythm_target(scores, SCP_TO_RHYTHM)
        entry = {'ecg_id': str(int(row['ecg_id'])), 'dataset': 'ptb-xl', 'version': '1.0.3', 'doi': '10.13026/kfzx-aw45',
                 'source_release': 'ptb-xl', 'record_path': record_path, 'signal_filename': Path(record_path).name + '.dat',
                 'header_sha256': official[record_path + '.hea'], 'signal_sha256': official[record_path + '.dat'],
                 'tensor_sha256': hashlib.sha256(x.astype('<f4', copy=False).tobytes()).hexdigest(),
                 'strat_fold': row['strat_fold'], 'patient_token': hashlib.sha256(('ptb-xl:' + row['patient_id']).encode()).hexdigest(),
                 'rhythm_target': rhythm, 'reference_label': PRIMARY9_CLASSES[rhythm] if rhythm >= 0 else 'unknown',
                 'source_scp_codes': json.dumps(scores, sort_keys=True)}
        entry.update({'target_' + name: int(y[j]) for j, name in enumerate(PATHOLOGY_CLASSES)})
        return x, y, mask, entry

    receipt = {'metadata_sha256': META_SHA, 'scp_statements_sha256': SCP_SHA, 'official_index_sha256': INDEX_SHA,
               'selection': 'All eligible fold 1-8 additions and fold 9 validation, independent of labels/predictions',
               'excluded': 'All known historical and followup native patients; no train/validation patient overlap',
               'preprocessing': 'Unchanged six frontal leads, 125 Hz, [6,1250]', 'roles': {}}
    receipt['transport'] = args.transport
    for name, selected in (('path_extra', additions), ('path_validation', validation)):
        xs, ys, masks, rows = [], [], [], []
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            for i, (x, y, mask, row) in enumerate(executor.map(materialize, selected), 1):
                xs.append(x); ys.append(y); masks.append(mask); rows.append(row)
                if i % 200 == 0 or i == len(selected):
                    print(f'{name}: verified {i}/{len(selected)}', flush=True)
        xx, yy, mm = np.stack(xs), np.stack(ys), np.stack(masks)
        np.savez_compressed(args.output / (name + '.npz'), x=xx, targets=yy, mask=mm,
                            y=np.asarray([r['rhythm_target'] for r in rows], dtype=np.int64),
                            ecg_id=np.asarray([r['ecg_id'] for r in rows]), tensor_sha256=np.asarray([r['tensor_sha256'] for r in rows]))
        write_csv(args.output / (name + '_records.csv'), rows)
        receipt['roles'][name] = {'records': len(rows), 'patients': len({r['patient_token'] for r in rows}),
                                 'positive_support': dict(zip(PATHOLOGY_CLASSES, yy.sum(0).astype(int).tolist())),
                                 'unknown_rhythm': int(sum(r['rhythm_target'] < 0 for r in rows))}
    receipt['files'] = {p.name: sha256(p) for p in sorted(args.output.iterdir()) if p.is_file()}
    (args.output / 'preparation.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
