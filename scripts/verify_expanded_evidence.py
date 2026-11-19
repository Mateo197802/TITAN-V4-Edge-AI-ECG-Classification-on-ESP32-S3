"""Recalculate phase10 population, calibration, selection and followup metrics."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from verify_joint_evidence import ROOT, recalc_validation, c, read_manifest, sha256, pathology_targets, PATHOLOGY_CLASSES, pathology_metrics, select_candidates
from prepare_joint_pathology_inputs import META_SHA, SCP_SHA, INDEX_SHA, BASE
from titan_v4.reproducibility.expanded_pathology import select_expanded_population
from titan_v4.reproducibility.joint_refinement import resolve_historical_ptb_ids, unambiguous_rhythm_target
from titan_v4.data.label_registry import SCP_TO_RHYTHM


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--write', action='store_true')
    args = parser.parse_args(); bundle = args.bundle
    torch.set_num_threads(2)
    data = bundle / 'data'
    metadata = data / 'ptbxl_database.csv'; scp = data / 'scp_statements.csv'; index = data / 'official_SHA256SUMS.txt'
    if [sha256(p) for p in (metadata, scp, index)] != [META_SHA, SCP_SHA, INDEX_SHA]:
        raise ValueError('Pinned official source differs')
    meta = read_manifest(metadata); byid = {str(int(r['ecg_id'])): r for r in meta}
    official = {}
    for line in index.read_text().splitlines():
        h, name = line.split(maxsplit=1); official[name.lstrip('*').removeprefix('./')] = h
    roles = {role: read_manifest(BASE / (role + '_records.csv')) for role in ('train', 'validation', 'followup')}
    pats = {role: {byid[str(int(r['ecg_id']))]['patient_id'] for r in rows if r['dataset'] == 'ptb-xl'} for role, rows in roles.items()}
    history = read_manifest(ROOT / 'outputs/reviewer_verification/arrhythmia_primary9/cedia_gold_2026-05-26/primary9_label_history_672.csv')
    aliases = resolve_historical_ptb_ids([str(int(r['record_id'][2:])) for r in history if r['record_id'].upper().startswith('HR')], set(byid))
    hp = {byid[rid]['patient_id'] for rid in aliases}
    original_ids = {str(int(r['ecg_id'])) for r in roles['train'] if r['dataset'] == 'ptb-xl'}
    expected = select_expanded_population(meta, pats['followup'] | hp, pats['train'], original_ids)
    published = [read_manifest(data / (name + '_records.csv')) for name in ('path_extra', 'path_validation')]
    for want, actual in zip(expected, published):
        if [str(int(r['ecg_id'])) for r in want] != [r['ecg_id'] for r in actual]:
            raise ValueError('Complete outcome-blind population differs')
        for row in actual:
            source = byid[row['ecg_id']]; scores = ast.literal_eval(source['scp_codes']); y, _ = pathology_targets('ptb-xl', scores)
            if row['record_path'] != source['filename_hr'] or json.loads(row['source_scp_codes']) != scores:
                raise ValueError('Official record mapping differs')
            if row['patient_token'] != hashlib.sha256(('ptb-xl:' + source['patient_id']).encode()).hexdigest():
                raise ValueError('Official patient token differs')
            if any(int(row['target_' + name]) != int(y[j]) for j, name in enumerate(PATHOLOGY_CLASSES)):
                raise ValueError('Official pathology target differs')
            if int(row['rhythm_target']) != unambiguous_rhythm_target(scores, SCP_TO_RHYTHM):
                raise ValueError('Official rhythm target differs')
            if row['header_sha256'] != official[row['record_path'] + '.hea'] or row['signal_sha256'] != official[row['record_path'] + '.dat']:
                raise ValueError('Official source file hash differs')
    train_patients = pats['train'] | {r['patient_id'] for r in expected[0]}
    if train_patients & ({r['patient_id'] for r in expected[1]} | pats['followup'] | hp):
        raise ValueError('Training overlaps held-out patient')
    source_targets = read_manifest(data / 'path_targets.csv')
    for row in source_targets:
        scores = ast.literal_eval(byid[str(int(row['ecg_id']))]['scp_codes']) if row['dataset'] == 'ptb-xl' else {}
        y, mask = pathology_targets(row['dataset'], scores)
        if int(row['pathology_observed']) != int(mask[0]) or any(int(row['target_' + name]) != int(y[j]) for j, name in enumerate(PATHOLOGY_CLASSES)):
            raise ValueError('Original target changed')
    protocol = json.loads((bundle / 'training/protocol.json').read_text())
    selection = json.loads((bundle / 'results/selection.json').read_text())
    manifest = bundle / 'training/INPUT_SHA256.json'
    execution = json.loads((bundle / 'results/execution.json').read_text())
    result_hashes = json.loads((bundle / 'results/SHA256SUMS.json').read_text())
    for name, digest in result_hashes.items():
        if Path(name).name != name or sha256(bundle / 'results' / name) != digest:
            raise ValueError('Sealed CEDIA result differs')
    submission = json.loads((data / 'submission.json').read_text())
    if not sha256(manifest) == execution['input_manifest_sha256'] == submission['input_manifest_sha256']:
        raise ValueError('Execution/submission manifest differs')
    for name, digest in json.loads(manifest.read_text()).items():
        path = bundle / 'training' / name
        if path.is_file() and sha256(path) != digest:
            raise ValueError('Executed source differs')
    inputs = json.loads(manifest.read_text())
    prepared = json.loads((data / 'preparation.json').read_text())
    for name in ('path_extra.npz', 'path_validation.npz', 'path_targets.npz'):
        if inputs[name] != prepared['files'][name]:
            raise ValueError('Prepared and executed arrays differ')
    if sha256(bundle / 'training/start.pth') != protocol['start_sha256'] or execution['parameters'] != 1143751:
        raise ValueError('Original architecture/start contract differs')
    base_raw = dict(np.load(bundle / 'results/baseline_validation.npz', allow_pickle=False))
    if len(base_raw['y']) != 1476 or len(base_raw['path_targets']) != 2027:
        raise ValueError('Validation dimensions differ')
    if list(base_raw['path_ecg_id']) != [r['ecg_id'] for r in published[1]]:
        raise ValueError('Calibration signal order differs')
    ys = np.asarray([[int(r['target_' + name]) for name in PATHOLOGY_CLASSES] for r in published[1]], np.float32)
    if not np.array_equal(base_raw['path_targets'], ys) or not (base_raw['path_mask'] == 1).all():
        raise ValueError('Calibration targets differ')
    original = np.load(ROOT / 'outputs/reviewer_verification/partial_backbone_refinement_2026-10-12/training/phase2_reference/validation_features.npz', allow_pickle=False)
    if any(not np.array_equal(base_raw[k], original[k]) for k in ('y', 'dataset', 'ecg_id')):
        raise ValueError('Rhythm validation changed')
    base = recalc_validation(base_raw)
    if any(selection['baseline'][k] != v for k, v in base.items()):
        raise ValueError('Control validation differs')
    candidates = []
    policy = protocol['selection']
    for row in selection['candidates']:
        if row['trial'] not in protocol['optimization']['trials']:
            raise ValueError('Candidate optimization differs from frozen protocol')
        raw = dict(np.load(bundle / 'results' / (row['name'] + '_validation.npz'), allow_pickle=False))
        if any(not np.array_equal(raw[k], base_raw[k]) for k in ('y', 'dataset', 'ecg_id', 'path_targets', 'path_mask', 'path_ecg_id')):
            raise ValueError('Candidate validation references changed')
        actual = recalc_validation(raw)
        if any(row[k] != v for k, v in actual.items()):
            raise ValueError('Candidate calibration/metrics differ')
        errors = {name: float(np.sqrt(np.mean(((raw[name] - base_raw[name]) / np.maximum(base_raw[name].std(0), .05)) ** 2))) for name in ('quality', 'biometrics')}
        checks = {'protected_recalls': all(m['recall'][label] >= base['sources'][s]['recall'][label] - policy['maximum_recall_drop'] - 1e-12
                  for s, m in actual['sources'].items() for label in policy['protected_classes'] if m['support'][label] >= policy['minimum_class_support']),
                  'rhythm_macro_preserved': actual['metrics']['macro_f1'] >= base['metrics']['macro_f1'] - .005 - 1e-12,
                  'rhythm_sources_preserved': actual['rhythm_score'] >= base['rhythm_score'] - .005 - 1e-12,
                  'quality_biometrics_preserved': max(errors.values()) <= .15}
        if errors != row['quality_biometrics_nrmse'] or checks != row['preservation_checks'] or all(checks.values()) != row['eligible']:
            raise ValueError('Preservation diagnostics differ')
        candidates.append(row)
    if {(r['trial']['name'], r['epoch']) for r in candidates} != {(t['name'], e) for t in protocol['optimization']['trials'] for e in protocol['optimization']['checkpoints']}:
        raise ValueError('Trial coverage differs')
    chosen = select_candidates(candidates, base, role='validation')
    if any(selection[k] != v for k, v in chosen.items()):
        raise ValueError('Validation selection differs')
    if selection['references_changed'] or not selection['evaluation_labels_loaded_after_selection'] or not selection['thresholds_fitted_on_validation_only']:
        raise ValueError('Selection boundary changed')
    refit = json.loads((bundle / 'results/refit.json').read_text())
    recomputed = {}
    for name in ('baseline', 'joint_selected', 'rhythm_selected'):
        rows = read_manifest(bundle / 'results' / (name + '_predictions.csv'))
        if [(r['cohort'], r['dataset'], r['ecg_id']) for r in rows] != [(r['cohort'], r['dataset'], r['ecg_id']) for r in roles['followup']]:
            raise ValueError('Followup population differs')
        rp = np.asarray([[float(r['prob_' + label]) for label in c.CLASSES] for r in rows])
        if any(r['reference'] != source['reference_label'] or r['tensor_sha256'] != source['tensor_sha256'] or r['prediction'] != c.CLASSES[int(p.argmax())] for r, source, p in zip(rows, roles['followup'], rp)):
            raise ValueError('Followup rhythm target/prediction differs')
        if any(int(r['pathology_observed']) != int(r['dataset'] == 'ptb-xl') for r in rows):
            raise ValueError('Unknown pathology source assigned a reference')
        selected = selection.get(name) if name != 'baseline' else None
        if name != 'baseline':
            checkpoint = bundle / 'model' / (name + '.pth')
            if sha256(checkpoint) != selection['checkpoints'][name]:
                raise ValueError('Chosen model hash differs')
            if selected and not refit[name].get('all_state_tensors_bitwise_equal'):
                raise ValueError('Selected refit not verified')
            state = c.load_model(checkpoint).state_dict(); reference = c.load_model(bundle / 'training/start.pth').state_dict()
            fitted = c.load_model(bundle / 'results' / (selected['name'] + '.pth')).state_dict() if selected else reference
            if any(not torch.equal(v, fitted[k]) for k, v in state.items()):
                raise ValueError('Published refit does not match selected candidate')
            allowed = ('head_pathology.',) if selected and selected['trial']['blocks'] == 0 else ('res_layers.3.', 'res_layers.4.', 'head_rhythm.', 'head_pathology.') if selected else ()
            if any(not torch.equal(v, reference[k]) for k, v in state.items() if not k.startswith(allowed) or k.endswith(('running_mean', 'running_var', 'num_batches_tracked'))):
                raise ValueError('Frozen parameters/buffers changed')
        thresholds = np.asarray((selected or base)['thresholds'])
        native = [r for r in rows if r['pathology_observed'] == '1']
        targets = np.asarray([[int(r['target_' + label]) for label in PATHOLOGY_CLASSES] for r in native])
        pp = np.asarray([[float(r['score_' + label]) for label in PATHOLOGY_CLASSES] for r in native])
        if not np.array_equal(pp >= thresholds, [[bool(int(r['pred_' + label])) for label in PATHOLOGY_CLASSES] for r in native]):
            raise ValueError('Frozen threshold decisions differ')
        for row in native:
            y, _ = pathology_targets('ptb-xl', ast.literal_eval(byid[str(int(row['ecg_id']))]['scp_codes']))
            if any(int(row['target_' + label]) != int(y[j]) for j, label in enumerate(PATHOLOGY_CLASSES)):
                raise ValueError('Followup pathology target changed')
        actual = {'rhythm': c.metrics(np.asarray([c.CLASSES.index(r['reference']) for r in rows]), rp), 'pathology': pathology_metrics(targets, pp, thresholds)}
        saved = json.loads((bundle / 'results' / (name + '_metrics.json')).read_text())
        if any(saved[k] != v for k, v in actual.items()):
            raise ValueError('Followup metrics differ')
        recomputed[name] = actual
    replay = json.loads((bundle / 'replay/validation_source_replay.json').read_text())
    if not replay['all_passed'] or replay['checkpoints'] != len(candidates) + 1:
        raise ValueError('Candidate source replay incomplete')
    for name in ('joint_selected', 'rhythm_selected'):
        inference = json.loads((bundle / 'replay' / (name + '_cpu_signal_replay.json')).read_text())
        if not inference['passed'] or inference['records'] != 2171 or inference['checkpoint_sha256'] != selection['checkpoints'][name]:
            raise ValueError('Followup source replay incomplete')
    for name in ('joint_selected', 'rhythm_selected'):
        historical = json.loads((bundle / 'historical672' / (name + '_report.json')).read_text())
        rows = read_manifest(bundle / 'historical672' / (name + '_paired_predictions.csv'))
        if historical['records'] != 672 or historical['checkpoint_sha256'] != selection['checkpoints'][name] or len(rows) != 672:
            raise ValueError('Historical signal run incomplete')
        lookup = {(r['source'], r['record_id']): r for r in history}
        if {(r['source'], r['record_id']) for r in rows} != set(lookup):
            raise ValueError('Historical population differs')
        probs = np.asarray([[float(r['prob_' + label]) for label in c.CLASSES] for r in rows])
        for column in ('original_rhythm_label', 'final_rhythm_label'):
            if any(r[column] != lookup[(r['source'], r['record_id'])][column] for r in rows):
                raise ValueError('Historical reference changed')
            metrics = c.metrics(np.asarray([c.CLASSES.index(r[column]) for r in rows]), probs)
            saved = historical['results'][column]
            if metrics['correct'] != saved['correct_predictions'] or abs(metrics['macro_f1'] - saved['macro_f1']) > 1e-12:
                raise ValueError('Historical metric differs')
    report = {'complete_source_populations_verified': True, 'added_training': len(published[0]), 'pathology_validation': len(published[1]),
              'source_file_hashes_verified': 2 * sum(map(len, published)), 'training_overlap_patients': 0,
              'candidates_recalculated': len(candidates), 'models': recomputed}
    summary = bundle / 'verification_summary.json'
    if args.write:
        summary.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    elif json.loads(summary.read_text()) != report:
        raise ValueError('Verification summary differs')
    sums = bundle / 'SHA256SUMS.json'
    inventory = {p.relative_to(bundle).as_posix(): sha256(p) for p in sorted(bundle.rglob('*')) if p.is_file() and p != sums and '__pycache__' not in p.parts and p.suffix != '.pyc'}
    if args.write:
        sums.write_text(json.dumps(inventory, indent=2) + '\n', encoding='utf-8')
    elif json.loads(sums.read_text()) != inventory:
        raise ValueError('Bundle hash inventory differs')
    print(json.dumps({'verified_artifacts': len(inventory), 'models': {name: {'accuracy': m['rhythm']['accuracy'], 'macro_f1': m['rhythm']['macro_f1'],
          'pathology_accuracy': m['pathology']['primary5_mean_label_accuracy'], 'pathology_f1': m['pathology']['primary5_macro_f1']} for name, m in recomputed.items()}}, indent=2))


if __name__ == '__main__':
    main()
