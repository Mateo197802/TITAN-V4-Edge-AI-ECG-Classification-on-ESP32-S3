"""Sealed packet verification and source-based inference after selection closes."""
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import tarfile
from urllib.parse import quote

import numpy as np
from scipy.special import expit, softmax
from sklearn.metrics import classification_report, accuracy_score, f1_score
import torch

from .contracts import PairedRecord, PRIMARY9_CLASSES, PATHOLOGY_CLASSES, CONTRACTS, SEEDS, STUDENT_SHA256
from .execution import (file_sha, verify_inventory, verify_result, IDENTITY_FIELDS, verify_frozen_inputs,
                        successful_stage, public_paths, safe_relative, BUCKETS, CEILINGS)
from .teachers import allocation, paired_rows, _json, seal_directory, adaptation_candidates
from .selection import load_predictions, prediction_metrics, qualify_teachers, select_students, student_eligibility
from .distillation import join_teacher_targets
from .students import load_exported_student, _producer_artifacts, ARMS, require_arm, assert_student_preserved
from .inputs import load_pair, load_student_tensor
from titan_v4.reproducibility.refinement import _ensure_file, RELEASES, source_release as refinement_source_release
from titan_v4.reproducibility.physionet import dataset_for_record
from titan_v4.reproducibility.joint_refinement import pathology_metrics, PRIMARY5_INDEX
from titan_v4.reproducibility.supervision_control import patient_folds


def assert_references(records, data, endpoint):
    keys = data['keys'].tolist()
    expected = {r.key: r for r in records}
    if len(keys) != len(set(keys)) or len(records) != len(expected) or set(keys) != set(expected):
        raise ValueError('Prediction coverage differs from frozen cohort')
    if data['logits'].shape != (len(records), 9 if endpoint == 'rhythm' else 10):
        raise ValueError('Prediction coverage/output shape differs')
    ordered = [expected[k] for k in keys]
    values = {'labels': np.asarray([r.rhythm_label for r in ordered]) if endpoint == 'rhythm' else np.asarray([r.path_targets for r in ordered]),
              'mask': np.ones(len(ordered)) if endpoint == 'rhythm' else np.asarray([r.path_mask for r in ordered]),
              'sources': np.asarray([r.dataset for r in ordered]),
              'patients': np.asarray([r.patient_token or r.key for r in ordered]),
              'roles': np.asarray([r.role for r in ordered]),
              'classes': np.asarray(PRIMARY9_CLASSES if endpoint == 'rhythm' else PATHOLOGY_CLASSES)}
    if any(not np.array_equal(data[k], v) for k, v in values.items()):
        raise ValueError('Prediction reference/class/patient identity differs')
    if not np.isfinite(data['logits']).all(): raise ValueError('Nonfinite prediction evidence')


def assert_replay(actual, expected, *, endpoint, thresholds=None):
    actual, expected = np.asarray(actual), np.asarray(expected)
    if (actual.shape != expected.shape or actual.ndim != 2 or not np.isfinite(actual).all()
            or not np.isfinite(expected).all() or not np.allclose(actual, expected, atol=1e-6, rtol=1e-5)):
        raise ValueError('Source replay numerical comparison failed')
    if endpoint == 'rhythm':
        probability, reference = softmax(actual, axis=1), softmax(expected, axis=1)
        same = np.array_equal(actual.argmax(1), expected.argmax(1))
    elif endpoint == 'pathology':
        thresholds = np.asarray(thresholds, np.float64)
        if thresholds.shape not in ((10,), expected.shape) or not np.isfinite(thresholds).all():
            raise ValueError('Frozen replay thresholds differ')
        probability, reference = expit(actual), expit(expected)
        same = np.array_equal(probability >= thresholds, reference >= thresholds)
    else: raise ValueError('Unknown replay endpoint')
    if not np.allclose(probability, reference, atol=1e-6, rtol=1e-5):
        raise ValueError('Source replay probability comparison failed')
    if not same: raise ValueError('Source replay threshold/class decision changed')
    return {'decisions_equal': True, 'maximum_absolute_logit_difference': float(np.max(np.abs(actual - expected))),
            'maximum_absolute_probability_difference': float(np.max(np.abs(probability - reference)))}


def assert_report_equal(actual, expected):
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected): raise ValueError('Report fields differ')
        for key in expected: assert_report_equal(actual[key], expected[key])
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected): raise ValueError('Report coverage differs')
        for a, b in zip(actual, expected): assert_report_equal(a, b)
    elif isinstance(expected, float):
        if not np.isfinite(actual) or not np.isclose(actual, expected, atol=1e-10, rtol=1e-10):
            raise ValueError('Recalculated metric differs')
    elif actual != expected: raise ValueError('Selection/report identity differs')


def validate_evaluation_scope(cohorts):
    if any(v.get('untouched_test') is not False for v in cohorts.values()):
        raise ValueError('Tracking/historical evaluation scope is not an untouched test')


def verify_allocation_receipt(receipt):
    if receipt.get('active') or not receipt.get('jobs'):
        raise ValueError('Incomplete allocation accounting')
    buckets = {}
    terminal = {'COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY', 'NODE_FAIL',
                'PREEMPTED', 'BOOT_FAIL', 'DEADLINE', 'REVOKED'}
    for job in receipt['jobs'].values():
        seconds = job.get('seconds')
        if (job.get('stage') not in BUCKETS or job.get('state') not in terminal
                or not isinstance(seconds, int) or isinstance(seconds, bool) or seconds < 0):
            raise ValueError('Unresolved allocation or invalid elapsed time')
        bucket = BUCKETS[job['stage']]
        buckets[bucket] = buckets.get(bucket, 0.) + seconds / 3600
    total = sum(buckets.values())
    if (not np.isclose(receipt['total'], total, atol=1e-12, rtol=0)
            or set(receipt['buckets']) != set(buckets)
            or any(not np.isclose(receipt['buckets'][k], v, atol=1e-12, rtol=0) for k, v in buckets.items())
            or total > 48 + 1e-12 or any(v > CEILINGS[k] + 1e-12 for k, v in buckets.items())):
        raise ValueError('Allocation totals or resource ceiling differ')
    return {'total_gpu_hours': total, 'buckets': buckets, 'allocations': len(receipt['jobs'])}


def verify_stage_snapshot(blob, expected_sha):
    contents = {}
    with tarfile.open(fileobj=io.BytesIO(blob), mode='r:gz') as archive:
        for member in archive.getmembers():
            safe_relative(member.name)
            if not member.isfile() or member.name in contents:
                raise ValueError('Unsafe or duplicate snapshot member')
            contents[member.name] = archive.extractfile(member).read()
    raw = contents.get('INPUT_SHA256.json', b'')
    if hashlib.sha256(raw).hexdigest() != expected_sha:
        raise ValueError('Execution snapshot identity differs')
    manifest = json.loads(raw)
    if not manifest or set(contents) != set(manifest) | {'INPUT_SHA256.json'}:
        raise ValueError('Execution snapshot coverage differs')
    for name, expected in manifest.items():
        safe_relative(name)
        if hashlib.sha256(contents[name]).hexdigest() != expected:
            raise ValueError('Execution snapshot source bytes differ')
    return {'files': len(manifest), 'code_sha256': expected_sha}


def assert_cohort_references(records, data):
    keys = data['keys'].tolist()
    expected = {r.key: r for r in records}
    if len(keys) != len(set(keys)) or len(keys) != len(records) or set(keys) != set(expected):
        raise ValueError('Cohort prediction coverage differs')
    ordered = [expected[k] for k in keys]
    references = {'y': np.asarray([r.rhythm_label for r in ordered]),
                  'original_y': np.asarray([r.original_rhythm_label if r.original_rhythm_label is not None else -1 for r in ordered]),
                  'path_targets': np.asarray([r.path_targets for r in ordered]),
                  'path_mask': np.asarray([r.path_mask for r in ordered])}
    if any(not np.array_equal(data[k], value) for k, value in references.items()):
        raise ValueError('Cohort reference/mask identity differs')
    for field, size in (('rhythm', 9), ('pathology', 10), ('quality', 1), ('biometrics', 3)):
        if data[field].shape != (len(records), size) or not np.isfinite(data[field]).all():
            raise ValueError('Cohort output shape/value differs')
    return ordered


def shared_patient_weights(rhythm_tokens, pathology_tokens, *, seed, replicates):
    tokens = sorted(set(rhythm_tokens.tolist()) | set(pathology_tokens.tolist()))
    if not tokens or replicates <= 0: raise ValueError('Empty patient bootstrap')
    lookup = {token: i for i, token in enumerate(tokens)}
    r = np.asarray([lookup[t] for t in rhythm_tokens]); p = np.asarray([lookup[t] for t in pathology_tokens])
    generator = np.random.default_rng(seed)
    for _ in range(replicates):
        counts = np.bincount(generator.integers(0, len(tokens), size=len(tokens)), minlength=len(tokens))
        yield counts[r], counts[p]


def assert_trial_coverage(teacher_results, student_results, *, adaptation, eligible):
    expected = {(e, c, 'heads', s) for e in ('rhythm', 'pathology') for c in CONTRACTS for s in SEEDS}
    expected |= {(e, c, 'adapt', s) for e, c in adaptation.items() for s in SEEDS}
    actual = [(r['endpoint'], r['contract'], r['stage'], r['seed']) for r in teacher_results]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError('Teacher trial coverage differs from registered comparisons')
    allowed = []
    for arm in ARMS:
        try: require_arm(arm, eligible)
        except ValueError: continue
        allowed.extend((arm, seed) for seed in SEEDS)
    actual = [(r['arm'], r['seed']) for r in student_results]
    if len(actual) != len(set(actual)) or set(actual) != set(allowed):
        raise ValueError('Student trial coverage differs from eligible arms')


def _verify_epoch_selection(records, controls, rows):
    warm = {e: prediction_metrics(load_predictions(controls['warm'][e]), e) for e in ('rhythm', 'pathology')}
    supervised = {r['seed']: r for r in records if r['arm'] == 'supervised'}
    for record in records:
        reference = warm if record['arm'] == 'supervised' else {
            e: prediction_metrics(load_predictions(supervised[record['seed']]['validation'][e]), e) for e in warm}
        best, utility = None, -1.
        if not 1 <= len(record['history']) <= 8: raise ValueError('Student epoch coverage differs')
        for epoch in record['history']:
            measures = {}
            for endpoint in warm:
                data = load_predictions(epoch['validation'][endpoint])
                assert_references([r for r in rows if r.role == endpoint + '_validation'], data, endpoint)
                measures[endpoint] = prediction_metrics(data, endpoint)
            assert_report_equal(measures, epoch['metrics'])
            decision = student_eligibility([measures] * 3, [reference] * 3)
            assert_report_equal(decision, epoch['decision'])
            current = (measures['rhythm']['macro_f1'] + measures['pathology']['macro_f1']) / 2
            if decision['eligible'] and current > utility: best, utility = epoch, current
        if best is None:
            if not record['fallback'] or record['epoch'] != -1 or record['checkpoint_sha256'] != STUDENT_SHA256:
                raise ValueError('Student warm fallback identity differs')
        elif record['fallback'] or record['epoch'] != best['epoch'] or record['checkpoint_sha256'] != best['checkpoint_sha256']:
            raise ValueError('Student epoch selection differs from validation utility')


def paired_validation_bootstrap(candidate, control):
    arrays = {name: {e: load_predictions(record[e]) for e in ('rhythm', 'pathology')}
              for name, record in (('candidate', candidate), ('control', control))}
    for endpoint in ('rhythm', 'pathology'):
        if any(not np.array_equal(arrays['candidate'][endpoint][key], arrays['control'][endpoint][key])
               for key in ('keys', 'labels', 'mask', 'patients', 'roles')):
            raise ValueError('Paired bootstrap identities differ')
    r, p = arrays['candidate']['rhythm'], arrays['candidate']['pathology']
    predictions = {}
    for name, values in arrays.items():
        thresholds = _validation_thresholds(None, values['pathology'])
        predictions[name] = {'rhythm': values['rhythm']['logits'].argmax(1),
                             'pathology': expit(values['pathology']['logits']) >= thresholds}
    samples = []
    for rw, pw in shared_patient_weights(r['patients'], p['patients'], seed=20261014, replicates=2000):
        scores = {}
        for name, values in predictions.items():
            scores[name] = np.asarray([
                f1_score(r['labels'], values['rhythm'], labels=list(range(9)), average='macro', sample_weight=rw, zero_division=0),
                f1_score(p['labels'][:, PRIMARY5_INDEX], values['pathology'][:, PRIMARY5_INDEX],
                         average='macro', sample_weight=pw, zero_division=0)])
        samples.append(scores['candidate'] - scores['control'])
    samples = np.asarray(samples)
    return {'scope': 'development validation conditional on frozen checkpoints', 'replicates': 2000, 'seed': 20261014,
            'shared_patient_sampling': True, 'unavailable_patient_ids_use_record_tokens': True,
            'rhythm_macro_f1_delta_interval_95': np.quantile(samples[:, 0], [.025, .5, .975]).tolist(),
            'pathology_primary5_macro_f1_delta_interval_95': np.quantile(samples[:, 1], [.025, .5, .975]).tolist(),
            'utility_delta_interval_95': np.quantile(samples.mean(1), [.025, .5, .975]).tolist()}


def _source_row(record):
    release = record.source_release or ('challenge-2021' if record.dataset in ('cpsc_2018', 'georgia') else record.dataset)
    return {'dataset': record.dataset, 'version': record.version, 'source_release': release,
            'doi': RELEASES[release][1], 'record_path': record.record_path,
            'signal_filename': record.signal_filename, 'header_sha256': record.header_sha256,
            'signal_sha256': record.signal_sha256}


def validated_source_release(record):
    row = _source_row(record)
    relative = safe_relative(record.record_path)
    if row['source_release'] == 'challenge-2021':
        if (len(relative.parts) != 4 or relative.parts[:2] != ('training', record.dataset)
                or dataset_for_record('training/' + record.dataset, relative.name) != record.dataset):
            raise ValueError('Historical Challenge family/path differs')
        release = 'challenge-2021'
    else:
        release = refinement_source_release(row)
    if row['source_release'] != release or record.version != RELEASES[release][0]:
        raise ValueError('Source release/version differs')
    signal = PurePosixPath(record.signal_filename)
    if signal.name != record.signal_filename or signal.stem != relative.name or signal.suffix not in ('.dat', '.mat'):
        raise ValueError('Source signal filename differs')
    return release


def ensure_source(record, data_root, *, offline):
    release = validated_source_release(record)
    root = (data_root / release / record.version).resolve()
    relative = safe_relative(record.record_path)
    header = root / (relative.as_posix() + '.hea')
    signal_relative = relative.parent / record.signal_filename
    signal = root / signal_relative
    if not header.resolve().is_relative_to(root) or not signal.resolve().is_relative_to(root):
        raise ValueError('Source cache path escapes release')
    url = 'https://physionet.org/files/' + release + '/' + record.version + '/'
    _ensure_file(header, url + quote(relative.as_posix(), safe='/') + '.hea', record.header_sha256, not offline)
    _ensure_file(signal, url + quote(signal_relative.as_posix(), safe='/'), record.signal_sha256, not offline)


def source_tensors(records, data_root, *, offline):
    unique = {}
    for r in records:
        if r.key in unique and (r.signal_sha256, r.student_tensor_sha256) != (
                unique[r.key].signal_sha256, unique[r.key].student_tensor_sha256):
            raise ValueError('Inconsistent source replay key identity')
        unique[r.key] = r
    def load(record):
        ensure_source(record, data_root, offline=offline)
        if record.role in ('historical', 'tracking'):
            tensor, flags = load_student_tensor(record, data_root)
        else:
            pair = load_pair(record, data_root, 'canonical_unfiltered', missing_policy='wfdb16_zero_fill')
            tensor, flags = pair.student, pair.quality_flags
        return record.key, tensor, list(flags)
    with ThreadPoolExecutor(max_workers=4) as pool:
        values = list(pool.map(load, unique.values()))
    return {k: x for k, x, _ in values}, {k: f for k, _, f in values if f}


def infer_records(checkpoint, sha256, records, tensors, *, device):
    model = load_exported_student(checkpoint, sha256).to(device).eval()
    streams = [[], [], [], []]
    for offset in range(0, len(records), 8):
        batch = torch.from_numpy(np.stack([tensors[r.key] for r in records[offset:offset + 8]])).to(device)
        with torch.no_grad(): output = model(batch)
        for i, values in enumerate(output): streams[i].append(values.cpu().numpy())
    return {'rhythm': np.concatenate(streams[0]), 'quality': np.concatenate(streams[1]),
            'biometrics': np.concatenate(streams[2]), 'pathology': np.concatenate(streams[3])}


def _validation_thresholds(record, data):
    metric = prediction_metrics(data, 'pathology')
    folds = patient_folds(data['patients']); threshold = np.empty(data['logits'].shape, np.float64)
    for fold in metric['crossfit']['folds']: threshold[folds == fold['fold']] = fold['thresholds']
    return threshold


def _rhythm_metrics(y, logits):
    prediction = logits.argmax(1)
    report = classification_report(y, prediction, labels=list(range(9)), target_names=list(PRIMARY9_CLASSES),
                                   output_dict=True, zero_division=0)
    return {'records': len(y), 'correct': int((prediction == y).sum()), 'accuracy': float(accuracy_score(y, prediction)),
            'macro_f1': report['macro avg']['f1-score'], 'weighted_f1': report['weighted avg']['f1-score'],
            'per_class': report}


def cohort_metrics(records, values, thresholds):
    y = np.asarray([r.rhythm_label for r in records])
    result = {'records': len(records), 'untouched_test': False, 'rhythm': _rhythm_metrics(y, values['rhythm'])}
    if all(r.original_rhythm_label is not None for r in records):
        result['original_reference_rhythm'] = _rhythm_metrics(np.asarray([r.original_rhythm_label for r in records]), values['rhythm'])
    mask = np.asarray([r.path_mask for r in records]); observed = mask.all(1)
    if observed.any():
        targets = np.asarray([r.path_targets for r in records])[observed]
        result['pathology'] = pathology_metrics(targets, expit(values['pathology'][observed]), thresholds)
    else: result['pathology'] = {'records': 0, 'observed': False}
    return result


def save_cohort(output, records, values, thresholds):
    output.mkdir(parents=True, exist_ok=False)
    metrics = cohort_metrics(records, values, thresholds)
    np.savez_compressed(output / 'predictions.npz', keys=np.asarray([r.key for r in records]),
        y=np.asarray([r.rhythm_label for r in records]), original_y=np.asarray([
            r.original_rhythm_label if r.original_rhythm_label is not None else -1 for r in records]),
        path_targets=np.asarray([r.path_targets for r in records]), path_mask=np.asarray([r.path_mask for r in records]),
        thresholds=np.asarray(thresholds), **values)
    with (output / 'predictions.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['record_key', 'role', 'reference', 'original_reference', 'prediction', 'signal_sha256', 'student_tensor_sha256']
                        + ['probability_' + c for c in PRIMARY9_CLASSES]
                        + ['path_target_' + c for c in PATHOLOGY_CLASSES]
                        + ['path_mask_' + c for c in PATHOLOGY_CLASSES]
                        + ['path_probability_' + c for c in PATHOLOGY_CLASSES])
        rp, pp = softmax(values['rhythm'], axis=1), expit(values['pathology'])
        for i, r in enumerate(records):
            writer.writerow([r.key, r.role, PRIMARY9_CLASSES[r.rhythm_label],
                PRIMARY9_CLASSES[r.original_rhythm_label] if r.original_rhythm_label is not None else '',
                PRIMARY9_CLASSES[int(rp[i].argmax())], r.signal_sha256, r.student_tensor_sha256,
                *rp[i].tolist(), *[v if known else '' for v, known in zip(r.path_targets, r.path_mask)],
                *r.path_mask, *pp[i].tolist()])
    _json(output / 'metrics.json', metrics)
    return metrics


def run_source_replay(bundle):
    device = allocation()
    producer = successful_stage(bundle, 'students')
    selected = json.loads((bundle / 'student_selection.json').read_text())
    assert_report_equal(public_paths(selected, bundle), producer['selection'])
    rows = paired_rows(bundle)
    validation = [r for r in rows if r.role in ('rhythm_validation', 'pathology_validation')]
    cohorts = {role: [r for r in rows if r.role == role] for role in ('tracking', 'historical')}
    tensors, flags = source_tensors(validation + cohorts['tracking'] + cohorts['historical'], bundle / 'raw', offline=True)
    output = bundle / 'source-replay'; output.mkdir(exist_ok=False)
    replayed, prediction_cache, student_records = [], {}, []
    for result_file in sorted((bundle / 'student-fits').glob('*/result.json')):
        if result_file.parent.name.endswith('-repeat'): continue
        relative = result_file.relative_to(bundle).as_posix()
        if file_sha(result_file) != producer['json_derivations'][relative]['original_sha256']:
            raise ValueError('Student result changed after selection')
        record = json.loads(result_file.read_text())
        student_records.append(record)
        signature = record['checkpoint_sha256']
        comparisons = {}
        for endpoint in ('rhythm', 'pathology'):
            data = load_predictions(record['validation'][endpoint])
            selected_rows = [r for r in validation if r.role == endpoint + '_validation']
            assert_references(selected_rows, data, endpoint)
            cache_key = (signature, endpoint)
            if cache_key not in prediction_cache:
                prediction_cache[cache_key] = infer_records(record['checkpoint'], signature, selected_rows, tensors, device=device)
            values = prediction_cache[cache_key]
            comparison = assert_replay(values[endpoint], data['logits'], endpoint=endpoint,
                thresholds=_validation_thresholds(record, data) if endpoint == 'pathology' else None)
            comparisons[endpoint] = comparison
        replayed.append({'checkpoint_sha256': signature, 'arm': record['arm'], 'seed': record['seed'], 'comparisons': comparisons})
    measured = {}
    models = {'selected': (Path(selected['checkpoint']), selected['checkpoint_sha256'], selected['thresholds'])}
    warm = json.loads((bundle / 'teacher-controls/controls.json').read_text())['warm']['pathology']
    models['warm'] = (bundle / 'models/student_start.pth', file_sha(bundle / 'models/student_start.pth'),
                      prediction_metrics(load_predictions(warm), 'pathology')['thresholds'])
    for name, (checkpoint, sha, thresholds) in models.items():
        measured[name] = {}
        for role, records in cohorts.items():
            first = infer_records(checkpoint, sha, records, tensors, device=device)
            second = infer_records(checkpoint, sha, records, tensors, device=device)
            if any(not np.array_equal(first[k], second[k]) for k in first):
                raise ValueError('Repeated source inference differs')
            measured[name][role] = save_cohort(output / name / role, records, first, thresholds)
        validate_evaluation_scope(measured[name])
    warm_descriptors = json.loads((bundle / 'teacher-controls/controls.json').read_text())['warm']
    candidate = next((r['validation'] for r in student_records if r['checkpoint_sha256'] == selected['checkpoint_sha256']), warm_descriptors)
    baseline = next(r['validation'] for r in student_records if r['arm'] == 'supervised' and r['seed'] == 20261014)
    if selected['selected_arm'] in ('warm_start', 'supervised'): baseline = warm_descriptors
    bootstrap = paired_validation_bootstrap(candidate, baseline)
    report = {'selection_producer': {k: producer[k] for k in IDENTITY_FIELDS},
              'selected_checkpoint_sha256': selected['checkpoint_sha256'], 'validation_models': replayed,
              'evaluated_models': {name: sha for name, (_, sha, _) in models.items()},
              'cohorts': measured, 'source_quality_flags': flags, 'repeated_inference_identical': True,
              'paired_validation_bootstrap': bootstrap,
              'source_inference_verified': True, 'selection_reopened': False}
    _json(output / 'report.json', report); seal_directory(output)
    return report


def _rooted_record(record, owner):
    def transform(value, key=None):
        if isinstance(value, dict): return {k: transform(v, k) for k, v in value.items()}
        if isinstance(value, list): return [transform(v, key) for v in value]
        if key in ('checkpoint', 'validation_npz') and isinstance(value, str):
            return str(owner / safe_relative(value))
        return value
    return transform(record)


def _check_repeats(bundle):
    receipts = []
    for namespace, owner, state_key in (('teacher-fits', 'teachers', 'delta'), ('student-fits', 'students', 'state_dict')):
        repeat_paths = sorted((bundle / owner / namespace).glob('*-repeat/result.json'))
        if not repeat_paths: raise ValueError('Missing repeated training evidence')
        for path in repeat_paths:
            original_path = path.parent.with_name(path.parent.name[:-len('-repeat')]) / 'result.json'
            a = _rooted_record(json.loads(original_path.read_text()), bundle / owner)
            b = _rooted_record(json.loads(path.read_text()), bundle / owner)
            pairs = [(a, b)] if owner == 'teachers' else list(zip(a['history'], b['history']))
            if owner == 'students' and (not pairs or len(a['history']) != len(b['history'])):
                raise ValueError('Repeated student epoch coverage differs')
            for first, second in pairs:
                x = torch.load(first['checkpoint'], map_location='cpu', weights_only=True)[state_key]
                y = torch.load(second['checkpoint'], map_location='cpu', weights_only=True)[state_key]
                if x.keys() != y.keys() or any(not torch.equal(x[k], y[k]) for k in x):
                    raise ValueError('Repeated fit parameters differ')
                endpoints = [first['endpoint']] if owner == 'teachers' else ['rhythm', 'pathology']
                for endpoint in endpoints:
                    one = first if owner == 'teachers' else first['validation'][endpoint]
                    two = second if owner == 'teachers' else second['validation'][endpoint]
                    if not np.array_equal(load_predictions(one)['logits'], load_predictions(two)['logits']):
                        raise ValueError('Repeated fit predictions differ')
            receipts.append({'namespace': namespace, 'trial': path.parent.name, 'comparisons': len(pairs)})
    return receipts


def verify_founder_bundle(bundle, *, data_root, offline, source_inference=True):
    bundle, data_root = Path(bundle).resolve(), Path(data_root).resolve()
    required = ('features', 'teachers', 'targets', 'students', 'replay')
    if any(not (bundle / stage / 'status.json').is_file() for stage in required):
        raise ValueError('Missing required completed study stage')
    verify_inventory(bundle, json.loads((bundle / 'BUNDLE_SHA256.json').read_text()), exact=True, seal_name='BUNDLE_SHA256.json')
    config = Path(__file__).resolve().parents[4] / 'configs/founder_transfer_2026-10-14.json'
    rows = [PairedRecord(**r) for r in verify_frozen_inputs(bundle, config)]
    statuses = {}
    receipts = json.loads((bundle / 'allocations.json').read_text())
    accounting = verify_allocation_receipt(receipts)
    for stage in required:
        status = json.loads((bundle / stage / 'status.json').read_text())
        job = receipts['jobs'][str(status['job_id'])]
        if job['state'] != 'COMPLETED' or job['exit_code'] != '0:0': raise ValueError('Incomplete stage allocation')
        statuses[stage] = verify_result(bundle / stage, status)['status']
        key = stage if status['attempt'] == 1 else stage + '-a' + str(status['attempt'])
        verify_stage_snapshot((bundle / 'snapshots' / (key + '-code.tar.gz')).read_bytes(), status['code_sha256'])
    controls = {'warm': {}}
    for role in ('task', 'warm'):
        for endpoint in ('rhythm', 'pathology'):
            path = bundle / 'features/teacher-controls' / role / endpoint / 'validation.npz'
            record = {'endpoint': endpoint, 'validation_npz': str(path), 'validation_npz_sha256': file_sha(path)}
            data = load_predictions(record)
            assert_references([r for r in rows if r.role == endpoint + '_validation'], data, endpoint)
            if role == 'task': controls[endpoint] = record
            else: controls['warm'][endpoint] = record
    teacher_results = []
    for path in sorted((bundle / 'teachers/teacher-fits').glob('*/result.json')):
        record = _rooted_record(json.loads(path.read_text()), bundle / 'teachers')
        data = load_predictions(record)
        assert_references([r for r in rows if r.role == record['endpoint'] + '_validation'], data, record['endpoint'])
        if not path.parent.name.endswith('-repeat'): teacher_results.append(record)
    qualification = qualify_teachers(teacher_results, controls)
    stored_qualification = {k: v for k, v in statuses['teachers']['qualification'].items() if k != 'repeated_fit_identical'}
    assert_report_equal(public_paths(qualification, bundle / 'teachers'), stored_qualification)
    if statuses['teachers']['qualification'].get('repeated_fit_identical') is not True:
        raise ValueError('Teacher repeated-training receipt missing')
    target_cache = bundle / 'targets/teacher-targets'
    target_metadata = json.loads((target_cache / 'metadata.json').read_text())
    assert_report_equal(target_metadata, statuses['targets']['target_metadata'])
    assert_report_equal(json.loads((target_cache / 'qualification.json').read_text()), statuses['teachers']['qualification'])
    assert_report_equal(target_metadata['qualification_producer'], {k: statuses['teachers'][k] for k in IDENTITY_FIELDS})
    join_teacher_targets([r for r in rows if r.role == 'train'], target_cache, checkpoint_root=bundle / 'teachers')
    student_results = []
    for path in sorted((bundle / 'students/student-fits').glob('*/result.json')):
        record = _rooted_record(json.loads(path.read_text()), bundle / 'students')
        for endpoint in ('rhythm', 'pathology'):
            data = load_predictions(record['validation'][endpoint])
            assert_references([r for r in rows if r.role == endpoint + '_validation'], data, endpoint)
        if not path.parent.name.endswith('-repeat'): student_results.append(record)
    head_qualification = qualify_teachers([r for r in teacher_results if r['stage'] == 'heads'], controls)
    assert_trial_coverage(teacher_results, student_results, adaptation=adaptation_candidates(head_qualification), eligible=qualification['eligible'])
    _verify_epoch_selection(student_results, controls, rows)
    warm = {'checkpoint': str(bundle / 'students/models/student_start.pth'),
            'checkpoint_sha256': file_sha(bundle / 'students/models/student_start.pth'),
            'metrics': {e: prediction_metrics(load_predictions(controls['warm'][e]), e) for e in ('rhythm', 'pathology')}}
    selection = select_students(student_results, warm)
    # The repeat/skipped receipts are verified separately, not selection inputs.
    stored = {k: v for k, v in statuses['students']['selection'].items() if k not in ('repeated_training_identical', 'skipped_arms')}
    assert_report_equal(public_paths(selection, bundle / 'students'), stored)
    replay = statuses['replay']['report']
    if replay['selected_checkpoint_sha256'] != selection['checkpoint_sha256'] or replay['selection_reopened']:
        raise ValueError('Replay used a stale selection')
    if replay['evaluated_models'] != {'selected': selection['checkpoint_sha256'], 'warm': warm['checkpoint_sha256']}:
        raise ValueError('Replay checkpoint inventory differs')
    if (set(replay['cohorts']) != {'selected', 'warm'} or not replay['repeated_inference_identical']
            or not replay['source_inference_verified']): raise ValueError('Incomplete source replay receipt')
    warm_thresholds = warm['metrics']['pathology']['thresholds']
    replay_packets = {}
    for name in ('selected', 'warm'):
        if set(replay['cohorts'][name]) != {'tracking', 'historical'}: raise ValueError('Missing replay cohort')
        validate_evaluation_scope(replay['cohorts'][name])
        for role in ('tracking', 'historical'):
            path = bundle / 'replay/source-replay' / name / role / 'predictions.npz'
            with np.load(path, allow_pickle=False) as archive: data = {k: archive[k] for k in archive.files}
            expected_rows = [r for r in rows if r.role == role]
            ordered = assert_cohort_references(expected_rows, data)
            thresholds = selection['thresholds'] if name == 'selected' else warm_thresholds
            if not np.array_equal(data['thresholds'], thresholds): raise ValueError('Frozen replay thresholds changed')
            assert_report_equal(cohort_metrics(ordered, data, thresholds), replay['cohorts'][name][role])
            replay_packets[(name, role)] = data
    expected_models = {(r['arm'], r['seed'], r['checkpoint_sha256']) for r in student_results}
    recorded_models = {(r['arm'], r['seed'], r['checkpoint_sha256']) for r in replay['validation_models']}
    if expected_models != recorded_models or len(recorded_models) != len(replay['validation_models']):
        raise ValueError('Incomplete validation source replay coverage')
    repeats = _check_repeats(bundle)
    candidate = next((r['validation'] for r in student_results if r['checkpoint_sha256'] == selection['checkpoint_sha256']), controls['warm'])
    baseline = next(r['validation'] for r in student_results if r['arm'] == 'supervised' and r['seed'] == 20261014)
    if selection['selected_arm'] in ('warm_start', 'supervised'): baseline = controls['warm']
    assert_report_equal(paired_validation_bootstrap(candidate, baseline), replay['paired_validation_bootstrap'])
    comparisons = {}; validation_comparisons = []
    if source_inference:
        selected_rows = [r for r in rows if r.role in ('tracking', 'historical', 'rhythm_validation', 'pathology_validation')]
        tensors, _ = source_tensors(selected_rows, data_root, offline=offline)
        prediction_cache = {}
        for record in student_results:
            comparison = {}
            for endpoint in ('rhythm', 'pathology'):
                data = load_predictions(record['validation'][endpoint])
                cohort = [r for r in selected_rows if r.role == endpoint + '_validation']
                key = (record['checkpoint_sha256'], endpoint)
                if key not in prediction_cache:
                    prediction_cache[key] = infer_records(record['checkpoint'], key[0], cohort, tensors, device='cpu')
                comparison[endpoint] = assert_replay(prediction_cache[key][endpoint], data['logits'], endpoint=endpoint,
                    thresholds=_validation_thresholds(record, data) if endpoint == 'pathology' else None)
            validation_comparisons.append({'arm': record['arm'], 'seed': record['seed'], 'comparisons': comparison})
        for name, model in (('selected', selection), ('warm', warm)):
            comparisons[name] = {}
            for role in ('tracking', 'historical'):
                cohort = [r for r in selected_rows if r.role == role]
                key = (model['checkpoint_sha256'], role)
                if key not in prediction_cache:
                    prediction_cache[key] = infer_records(model['checkpoint'], key[0], cohort, tensors, device='cpu')
                values = prediction_cache[key]; reference = replay_packets[(name, role)]
                thresholds = selection['thresholds'] if name == 'selected' else warm_thresholds
                comparisons[name][role] = {e: assert_replay(values[e], reference[e], endpoint=e,
                    thresholds=thresholds if e == 'pathology' else None) for e in ('rhythm', 'pathology')}
                assert_report_equal(cohort_metrics(cohort, values, thresholds), replay['cohorts'][name][role])
    return {'verified': True, 'source_inference_verified': source_inference, 'comparisons': comparisons,
            'allocation_accounting': accounting,
            'validation_comparisons': validation_comparisons, 'repeated_training_checks': repeats,
            'selection': public_paths(selection, bundle / 'students'), 'teacher_endpoints': sorted(qualification['eligible']),
            'scope': 'development validation plus previously examined tracking/historical cohorts'}
