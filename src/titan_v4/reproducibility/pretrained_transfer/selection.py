"""Validation-only, three-seed endpoint qualification with explicit guards."""
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
from scipy.special import expit, softmax
from sklearn.metrics import average_precision_score, f1_score, recall_score

from .contracts import PRIMARY9_CLASSES
from .execution import file_sha
from titan_v4.reproducibility.joint_refinement import (
    PATHOLOGY_CLASSES, PRIMARY5_INDEX, calibrate_thresholds, pathology_metrics)
from titan_v4.reproducibility.supervision_control import patient_folds, crossfit_pathology

SEEDS = (20261014, 20261015, 20261016)


def endpoint_metrics(logits, labels, mask, sources, patients, *, endpoint, role):
    if role != 'validation':
        raise ValueError('Selection requires validation, never tracking')
    logits, labels, mask = np.asarray(logits), np.asarray(labels), np.asarray(mask)
    sources, patients = np.asarray(sources), np.asarray(patients)
    columns = 9 if endpoint == 'rhythm' else 10 if endpoint == 'pathology' else 0
    if (not columns or logits.shape != (len(labels), columns) or not len(labels)
            or sources.shape != (len(labels),) or patients.shape != (len(labels),)
            or not np.isfinite(logits).all()):
        raise ValueError('Endpoint validation shape/value differs')
    if endpoint == 'rhythm':
        if labels.ndim != 1 or mask.shape != labels.shape or not np.isin(labels, range(9)).all() or not (mask == 1).all():
            raise ValueError('Observed rhythm validation contract differs')
        probability = softmax(logits, axis=1); prediction = logits.argmax(1)
        onehot = np.eye(9)[labels.astype(int)]
        recalls = recall_score(labels, prediction, labels=list(range(9)), average=None, zero_division=0)
        per_ap = [float(average_precision_score(onehot[:, j], probability[:, j]))
                  if onehot[:, j].any() else 0. for j in range(9)]
        result = {'macro_f1': float(f1_score(labels, prediction, labels=list(range(9)), average='macro', zero_division=0)),
                  'accuracy': float(np.mean(prediction == labels)), 'macro_ap': float(np.mean(per_ap)),
                  'classes': {name: {'support': int((labels == j).sum()), 'recall': float(recalls[j]), 'ap': per_ap[j]}
                              for j, name in enumerate(PRIMARY9_CLASSES)}}
        result['sources'] = {str(s): {'support': int((sources == s).sum()), 'macro_f1':
            float(f1_score(labels[sources == s], prediction[sources == s], labels=list(range(9)),
                           average='macro', zero_division=0))} for s in sorted(set(sources.tolist()))}
        return result
    if labels.shape != logits.shape or mask.shape != labels.shape or not (mask == 1).all() or not np.isin(labels, [0, 1]).all():
        raise ValueError('Observed native pathology validation shape/mask differs')
    probability = expit(logits)
    folds = patient_folds(patients)
    crossfit = crossfit_pathology(labels, probability, folds, role='validation')
    prediction = np.zeros(labels.shape, bool)
    for record in crossfit['folds']:
        held = folds == record['fold']
        prediction[held] = probability[held] >= np.asarray(record['thresholds'])
    tp = (prediction & (labels == 1)).sum(0); support = labels.sum(0)
    recalls = tp / np.maximum(1, support)
    per_ap = [float(average_precision_score(labels[:, j], probability[:, j])) for j in range(10)]
    thresholds = calibrate_thresholds(labels, probability, role='validation')
    result = {'macro_f1': crossfit['primary5_macro_f1'],
              'accuracy': float((prediction[:, PRIMARY5_INDEX] == labels[:, PRIMARY5_INDEX]).mean()),
              'macro_ap': float(np.mean(np.asarray(per_ap)[list(PRIMARY5_INDEX)])),
              'thresholds': thresholds.tolist(), 'crossfit': crossfit,
              'full_calibration': pathology_metrics(labels, probability, thresholds),
              'classes': {name: {'support': int(support[j]), 'recall': float(recalls[j]), 'ap': per_ap[j],
                                'f1': crossfit['per_label_f1'][j]} for j, name in enumerate(PATHOLOGY_CLASSES)}}
    result['sources'] = {str(s): {'support': int((sources == s).sum()), 'macro_f1':
        float(f1_score(labels[sources == s][:, PRIMARY5_INDEX], prediction[sources == s][:, PRIMARY5_INDEX],
                       average='macro', zero_division=0))} for s in sorted(set(sources.tolist()))}
    return result


def eligibility(metrics, control, endpoint):
    if len(metrics) != 3:
        raise ValueError('Exactly three seeds required')
    mean = {field: float(np.mean([r[field] for r in metrics])) for field in ('macro_f1', 'accuracy', 'macro_ap')}
    if not all(np.isfinite(v) and 0 <= v <= 1 for v in mean.values()):
        raise ValueError('Invalid selection metric')
    reasons = []
    if mean['macro_f1'] + 1e-12 < control['macro_f1'] + .01:
        reasons.append('endpoint_macro_f1_gain_below_0.01')
    if endpoint == 'rhythm' and mean['accuracy'] + 1e-12 < control['accuracy'] - .005:
        reasons.append('rhythm_accuracy_regression')
    if endpoint == 'pathology' and mean['macro_ap'] + 1e-12 < control['macro_ap'] + .005:
        reasons.append('pathology_macro_ap_gain_below_0.005')
    for name, base in control['classes'].items():
        values = [r['classes'][name] for r in metrics]
        if any(v['support'] != base['support'] for v in values):
            raise ValueError('Selection class support differs')
        if base['support'] >= 20 and np.mean([v['recall'] for v in values]) + 1e-12 < base['recall'] - .05:
            reasons.append('class_recall:' + name)
    for name, base in control['sources'].items():
        values = [r['sources'][name] for r in metrics]
        if any(v['support'] != base['support'] for v in values):
            raise ValueError('Selection source support differs')
        if base['support'] >= 100 and np.mean([v['macro_f1'] for v in values]) + 1e-12 < base['macro_f1'] - .02:
            reasons.append('source_macro_f1:' + name)
    return {'eligible': not reasons, 'mean': mean, 'reasons': reasons}


def load_predictions(record):
    path = Path(record['validation_npz'])
    if file_sha(path) != record['validation_npz_sha256']:
        raise ValueError('Validation prediction bytes differ')
    with np.load(path, allow_pickle=False) as archive:
        data = {k: archive[k] for k in archive.files}
    expected = PRIMARY9_CLASSES if record['endpoint'] == 'rhythm' else PATHOLOGY_CLASSES
    if tuple(data['classes'].tolist()) != tuple(expected) or set(data['roles'].tolist()) != {record['endpoint'] + '_validation'}:
        raise ValueError('Validation class order/role differs')
    if len(data['keys']) != len(set(data['keys'].tolist())):
        raise ValueError('Duplicate validation record keys')
    return data


def prediction_metrics(data, endpoint):
    return endpoint_metrics(data['logits'], data['labels'], data['mask'], data['sources'],
                            data['patients'], endpoint=endpoint, role='validation')


def qualify_teachers(results, controls):
    groups = defaultdict(list)
    for r in results:
        if r.get('scope') != 'validation' or r.get('seed') not in SEEDS or r.get('endpoint') not in ('rhythm', 'pathology'):
            raise ValueError('Teacher qualification is validation-only with registered seeds')
        groups[(r['endpoint'], r['contract'], r['stage'])].append(r)
    decisions, chosen = [], {}
    for (endpoint, contract, stage), records in sorted(groups.items()):
        records.sort(key=lambda r: r['seed'])
        if [r['seed'] for r in records] != list(SEEDS):
            raise ValueError('Teacher group must retain all three distinct seeds')
        arrays = [load_predictions(r) for r in records]
        reference = load_predictions(controls[endpoint])
        for data in arrays:
            if any(not np.array_equal(data[k], reference[k]) for k in ('keys', 'labels', 'mask', 'sources', 'patients', 'roles', 'classes')):
                raise ValueError('Teacher and control validation identities differ')
        baseline = prediction_metrics(reference, endpoint)
        measures = [prediction_metrics(data, endpoint) for data in arrays]
        decision = eligibility(measures, baseline, endpoint)
        ensemble = dict(reference, logits=np.mean(np.stack([a['logits'] for a in arrays]), axis=0))
        combined = prediction_metrics(ensemble, endpoint)
        ensemble_decision = eligibility([combined] * 3, baseline, endpoint)
        decision.update(endpoint=endpoint, contract=contract, stage=stage,
                        seed_metrics=measures, control=baseline, ensemble=combined,
                        ensemble_eligible=ensemble_decision['eligible'],
                        ensemble_reasons=ensemble_decision['reasons'])
        decision['eligible'] = decision['eligible'] and ensemble_decision['eligible']
        if 'warm' in controls:
            warm_arrays = load_predictions(controls['warm'][endpoint])
            if any(not np.array_equal(warm_arrays[k], reference[k]) for k in ('keys', 'labels', 'mask', 'sources', 'patients')):
                raise ValueError('Warm control validation identity differs')
            warm_metric = prediction_metrics(warm_arrays, endpoint)
            warm_mean = eligibility(measures, warm_metric, endpoint)
            warm_ensemble = eligibility([combined] * 3, warm_metric, endpoint)
            decision['warm_control'] = warm_metric
            decision['warm_comparison'] = warm_mean
            decision['warm_ensemble_comparison'] = warm_ensemble
            decision['eligible'] = decision['eligible'] and warm_mean['eligible'] and warm_ensemble['eligible']
        checkpoints = []
        for r in records:
            if file_sha(r['checkpoint']) != r['checkpoint_sha256']:
                raise ValueError('Teacher checkpoint bytes differ')
            checkpoints.append({'seed': r['seed'], 'path': r['checkpoint'], 'sha256': r['checkpoint_sha256']})
        decision['checkpoints'] = checkpoints
        decisions.append(decision)
        if decision['eligible'] and (endpoint not in chosen or
                (decision['mean']['macro_f1'], contract, stage) >
                (chosen[endpoint]['mean']['macro_f1'], chosen[endpoint]['contract'], chosen[endpoint]['stage'])):
            chosen[endpoint] = decision
    return {'scope': 'validation', 'decisions': decisions, 'eligible': chosen,
            'seeds': list(SEEDS), 'selection_uses_tracking': False}


def mean_endpoint(metrics):
    result = {name: float(np.mean([m[name] for m in metrics])) for name in ('macro_f1', 'accuracy', 'macro_ap')}
    for group in ('classes', 'sources'):
        if any(set(m[group]) != set(metrics[0][group]) for m in metrics):
            raise ValueError('Endpoint strata differ')
        result[group] = {}
        for name, reference in metrics[0][group].items():
            values = [m[group][name] for m in metrics]
            if any(v['support'] != reference['support'] for v in values):
                raise ValueError('Endpoint support differs')
            averaged = {'support': reference['support']}
            for field in ('recall', 'ap', 'f1', 'macro_f1'):
                if all(field in v for v in values): averaged[field] = float(np.mean([v[field] for v in values]))
            result[group][name] = averaged
    return result


def student_eligibility(measures, baseline):
    if len(measures) != 3 or len(baseline) != 3:
        raise ValueError('Student comparisons require all three seeds')
    mean = {e: mean_endpoint([m[e] for m in measures]) for e in ('rhythm', 'pathology')}
    control = {e: mean_endpoint([m[e] for m in baseline]) for e in ('rhythm', 'pathology')}
    delta = {e: mean[e]['macro_f1'] - control[e]['macro_f1'] for e in mean}
    reasons = []
    if not ((delta['rhythm'] >= .01 - 1e-12 and delta['pathology'] >= -.005 - 1e-12)
            or (delta['pathology'] >= .01 - 1e-12 and delta['rhythm'] >= -.005 - 1e-12)):
        reasons.append('no_guarded_endpoint_gain')
    if mean['rhythm']['accuracy'] + 1e-12 < control['rhythm']['accuracy'] - .005:
        reasons.append('rhythm_accuracy_regression')
    for endpoint in mean:
        for group, field, minimum, allowed in (('classes', 'recall', 20, .05), ('sources', 'macro_f1', 100, .02)):
            if set(mean[endpoint][group]) != set(control[endpoint][group]):
                raise ValueError('Student comparison strata differ')
            for name, base in control[endpoint][group].items():
                candidate = mean[endpoint][group][name]
                if candidate['support'] != base['support']:
                    raise ValueError('Student comparison support differs')
                if base['support'] >= minimum and candidate[field] + 1e-12 < base[field] - allowed:
                    reasons.append(endpoint + ':' + group + ':' + name)
    return {'eligible': not reasons, 'mean': mean, 'baseline': control, 'deltas': delta, 'reasons': reasons}


def select_students(results, controls):
    from .students import ARMS, load_exported_student
    groups = defaultdict(list)
    for record in results:
        if record.get('scope') != 'validation' or record.get('arm') not in ARMS or record.get('seed') not in SEEDS:
            raise ValueError('Student selection is validation-only with registered arms/seeds')
        groups[record['arm']].append(record)
    if 'supervised' not in groups:
        raise ValueError('Missing supervised three-seed comparison')
    measured = {}
    arrays = {}
    for arm, records in groups.items():
        records.sort(key=lambda r: r['seed'])
        if [r['seed'] for r in records] != list(SEEDS):
            raise ValueError('Missing or duplicate student seed')
        measured[arm] = []
        for record in records:
            load_exported_student(Path(record['checkpoint']), record['checkpoint_sha256'])
            data = {e: load_predictions(record['validation'][e]) for e in ('rhythm', 'pathology')}
            arrays[(arm, record['seed'])] = data
            measured[arm].append({e: prediction_metrics(data[e], e) for e in data})
    for (arm, seed), data in arrays.items():
        reference = arrays[('supervised', seed)]
        for endpoint in data:
            if any(not np.array_equal(data[endpoint][k], reference[endpoint][k])
                   for k in ('keys', 'labels', 'mask', 'sources', 'patients', 'roles', 'classes')):
                raise ValueError('Student comparison validation identities differ')
    table, candidates = [], []
    for arm in sorted(groups):
        decision = student_eligibility(measured[arm], measured['supervised'])
        fixed = student_eligibility([measured[arm][0]] * 3, [measured['supervised'][0]] * 3)
        decision.update(arm=arm, fixed_seed=SEEDS[0], fixed_seed_eligible=fixed['eligible'],
                        fixed_seed_reasons=fixed['reasons'], seed_metrics=measured[arm],
                        total_fit_seconds=sum(r['elapsed_seconds'] for r in groups[arm]))
        table.append(decision)
        if decision['eligible'] and fixed['eligible']:
            candidates.append((decision, groups[arm][0]))
    if candidates:
        decision, chosen = min(candidates, key=lambda pair: (
            -(pair[0]['mean']['rhythm']['macro_f1'] + pair[0]['mean']['pathology']['macro_f1']) / 2,
            pair[0]['total_fit_seconds'], pair[0]['arm']))
        metrics = measured[decision['arm']][0]
        return {'scope': 'validation', 'selected_arm': decision['arm'], 'seed': SEEDS[0], 'promoted': True,
                'checkpoint': chosen['checkpoint'], 'checkpoint_sha256': chosen['checkpoint_sha256'],
                'thresholds': prediction_metrics(arrays[(decision['arm'], SEEDS[0])]['pathology'], 'pathology')['thresholds'],
                'metrics': metrics, 'comparisons': table, 'selection_uses_tracking': False}
    continuation = student_eligibility(measured['supervised'], [controls['metrics']] * 3)
    continuation_fixed = student_eligibility([measured['supervised'][0]] * 3, [controls['metrics']] * 3)
    if continuation['eligible'] and continuation_fixed['eligible']:
        chosen = groups['supervised'][0]
        return {'scope': 'validation', 'selected_arm': 'supervised', 'seed': SEEDS[0], 'promoted': True,
                'checkpoint': chosen['checkpoint'], 'checkpoint_sha256': chosen['checkpoint_sha256'],
                'thresholds': measured['supervised'][0]['pathology']['thresholds'],
                'metrics': measured['supervised'][0], 'comparisons': table, 'selection_uses_tracking': False,
                'supervised_vs_warm': continuation, 'supervised_fixed_vs_warm': continuation_fixed,
                'transfer_improvement': False}
    load_exported_student(Path(controls['checkpoint']), controls['checkpoint_sha256'])
    return {'scope': 'validation', 'selected_arm': 'warm_start', 'seed': SEEDS[0], 'promoted': False,
            'checkpoint': controls['checkpoint'], 'checkpoint_sha256': controls['checkpoint_sha256'],
            'thresholds': controls['metrics']['pathology']['thresholds'], 'metrics': controls['metrics'],
            'comparisons': table, 'selection_uses_tracking': False}
