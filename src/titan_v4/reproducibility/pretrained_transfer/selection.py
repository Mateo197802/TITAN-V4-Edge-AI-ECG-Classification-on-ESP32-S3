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
