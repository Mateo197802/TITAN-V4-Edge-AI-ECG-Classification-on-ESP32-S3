"""Evaluate aligned CNN weight interpolation with validation-only selection."""
import csv
import json
import os
from pathlib import Path
import socket
import time

import numpy as np
import torch

import contract as c


def outputs(model, data):
    model.eval()
    parts = [[], [], [], []]
    with torch.inference_mode():
        for start in range(0, len(data['x']), 128):
            value = model(torch.from_numpy(data['x'][start:start + 128]).cuda())
            for bucket, tensor in zip(parts, value, strict=True):
                bucket.append(tensor.cpu().numpy())
    return tuple(np.concatenate(value) for value in parts)


def probabilities(values):
    return torch.softmax(torch.from_numpy(values[0]), 1).numpy()


def auxiliary_errors(values, reference):
    return {name: float(np.sqrt(np.mean(((v - r) / np.maximum(r.std(0), .05)) ** 2)))
            for name, v, r in zip(('quality', 'biometrics', 'pathology'), values[1:], reference[1:], strict=True)}


def interpolate(base, donor, alpha, allowed):
    if set(base) != set(donor):
        raise ValueError('Donor architecture keys differ')
    state = {}
    for name, value in base.items():
        other = donor[name]
        if value.shape != other.shape or value.dtype != other.dtype:
            raise ValueError('Donor tensor shape/dtype differs: ' + name)
        if torch.equal(value, other):
            state[name] = value.clone()
        elif not name.startswith(tuple(allowed)):
            raise ValueError('Donor changes an unauthorized module: ' + name)
        elif value.is_floating_point():
            state[name] = value * (1 - alpha) + other * alpha
        else:
            state[name] = value.clone()
    return state


def main():
    c.require_cedia_gpu()
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(2)
    protocol = json.loads((c.ROOT / 'protocol.json').read_text())
    manifest = json.loads((c.ROOT / 'INPUT_SHA256.json').read_text())
    for name, expected in manifest.items():
        if c.digest(c.ROOT / name) != expected:
            raise ValueError('Input hash changed: ' + name)
    out = c.ROOT / 'results'
    out.mkdir(exist_ok=False)
    validation = dict(np.load(c.ROOT / 'validation.npz', allow_pickle=False))
    if len(validation['y']) != 1476:
        raise ValueError('Unexpected validation count')
    base_model = c.load_model(c.ROOT / 'base.pth').cuda()
    base_state = {k: v.cpu().clone() for k, v in base_model.state_dict().items()}
    reference = outputs(base_model, validation)
    base_p = probabilities(reference)
    base_score, _, base_sources = c.validation_score(validation, base_p, validation['gold'])
    policy = protocol['selection']
    candidates = []
    best = None
    selected_state = base_state
    for donor_name in protocol['donors']:
        donor = c.load_model(c.ROOT / donor_name).state_dict()
        for alpha in protocol['coefficients']:
            state = interpolate(base_state, donor, alpha, protocol['allowed_changed_prefixes'])
            model = c.load_model(c.ROOT / 'base.pth').cuda()
            model.load_state_dict(state, strict=True)
            value = outputs(model, validation)
            p = probabilities(value)
            score, preserve_gold, sources = c.validation_score(validation, p, validation['gold'])
            checks = []
            for source, metrics in sources.items():
                for label in policy['protected_classes']:
                    if metrics['support'][label] >= policy['minimum_class_support']:
                        checks.append(metrics['recall'][label] >= base_sources[source]['recall'][label] - policy['maximum_protected_recall_drop'] - 1e-12)
            aux = auxiliary_errors(value, reference)
            eligible = preserve_gold and all(checks) and max(aux.values()) <= policy['maximum_auxiliary_nrmse']
            name = donor_name.removesuffix('.pth') + '_alpha' + str(alpha)
            record = {'name': name, 'donor': donor_name, 'alpha': alpha, 'validation_score': score,
                      'eligible': bool(eligible), 'metrics': c.metrics(validation['y'], p),
                      'sources': sources, 'auxiliary_nrmse': aux}
            candidates.append(record)
            np.savez_compressed(out / (name + '_validation.npz'), probabilities=p, y=validation['y'],
                                dataset=validation['dataset'], ecg_id=validation['ecg_id'])
            print(json.dumps({k: record[k] for k in ('name', 'validation_score', 'eligible', 'auxiliary_nrmse')}), flush=True)
            if eligible and score >= base_score + policy['minimum_gain'] and (best is None or score > best['validation_score']):
                best, selected_state = record, state
    selected_name = 'base' if best is None else best['name']
    torch.save({'state_dict': selected_state, 'classes': c.CLASSES, 'parameters': c.PARAMETERS}, out / 'selected.pth')
    selection = {'selected': selected_name, 'base_validation_score': base_score,
                 'base_validation_metrics': c.metrics(validation['y'], base_p), 'candidates': candidates,
                 'selected_validation_metrics': c.metrics(validation['y'], base_p) if best is None else best['metrics'],
                 'selection_data': 'validation only', 'selected_checkpoint_sha256': c.digest(out / 'selected.pth'),
                 'labels_changed': False, 'gradient_updates': False}
    c.save(out / 'selection.json', selection)
    # Evaluation labels are read only after the validation selection has been persisted.
    data = dict(np.load(c.ROOT / 'evaluation.npz', allow_pickle=False))
    model = c.load_model(out / 'selected.pth').cuda()
    p = probabilities(outputs(model, data))
    repeat = probabilities(outputs(model, data))
    if not np.array_equal(p, repeat):
        raise ValueError('Repeated inference differed')
    rows = []
    for index in range(len(data['y'])):
        row = {'cohort': str(data['cohort'][index]), 'dataset': str(data['dataset'][index]),
               'ecg_id': str(data['ecg_id'][index]), 'reference': c.CLASSES[int(data['y'][index])],
               'prediction': c.CLASSES[int(p[index].argmax())]}
        row.update({f'prob_{label}': float(p[index,j]) for j,label in enumerate(c.CLASSES)})
        rows.append(row)
    with (out / 'followup_predictions.csv').open('x',newline='',encoding='utf-8') as handle:
        writer = csv.DictWriter(handle,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    cohorts = {str(cohort): c.metrics(data['y'][idx], p[idx]) for cohort in np.unique(data['cohort'])
               if (idx := data['cohort'] == cohort).any()}
    c.save(out / 'followup_metrics.json', {'records':len(rows), 'cohorts':cohorts, 'repeat_probabilities_bitwise_equal':True})
    c.save(out / 'execution.json', {'job_id':os.environ['SLURM_JOB_ID'], 'hostname':socket.gethostname(),
        'finished_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()), 'torch':torch.__version__,
        'numpy':np.__version__, 'cuda':torch.version.cuda, 'gpu':torch.cuda.get_device_name(0),
        'input_manifest_sha256':c.digest(c.ROOT/'INPUT_SHA256.json'), 'parameters':c.PARAMETERS,
        'selection_persisted_before_evaluation_labels_loaded':True})
    c.save(out/'SHA256SUMS.json',{path.name:c.digest(path) for path in sorted(out.iterdir()) if path.is_file()})
    print(json.dumps({'COMPLETED':selected_name, 'records':len(rows)}),flush=True)


if __name__ == '__main__':
    main()
