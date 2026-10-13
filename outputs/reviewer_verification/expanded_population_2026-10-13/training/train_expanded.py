"""Continue the existing model with natural native pathology prevalence, on CEDIA."""
import json
import math
import os
import platform
import random
import socket
import time

import numpy as np
import torch
from torch.nn import functional as F

import contract as c
import baseline_trainer as b
from balanced_metrics import rhythm_training_weights, agreement_anchor_loss, asymmetric_pathology_loss, select_candidates

PROTOCOL = json.loads((c.ROOT / 'protocol.json').read_text())
b.PROTOCOL = PROTOCOL


def report(validation, values, reference, path_validation, path_values, baseline):
    rp = b.rhythm_prob(values)
    sources = c.source_metrics(validation, rp)
    score = float(np.mean([r['macro_f1'] for r in sources.values()]))
    metrics = c.metrics(validation['y'], rp)
    pp = b.path_prob(path_values)
    thresholds = b.calibrate_thresholds(path_validation['targets'], pp, role='validation')
    pathology = b.pathology_metrics(path_validation['targets'], pp, thresholds)
    errors = b.quality_bio_errors(values, reference)
    policy = PROTOCOL['selection']
    checks = {'protected_recalls': b.protected_recalls(sources, baseline['sources']),
              'rhythm_macro_preserved': metrics['macro_f1'] >= baseline['metrics']['macro_f1'] - policy['maximum_rhythm_macro_f1_drop'] - 1e-12,
              'rhythm_sources_preserved': score >= baseline['rhythm_score'] - policy['maximum_source_mean_f1_drop'] - 1e-12,
              'quality_biometrics_preserved': max(errors.values()) <= policy['maximum_quality_biometrics_nrmse']}
    return {'metrics': metrics, 'sources': sources, 'rhythm_score': score, 'pathology': pathology,
            'thresholds': thresholds.tolist(), 'joint_score': score + .5 * pathology['primary5_macro_f1'],
            'quality_biometrics_nrmse': errors, 'preservation_checks': checks, 'eligible': all(checks.values())}


def save_validation(out, name, val, values, pval, pvalues):
    np.savez_compressed(out / (name + '_validation.npz'), rhythm_probabilities=b.rhythm_prob(values),
                        pathology_probabilities=b.path_prob(pvalues), y=val['y'], dataset=val['dataset'], ecg_id=val['ecg_id'],
                        path_targets=pval['targets'], path_mask=pval['mask'], path_ecg_id=pval['ecg_id'],
                        quality=values[1], biometrics=values[2])


def run_trial(trial, epochs, train, extra, paths, val, pval, initial, references, val_ref, base, out):
    seed = PROTOCOL['optimization']['seed']
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    model = c.load_model(c.ROOT / 'start.pth').cuda()
    params = b.configure(model, trial['blocks'], True)
    expected = 19498 if trial['blocks'] == 0 else 778943
    if sum(p.numel() for p in params) != expected:
        raise ValueError('Trainable parameter contract differs')
    optimizer = torch.optim.AdamW(params, lr=trial['lr'], weight_decay=.01)
    x = torch.from_numpy(np.concatenate((train['x'], extra['x']))).cuda()
    yy = np.concatenate((train['y'], extra['y']))
    y = torch.from_numpy(yy).cuda()
    targets = torch.from_numpy(np.concatenate((paths['train_targets'], extra['targets']))).cuda()
    mask = torch.from_numpy(np.concatenate((paths['train_mask'], extra['mask']))).cuda()
    sources = np.concatenate((train['dataset'], np.full(len(extra['x']), 'ptb-xl')))
    weights = torch.from_numpy(rhythm_training_weights(yy, sources, role='train')).cuda()
    start_prob = torch.from_numpy(b.rhythm_prob(references)).cuda()
    teacher = torch.from_numpy(train['teacher']).cuda()
    auxiliary = tuple(torch.from_numpy(a).cuda() for a in references[1:3])
    scales = tuple(a.std(0, unbiased=False).clamp_min(.05) for a in auxiliary)
    label_weights = torch.ones(10, device='cuda'); label_weights[list(b.PRIMARY5_INDEX)] = 2
    records, history, state = [], [], None
    for epoch in range(1, epochs + 1):
        b.configure(model, trial['blocks'], True)
        factor = .1 + .9 * .5 * (1 + math.cos(math.pi * (epoch - 1) / PROTOCOL['optimization']['epochs']))
        for group in optimizer.param_groups:
            group['lr'] = trial['lr'] * factor
        order = torch.randperm(len(y), device='cuda')
        totals = []
        for start in range(0, len(order), 128):
            ids = order[start:start + 128]
            value = model(x[ids])
            path_loss = asymmetric_pathology_loss(value[3], targets[ids], mask[ids], gamma_negative=0, clip=0, label_weights=label_weights)
            rhythm_loss = path_loss * 0
            if trial['blocks']:
                known = y[ids] >= 0
                if known.any():
                    rid = ids[known]
                    rhythm_loss = (F.cross_entropy(value[0][known], y[rid], reduction='none') * weights[rid]).mean()
                    original = rid < len(train['y'])
                    if original.any():
                        oid = rid[original]
                        rhythm_loss = rhythm_loss + .15 * agreement_anchor_loss(value[0][known][original], teacher[oid], y[oid])
                rhythm_loss = rhythm_loss + .1 * agreement_anchor_loss(value[0], start_prob[ids], y[ids])
                rhythm_loss = rhythm_loss + sum((((v - t[ids]) / s) ** 2).mean() for v, t, s in zip(value[1:3], auxiliary, scales))
            loss = rhythm_loss + trial['path_weight'] * path_loss
            if not torch.isfinite(loss):
                raise ValueError('Non-finite loss')
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1); optimizer.step()
            totals.append([float(loss.detach()), float(rhythm_loss.detach()), float(path_loss.detach())])
        b.assert_frozen(model, initial, trial['blocks'], True)
        history.append({'epoch': epoch, 'lr': trial['lr'] * factor, 'mean_losses': np.mean(totals, 0).tolist()})
        if epoch not in PROTOCOL['optimization']['checkpoints'] and epoch != epochs:
            continue
        values, pvalues = b.outputs(model, val['x']), b.outputs(model, pval['x'])
        name = trial['name'] + f'_e{epoch:02d}'
        record = {'name': name, 'trial': trial, 'epoch': epoch, **report(val, values, val_ref, pval, pvalues, base)}
        records.append(record)
        state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(json.dumps({'candidate': name, 'rhythm_accuracy': record['metrics']['accuracy'],
                          'rhythm_macro_f1': record['metrics']['macro_f1'], 'pathology_macro_f1': record['pathology']['primary5_macro_f1'],
                          'eligible': record['eligible']}), flush=True)
        if out:
            torch.save({'state_dict': state, 'classes': c.CLASSES, 'pathology_classes': b.PATHOLOGY_CLASSES, 'parameters': c.PARAMETERS}, out / (name + '.pth'))
            save_validation(out, name, val, values, pval, pvalues)
    return records, history, state


def main():
    c.require_cedia_gpu()
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(2)
    for name, digest in json.loads((c.ROOT / 'INPUT_SHA256.json').read_text()).items():
        if c.digest(c.ROOT / name) != digest:
            raise ValueError('Input hash differs: ' + name)
    if c.digest(c.ROOT / 'start.pth') != PROTOCOL['start_sha256']:
        raise ValueError('Warm start differs')
    train, val, extra, pval = [dict(np.load(c.ROOT / name, allow_pickle=False)) for name in ('train.npz', 'validation.npz', 'path_extra.npz', 'path_validation.npz')]
    with np.load(c.ROOT / 'path_targets.npz', allow_pickle=False) as f:
        paths = {k: f[k] for k in ('train_targets', 'train_mask')}
    import hashlib
    for data, count in ((train, 8393), (val, 1476), (extra, 13984), (pval, 2027)):
        if data['x'].shape != (count, 6, 1250) or data['x'].dtype != np.float32 or not np.isfinite(data['x']).all():
            raise ValueError('Input tensor contract differs')
        for x, digest in zip(data['x'], data['tensor_sha256']):
            if hashlib.sha256(x.astype('<f4', copy=False).tobytes()).hexdigest() != str(digest):
                raise ValueError('Frozen source tensor hash differs')
    model = c.load_model(c.ROOT / 'start.pth').cuda()
    initial = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    tr, er, vr, pr = (b.outputs(model, data['x']) for data in (train, extra, val, pval))
    references = tuple(np.concatenate((a, z)) for a, z in zip(tr, er))
    rp = b.rhythm_prob(vr); sources = c.source_metrics(val, rp)
    thresholds = b.calibrate_thresholds(pval['targets'], b.path_prob(pr), role='validation')
    base = {'metrics': c.metrics(val['y'], rp), 'sources': sources,
            'rhythm_score': float(np.mean([r['macro_f1'] for r in sources.values()])),
            'pathology': b.pathology_metrics(pval['targets'], b.path_prob(pr), thresholds), 'thresholds': thresholds.tolist()}
    base['joint_score'] = base['rhythm_score'] + .5 * base['pathology']['primary5_macro_f1']
    out = c.ROOT / 'results'; out.mkdir(exist_ok=False)
    c.save(out / 'execution.json', {'job_id': os.environ['SLURM_JOB_ID'], 'hostname': socket.gethostname(),
           'python': platform.python_version(), 'torch': torch.__version__, 'numpy': np.__version__, 'cuda': torch.version.cuda,
           'gpu': torch.cuda.get_device_name(0), 'parameters': c.PARAMETERS, 'input_manifest_sha256': c.digest(c.ROOT / 'INPUT_SHA256.json'),
           'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())})
    save_validation(out, 'baseline', val, vr, pval, pr)
    records, histories = [], {}
    for trial in PROTOCOL['optimization']['trials']:
        reports, history, _ = run_trial(trial, 20, train, extra, paths, val, pval, initial, references, vr, base, out)
        records.extend(reports); histories[trial['name']] = history
    selected = select_candidates(records, base, role='validation')
    c.save(out / 'training_histories.json', histories)
    refits = {}
    for role, record in selected.items():
        if record is None:
            state = initial; refits[role] = {'needed': False}
        else:
            _, _, state = run_trial(record['trial'], record['epoch'], train, extra, paths, val, pval, initial, references, vr, base, None)
            old = torch.load(out / (record['name'] + '.pth'), map_location='cpu', weights_only=True)['state_dict']
            if not all(torch.equal(old[k], state[k]) for k in old):
                raise ValueError('Deterministic refit differs')
            refits[role] = {'needed': True, 'candidate': record['name'], 'all_state_tensors_bitwise_equal': True}
        torch.save({'state_dict': state, 'classes': c.CLASSES, 'pathology_classes': b.PATHOLOGY_CLASSES, 'parameters': c.PARAMETERS}, out / (role + '.pth'))
    c.save(out / 'selection.json', {'baseline': base, 'candidates': records, **selected, 'selection_data': 'validation only',
           'thresholds_fitted_on_validation_only': True, 'evaluation_labels_loaded_after_selection': True, 'references_changed': False,
           'checkpoints': {role: c.digest(out / (role + '.pth')) for role in selected}})
    c.save(out / 'refit.json', refits)
    # Followup remains unopened until weights and thresholds have been selected and saved.
    data = dict(np.load(c.ROOT / 'evaluation.npz', allow_pickle=False))
    with np.load(c.ROOT / 'path_targets.npz', allow_pickle=False) as f:
        ey, em = f['followup_targets'], f['followup_mask']
    for role, record in [('baseline', None), *selected.items()]:
        model = c.load_model(c.ROOT / 'start.pth' if role == 'baseline' else out / (role + '.pth')).cuda()
        b.export_predictions(out, role, model, data, ey, em, np.asarray((record or base)['thresholds']))
    c.save(out / 'SHA256SUMS.json', {p.name: c.digest(p) for p in sorted(out.iterdir()) if p.is_file()})
    print(json.dumps({'COMPLETED': True, **{role: r['name'] if r else 'start' for role, r in selected.items()}}), flush=True)


if __name__ == '__main__':
    main()
