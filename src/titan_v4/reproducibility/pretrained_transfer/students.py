"""Controlled continuation of the unchanged four-head TITAN student."""
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from titan_v4.models.titan_v4_net import TitanV4Lite
from titan_v4.reproducibility.joint_refinement import masked_pathology_loss
from titan_v4.reproducibility.supervision_control import source_masses, fixed_source_weights
from .contracts import STUDENT_SHA256, PRIMARY9_CLASSES, PATHOLOGY_CLASSES, SEEDS
from .execution import file_sha, verify_inventory, successful_stage, IDENTITY_FIELDS
from .teachers import allocation, paired_rows, open_cache, save_predictions, seal_directory, _json
from .distillation import join_teacher_targets, transfer_losses
from .selection import student_eligibility, prediction_metrics, load_predictions, select_students

ARMS = ('supervised', 'supervised_bn', 'rhythm_transfer', 'pathology_transfer', 'joint_transfer')
TRAINABLE = ('res_layers.3.', 'res_layers.4.', 'head_rhythm.', 'head_pathology.')


def _architecture(model):
    signature = []
    for name, module in model.named_modules():
        attributes = {}
        for field in ('in_channels', 'out_channels', 'kernel_size', 'stride', 'padding', 'dilation',
                      'groups', 'in_features', 'out_features', 'num_features', 'eps', 'momentum', 'p', 'inplace'):
            if hasattr(module, field): attributes[field] = getattr(module, field)
        signature.append((name, type(module).__name__, attributes, module.extra_repr()))
    return signature


def assert_deployment_contract(model):
    with torch.random.fork_rng(devices=[]):
        expected = TitanV4Lite(in_channels=6, num_rhythm=9, num_pathology=10)
    if (sum(p.numel() for p in model.parameters()) != 1143751 or model.morphology_dim != 0
            or _architecture(model) != _architecture(expected)):
        raise ValueError('Student parameter/head/operation deployment contract differs')


def configure_student(model):
    assert_deployment_contract(model)
    model.eval()
    model.res_layers[3].train(); model.res_layers[4].train()
    model.head_rhythm.train(); model.head_pathology.train()
    for name, parameter in model.named_parameters(): parameter.requires_grad_(name.startswith(TRAINABLE))
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm): module.eval()


def assert_student_preserved(model, initial, *, arm):
    assert_deployment_contract(model)
    for name, value in model.state_dict().items():
        buffer = name.endswith(('running_mean', 'running_var', 'num_batches_tracked'))
        allowed = (not buffer and name.startswith(TRAINABLE)) or (buffer and arm == 'supervised_bn')
        if not allowed and (name not in initial or not torch.equal(value.cpu(), initial[name])):
            raise ValueError('Student frozen parameter/buffer changed: ' + name)


def load_exported_student(path, expected_sha):
    if file_sha(path) != expected_sha: raise ValueError('Student checkpoint hash differs')
    data = torch.load(path, weights_only=True, map_location='cpu')
    if (tuple(data.get('classes', ())) != tuple(PRIMARY9_CLASSES)
            or tuple(data.get('pathology_classes', ())) != tuple(PATHOLOGY_CLASSES)
            or data.get('parameters') != 1143751):
        raise ValueError('Student exported head metadata differs')
    with torch.random.fork_rng(devices=[]): model = TitanV4Lite(in_channels=6, num_rhythm=9, num_pathology=10)
    expected = model.state_dict(); state = data['state_dict']
    if state.keys() != expected.keys() or any(
            not isinstance(value, torch.Tensor) or value.shape != expected[name].shape
            or value.dtype != expected[name].dtype or not torch.isfinite(value).all() for name, value in state.items()):
        raise ValueError('Student checkpoint state shape/dtype/value differs')
    model.load_state_dict(data['state_dict'], strict=True)
    if any(not torch.isfinite(v).all() for v in model.state_dict().values()):
        raise ValueError('Nonfinite student state')
    assert_deployment_contract(model)
    return model


def load_student(path):
    if file_sha(path) != STUDENT_SHA256:
        raise ValueError('Pinned student warm-start hash differs')
    return load_exported_student(path, STUDENT_SHA256)


def require_arm(arm, teachers):
    if arm not in ARMS: raise ValueError('Unknown student arm')
    need = {'rhythm_transfer': {'rhythm'}, 'pathology_transfer': {'pathology'},
            'joint_transfer': {'rhythm', 'pathology'}}.get(arm, set())
    if not need <= set(teachers): raise ValueError('Student arm has an ineligible teacher')


def supervised_losses(values, y, path_targets, path_mask, weights, positive_weights):
    r, p = values['rhythm'], values['pathology']
    if (r.ndim != 2 or r.shape[1] != 9 or p.shape != (len(r), 10) or y.shape != (len(r),)
            or y.dtype not in (torch.int64, torch.int32) or not ((y >= -1) & (y < 9)).all()
            or weights.shape != y.shape or not torch.isfinite(weights).all() or (weights < 0).any()
            or not torch.isfinite(r).all() or not torch.isfinite(p).all()):
        raise ValueError('Student supervised contract differs')
    known = y >= 0
    rhythm = r.sum() * 0
    if known.any(): rhythm = (F.cross_entropy(r[known], y[known].long(), reduction='none') * weights[known]).sum() / known.sum()
    pathology = masked_pathology_loss(p, path_targets, path_mask, positive_weights)
    return {'rhythm_loss': rhythm, 'pathology_loss': pathology}


def balanced_bn_indices(records, seed):
    if not records or any(r.role != 'train' for r in records):
        raise ValueError('BN recalibration must be train-only')
    groups = {}
    for i, r in enumerate(records): groups.setdefault((r.dataset, r.rhythm_label), []).append(i)
    generator = np.random.default_rng(seed + 1000)
    selected = np.concatenate([generator.choice(groups[key], size=128, replace=True) for key in sorted(groups)])
    return generator.permutation(selected)


def _producer_artifacts(bundle, producer):
    key = producer['stage'] + ('-a' + str(producer['attempt']) if producer['attempt'] != 1 else '')
    return bundle / 'stages' / key / 'results'


def student_data(bundle):
    feature = successful_stage(bundle, 'features')
    target = successful_stage(bundle, 'targets')
    cache = bundle / 'feature-cache'; targets = bundle / 'teacher-targets'
    if file_sha(cache / 'SHA256SUMS.json') != file_sha(_producer_artifacts(bundle, feature) / 'feature-cache/SHA256SUMS.json'):
        raise ValueError('Student cache differs from sealed feature producer')
    if file_sha(targets / 'SHA256SUMS.json') != file_sha(_producer_artifacts(bundle, target) / 'teacher-targets/SHA256SUMS.json'):
        raise ValueError('Student targets differ from sealed target producer')
    inventory = json.loads((cache / 'SHA256SUMS.json').read_text())
    verify_inventory(cache, {k: inventory[k] for k in ('student.npy', 'keys.json', 'cache.json')})
    metadata = json.loads((targets / 'metadata.json').read_text())
    if metadata != target['target_metadata']:
        raise ValueError('Target metadata differs from producer')
    rows = paired_rows(bundle)
    train = [r for r in rows if r.role == 'train']
    joined = join_teacher_targets(train, targets)
    keys = json.loads((cache / 'keys.json').read_text())
    if len(keys) != len(set(keys)):
        raise ValueError('Duplicate student input keys')
    x = np.load(cache / 'student.npy', mmap_mode='r', allow_pickle=False)
    if x.shape != (len(keys), 6, 1250) or x.dtype != np.float32:
        raise ValueError('Student six-lead sample contract differs')
    return rows, train, x, {k: i for i, k in enumerate(keys)}, joined, metadata


def _evaluate(model, rows, x, lookup, output, *, baseline_aux=None):
    model.eval(); result, metrics, auxiliary = {}, {}, {}
    for endpoint, index in (('rhythm', 0), ('pathology', 3)):
        selected = [r for r in rows if r.role == endpoint + '_validation']
        logits, quality, biometrics = [], [], []
        for offset in range(0, len(selected), 8):
            ids = [lookup[r.key] for r in selected[offset:offset + 8]]
            batch = torch.from_numpy(np.asarray(x[ids]).copy()).cuda()
            with torch.no_grad(): values = model(batch)
            logits.append(values[index].cpu().numpy())
            quality.append(values[1].cpu().numpy()); biometrics.append(values[2].cpu().numpy())
        root = output / endpoint; root.mkdir(parents=True)
        record = save_predictions(root, selected, np.concatenate(logits), endpoint)
        result[endpoint] = record; metrics[endpoint] = record['metrics']
        auxiliary[endpoint] = {'quality': np.concatenate(quality), 'biometrics': np.concatenate(biometrics)}
    drift = {}
    if baseline_aux is not None:
        for endpoint in auxiliary:
            drift[endpoint] = {name + '_rmse': np.sqrt(np.mean((value - baseline_aux[endpoint][name])**2, axis=0)).tolist()
                               for name, value in auxiliary[endpoint].items()}
    return {'validation': result, 'metrics': metrics, 'auxiliary_output_drift': drift}, auxiliary


def recalibrate_bn(model, x, indices):
    saved = {name: value.detach().clone() for name, value in model.named_buffers()}
    momentum = {}
    model.eval()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            momentum[name] = module.momentum
            module.reset_running_stats(); module.momentum = None; module.train()
    with torch.no_grad():
        for offset in range(0, len(indices), 128):
            batch = torch.from_numpy(np.asarray(x[indices[offset:offset + 128]]).copy()).cuda()
            model(batch)
    for name, module in model.named_modules():
        if name in momentum: module.momentum = momentum[name]
    after = {name: value.detach().clone() for name, value in model.named_buffers()}
    diff = {name: {'before_sha256': hashlib.sha256(saved[name].cpu().numpy().tobytes()).hexdigest(),
                   'after_sha256': hashlib.sha256(value.cpu().numpy().tobytes()).hexdigest(),
                   'changed': not torch.equal(saved[name], value)} for name, value in after.items()}
    return saved, diff


def _save_student(path, model, seed, arm, epoch):
    assert_deployment_contract(model)
    torch.save({'state_dict': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                'classes': list(PRIMARY9_CLASSES), 'pathology_classes': list(PATHOLOGY_CLASSES),
                'parameters': 1143751, 'warm_start_sha256': STUDENT_SHA256,
                'seed': seed, 'arm': arm, 'epoch': epoch,
                'input_contract': {'leads': ['I', 'II', 'III', 'aVR', 'aVL', 'aVF'], 'hz': 125, 'samples': 1250}}, path)


def training_snapshot(model, optimizer, scheduler, *, epoch, seed):
    return {'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
            'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(),
            'training_buffers': {name: value.detach().cpu().clone() for name, value in model.named_buffers()},
            'permutation_seed': seed + epoch, 'next_permutation_seed': seed + epoch + 1, 'epoch': epoch}


def fit_student(bundle: Path, *, arm: str, seed: int, repeat=False) -> dict:
    device = allocation(seed)
    if seed not in SEEDS: raise ValueError('Unregistered student seed')
    rows, train, x, lookup, targets, metadata = student_data(bundle)
    require_arm(arm, metadata['teachers'])
    trial = arm + '-' + str(seed) + ('-repeat' if repeat else '')
    from .execution import fit_identity, prepare_fit_trial
    identity = fit_identity('student', arm=arm, seed=seed, repeat=repeat)
    output = bundle / 'student-fits' / trial
    completed = prepare_fit_trial(output, identity)
    if completed is not None: return completed
    output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    model = load_student(bundle / 'models/student_start.pth').to(device)
    initial = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    warm, baseline_aux = _evaluate(model, rows, x, lookup, output / 'warm')
    if arm == 'supervised':
        reference = warm['metrics']
    else:
        base = json.loads((bundle / 'student-fits' / ('supervised-' + str(seed)) / 'result.json').read_text())
        reference = {e: prediction_metrics(load_predictions(base['validation'][e]), e) for e in ('rhythm', 'pathology')}
    configure_student(model)
    backbone = [p for name, p in model.named_parameters() if p.requires_grad and name.startswith('res_layers.')]
    heads = [p for name, p in model.named_parameters() if p.requires_grad and name.startswith(('head_rhythm.', 'head_pathology.'))]
    optimizer = torch.optim.AdamW([{'params': backbone, 'lr': 5e-5}, {'params': heads, 'lr': 1e-3}], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
    y = targets['rhythm_labels']; py = targets['path_targets']; mask = targets['path_mask']
    sources = np.asarray([r.dataset for r in train])
    weights = fixed_source_weights(y, sources, source_masses(y, sources, role='train'), role='train')
    positives = (py * mask).sum(0); negatives = ((1 - py) * mask).sum(0)
    if (positives == 0).any() or (negatives == 0).any(): raise ValueError('Pathology training class lacks observed support')
    positive = torch.from_numpy(np.clip(np.sqrt(negatives / positives), 1, 4)).to(device)
    train_ids = np.asarray([lookup[r.key] for r in train])
    bn_ids = train_ids[balanced_bn_indices(train, seed)] if arm == 'supervised_bn' else None
    if bn_ids is not None:
        _json(output / 'bn_record_keys.json', [train[i].key for i in balanced_bn_indices(train, seed)])
    best = {'checkpoint': str(bundle / 'models/student_start.pth'), 'checkpoint_sha256': STUDENT_SHA256,
            **warm, 'epoch': -1, 'fallback': True}
    score = -1.; history = []
    for epoch in range(8):
        configure_student(model)
        order = np.random.default_rng(seed + epoch).permutation(len(train))
        total_loss = 0.; supports = {'rhythm': 0, 'pathology': 0}
        for offset in range(0, len(order), 128):
            if time.monotonic() - start > 90 * 60: raise TimeoutError('Student fit exceeded 90 minutes')
            ids = order[offset:offset + 128]
            batch = torch.from_numpy(np.asarray(x[train_ids[ids]]).copy()).to(device)
            values = model(batch); current = {'rhythm': values[0], 'pathology': values[3]}
            supervised = supervised_losses(current, torch.from_numpy(y[ids]).to(device),
                torch.from_numpy(py[ids]).to(device), torch.from_numpy(mask[ids]).to(device),
                torch.from_numpy(weights[ids]).to(device), positive)
            soft = {key: value[ids] if isinstance(value, np.ndarray) and key != 'path_thresholds' else value
                    for key, value in targets.items() if key in (
                        'rhythm_labels', 'path_targets', 'path_mask', 'rhythm_teacher_logits', 'path_teacher_logits',
                        'path_thresholds', 'path_teacher_probabilities')}
            if arm not in ('rhythm_transfer', 'joint_transfer'): soft['rhythm_teacher_logits'] = None
            if arm not in ('pathology_transfer', 'joint_transfer'): soft['path_teacher_logits'] = None
            transferred = transfer_losses(current, soft)
            loss = supervised['rhythm_loss'] + supervised['pathology_loss'] + transferred['rhythm_loss'] + transferred['pathology_loss']
            if not torch.isfinite(loss): raise ValueError('Nonfinite student loss')
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(backbone + heads, 1., error_if_nonfinite=True)
            optimizer.step()
            total_loss += float(loss.detach()) * len(ids)
            supports['rhythm'] += transferred['rhythm_support']; supports['pathology'] += transferred['pathology_support']
        scheduler.step()
        epoch_root = output / 'epochs' / str(epoch); epoch_root.mkdir(parents=True)
        saved_buffers = None; buffer_diff = {}
        if bn_ids is not None: saved_buffers, buffer_diff = recalibrate_bn(model, x, bn_ids)
        measured, _ = _evaluate(model, rows, x, lookup, epoch_root / 'predictions', baseline_aux=baseline_aux)
        assert_student_preserved(model, initial, arm=arm)
        decision = student_eligibility([measured['metrics']] * 3, [reference] * 3)
        checkpoint = epoch_root / 'checkpoint.pth'; _save_student(checkpoint, model, seed, arm, epoch)
        record = {'epoch': epoch, 'checkpoint': str(checkpoint), 'checkpoint_sha256': file_sha(checkpoint),
                  **measured, 'decision': decision, 'mean_loss': total_loss / len(train),
                  'eligible_transfer_supports': supports, 'bn_buffer_diff': buffer_diff}
        history.append(record)
        utility = (measured['metrics']['rhythm']['macro_f1'] + measured['metrics']['pathology']['macro_f1']) / 2
        if decision['eligible'] and utility > score:
            best = {**record, 'fallback': False}; score = utility
        if saved_buffers is not None:
            for name, value in model.named_buffers(): value.copy_(saved_buffers[name])
        torch.save(training_snapshot(model, optimizer, scheduler, epoch=epoch, seed=seed), epoch_root / 'training_state.pth')
        _json(epoch_root / 'result.json', record)
        print(json.dumps({'student': trial, 'epoch': epoch, 'rhythm_macro_f1': measured['metrics']['rhythm']['macro_f1'],
                          'pathology_macro_f1': measured['metrics']['pathology']['macro_f1'], 'eligible': decision['eligible']}), flush=True)
    result = {**best, 'scope': 'validation', 'arm': arm, 'seed': seed, 'history': history,
              'elapsed_seconds': time.monotonic() - start, 'parameters': 1143751,
              'warm_start_sha256': STUDENT_SHA256, 'optimizer_resume': False, 'fit_identity': identity}
    _json(output / 'result.json', result); seal_directory(output)
    return result


def run_students(bundle):
    allocation()
    metadata = json.loads((bundle / 'teacher-targets/metadata.json').read_text())
    results, skipped = [], {}
    for arm in ARMS:
        try: require_arm(arm, metadata['teachers'])
        except ValueError as exc:
            skipped[arm] = str(exc); continue
        for seed in SEEDS: results.append(fit_student(bundle, arm=arm, seed=seed))
    warm = next(r for r in results if r['arm'] == 'supervised' and r['seed'] == SEEDS[0])
    warm_descriptors = json.loads((bundle / 'teacher-controls/controls.json').read_text())['warm']
    controls = {'checkpoint': str(bundle / 'models/student_start.pth'), 'checkpoint_sha256': STUDENT_SHA256,
                'metrics': {e: prediction_metrics(load_predictions(warm_descriptors[e]), e) for e in ('rhythm', 'pathology')}}
    selection = select_students(results, controls)
    arm = selection['selected_arm'] if selection['promoted'] else 'supervised'
    original = next(r for r in results if r['arm'] == arm and r['seed'] == SEEDS[0])
    repeated = fit_student(bundle, arm=arm, seed=SEEDS[0], repeat=True)
    if len(original['history']) != len(repeated['history']): raise ValueError('Repeated student epoch coverage differs')
    for a, b in zip(original['history'], repeated['history']):
        first = torch.load(a['checkpoint'], weights_only=True, map_location='cpu')['state_dict']
        second = torch.load(b['checkpoint'], weights_only=True, map_location='cpu')['state_dict']
        if first.keys() != second.keys() or any(not torch.equal(first[k], second[k]) for k in first):
            raise ValueError('Repeated student training state differs')
        for endpoint in ('rhythm', 'pathology'):
            if not np.array_equal(load_predictions(a['validation'][endpoint])['logits'], load_predictions(b['validation'][endpoint])['logits']):
                raise ValueError('Repeated student validation logits differ')
    selection['repeated_training_identical'] = True
    selection['skipped_arms'] = skipped
    _json(bundle / 'student_results.json', results)
    _json(bundle / 'student_selection.json', selection)
    return selection
