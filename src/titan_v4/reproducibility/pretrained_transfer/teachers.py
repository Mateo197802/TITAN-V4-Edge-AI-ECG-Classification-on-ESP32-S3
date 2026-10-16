"""Allocation-only feature extraction and task adaptation of the pinned teacher."""
import csv
import json
import os
from pathlib import Path
import random
import shutil
import time

import numpy as np
from scipy.special import expit, softmax
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from .contracts import CONTRACTS, PRIMARY9_CLASSES, PairedRecord
from .execution import file_sha, verify_inventory, verify_frozen_inputs, MANIFEST_SHA
from .founder import load_founder, founder_features, FOUNDER_SHA256
from .selection import endpoint_metrics, prediction_metrics, load_predictions, qualify_teachers, SEEDS
from titan_v4.reproducibility.joint_refinement import PATHOLOGY_CLASSES, masked_pathology_loss
from titan_v4.reproducibility.supervision_control import source_masses, fixed_source_weights

CONTROL_MODELS = {
    'rhythm': ('outputs/reviewer_verification/supervision_control_2026-10-14/model/rhythm_selected.pth',
               '0941b585d62416ab176942a2b0a65c2c8824c2cffb87609a515e6f36de126d8c'),
    'pathology': ('outputs/reviewer_verification/plateau_audit_2026-10-14/training/phase6.pth',
                  'fd8a5d6ca95b7875e53305ef0fdc3962f2aa4d493fa16630befc7a75e1cb2249')}


def allocation(seed=20261014):
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():
        raise RuntimeError('Teacher execution requires a CUDA Slurm allocation')
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    return torch.device('cuda')


def teacher_pathology_loss(logits, targets, mask, *, positive_weights=None):
    if not torch.isfinite(logits).all():
        raise ValueError('Nonfinite teacher logits')
    positive_weights = torch.ones(10, device=logits.device) if positive_weights is None else positive_weights
    return masked_pathology_loss(logits, targets, mask, positive_weights)


def configure_teacher(model, stage):
    if stage not in ('heads', 'adapt'):
        raise ValueError('Unknown teacher fit stage')
    model.eval()
    last = 'stage_list.' + str(len(model.stage_list) - 1) + '.'
    for name, p in model.named_parameters():
        p.requires_grad_(name.startswith('dense.') or (stage == 'adapt' and name.startswith(last)))
    return [name for name, p in model.named_parameters() if p.requires_grad]


def fitting_rows(rows, endpoint):
    if endpoint not in ('rhythm', 'pathology') or not rows or any(r.role != 'train' for r in rows):
        raise ValueError('Teacher fitting requires train rows only')
    selected = [r for r in rows if r.rhythm_label >= 0] if endpoint == 'rhythm' else [r for r in rows if any(r.path_mask)]
    if not selected:
        raise ValueError('No observed training support')
    return selected


def paired_rows(bundle):
    configuration = Path(__file__).resolve().parents[4] / 'configs/founder_transfer_2026-10-14.json'
    return [PairedRecord(**r) for r in verify_frozen_inputs(bundle, configuration)]


class SourcePairs(Dataset):
    def __init__(self, rows, root, contract):
        self.rows, self.root, self.contract = rows, root, contract

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        from .inputs import load_pair
        pair = load_pair(self.rows[index], self.root, self.contract, missing_policy='wfdb16_zero_fill')
        return index, pair.teacher, pair.student


def _json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def seal_directory(root):
    inventory = {p.relative_to(root).as_posix(): file_sha(p) for p in root.rglob('*')
                 if p.is_file() and p.name != 'SHA256SUMS.json'}
    _json(root / 'SHA256SUMS.json', inventory)
    return file_sha(root / 'SHA256SUMS.json')


def cache_features(bundle):
    device = allocation()
    rows = [r for r in paired_rows(bundle) if r.role in ('train', 'rhythm_validation', 'pathology_validation')]
    unique = {}
    for r in rows:
        if r.key in unique and (r.signal_sha256, r.student_tensor_sha256) != (
                unique[r.key].signal_sha256, unique[r.key].student_tensor_sha256):
            raise ValueError('Inconsistent paired source identity')
        unique[r.key] = r
    ordered = [unique[k] for k in sorted(unique)]
    output = bundle / 'feature-cache'; output.mkdir(exist_ok=False)
    if shutil.disk_usage(bundle).free < len(ordered) * (4 * 12 * 5000 * 4 + 6 * 1250 * 4) + 5 * 1024**3:
        raise ValueError('Insufficient disk for streamed on-disk teacher caches')
    _json(output / 'keys.json', [r.key for r in ordered])
    student = np.lib.format.open_memmap(output / 'student.npy', mode='w+', dtype='float32', shape=(len(ordered), 6, 1250))
    model = load_founder(bundle / 'weights/12_lead_ECGFounder.pth', outputs=150, seed=20261014).to(device).eval()
    model.requires_grad_(False)
    start = time.monotonic(); contracts = {}
    for ci, contract in enumerate(CONTRACTS):
        feature_path = output / (contract + '-features.npy')
        tensor_path = output / (contract + '-signals.npy')
        features = np.lib.format.open_memmap(feature_path, mode='w+', dtype='float32', shape=(len(ordered), 1024))
        tensors = np.lib.format.open_memmap(tensor_path, mode='w+', dtype='float32', shape=(len(ordered), 12, 5000))
        loader = DataLoader(SourcePairs(ordered, bundle / 'raw', contract), batch_size=8, num_workers=4,
                            shuffle=False, persistent_workers=True, pin_memory=True)
        for indices, teacher, original in loader:
            with torch.no_grad():
                _, embedding = founder_features(model, teacher.to(device, non_blocking=True))
            ids = indices.numpy()
            features[ids] = embedding.cpu().numpy(); tensors[ids] = teacher.numpy()
            if ci == 0:
                student[ids] = original.numpy()
        features.flush(); tensors.flush(); student.flush()
        del features, tensors, loader
        contracts[contract] = {'features': feature_path.name, 'signals': tensor_path.name}
        print(json.dumps({'completed_contract': contract, 'records': len(ordered), 'elapsed_seconds': time.monotonic() - start}), flush=True)
    del student, model
    _json(output / 'cache.json', {'contracts': contracts, 'records': len(ordered), 'teacher_sha256': FOUNDER_SHA256,
                                 'paired_manifest_sha256': MANIFEST_SHA, 'batch': 8,
                                 'peak_bytes': torch.cuda.max_memory_allocated(), 'elapsed_seconds': time.monotonic() - start})
    seal_directory(output)
    return json.loads((output / 'cache.json').read_text())


def open_cache(bundle, contract, *, verify=True):
    cache = bundle / 'feature-cache'
    inventory = json.loads((cache / 'SHA256SUMS.json').read_text())
    required = ('keys.json', 'cache.json', 'student.npy', contract + '-features.npy', contract + '-signals.npy')
    if verify:
        verify_inventory(cache, {n: inventory[n] for n in required})
    metadata = json.loads((cache / 'cache.json').read_text())
    if metadata['paired_manifest_sha256'] != MANIFEST_SHA or metadata['teacher_sha256'] != FOUNDER_SHA256:
        raise ValueError('Feature cache identity differs')
    keys = json.loads((cache / 'keys.json').read_text())
    if len(keys) != len(set(keys)):
        raise ValueError('Duplicate feature keys')
    return {name: np.load(cache / (name + '.npy'), mmap_mode='r', allow_pickle=False)
            for name in ('student', contract + '-features', contract + '-signals')}, {k: i for i, k in enumerate(keys)}


def save_predictions(root, rows, logits, endpoint):
    if not rows or any(r.role != endpoint + '_validation' for r in rows):
        raise ValueError('Prediction metrics require validation rows only')
    labels = np.asarray([r.rhythm_label for r in rows], np.int64) if endpoint == 'rhythm' else np.asarray([r.path_targets for r in rows], np.float32)
    mask = np.ones(len(rows), np.float32) if endpoint == 'rhythm' else np.asarray([r.path_mask for r in rows], np.float32)
    classes = PRIMARY9_CLASSES if endpoint == 'rhythm' else PATHOLOGY_CLASSES
    arrays = {'keys': np.asarray([r.key for r in rows]), 'labels': labels, 'mask': mask,
              'logits': np.asarray(logits), 'sources': np.asarray([r.dataset for r in rows]),
              'patients': np.asarray([r.patient_token or r.key for r in rows]),
              'roles': np.asarray([r.role for r in rows]), 'classes': np.asarray(classes)}
    metric = endpoint_metrics(logits, labels, mask, arrays['sources'], arrays['patients'], endpoint=endpoint, role='validation')
    path = root / 'validation.npz'
    np.savez_compressed(path, **arrays)
    with (root / 'validation.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['record_key', 'source', 'patient_token'] + ['target_' + c for c in classes]
                        + ['logit_' + c for c in classes] + ['probability_' + c for c in classes])
        probabilities = softmax(logits, axis=1) if endpoint == 'rhythm' else expit(logits)
        for i, r in enumerate(rows):
            target = np.eye(9, dtype=int)[labels[i]].tolist() if endpoint == 'rhythm' else labels[i].tolist()
            writer.writerow([r.key, r.dataset, r.patient_token or '', *target, *np.asarray(logits[i]).tolist(),
                             *probabilities[i].tolist()])
    return {'endpoint': endpoint, 'scope': 'validation', 'validation_npz': str(path),
            'validation_npz_sha256': file_sha(path), 'metrics': metric}


def controls(bundle):
    device = allocation()
    from titan_v4.models.titan_v4_net import TitanV4Lite
    configuration = Path(__file__).resolve().parents[4] / 'configs/founder_transfer_2026-10-14.json'
    config = json.loads(configuration.read_text())
    checkpoint = bundle / 'models/student_start.pth'
    if file_sha(checkpoint) != config['student_sha256']:
        raise ValueError('Control warm-start differs')
    cache, lookup = open_cache(bundle, 'canonical_unfiltered')
    rows = paired_rows(bundle); results = {'warm': {}}
    output = bundle / 'teacher-controls'; output.mkdir(exist_ok=False)
    for role, endpoint, index in (('warm', 'rhythm', 0), ('warm', 'pathology', 3),
                                   ('task', 'rhythm', 0), ('task', 'pathology', 3)):
        path = checkpoint if role == 'warm' else bundle / 'models' / (endpoint + '_control.pth')
        expected = config['student_sha256'] if role == 'warm' else CONTROL_MODELS[endpoint][1]
        if file_sha(path) != expected: raise ValueError('Pinned comparison control differs')
        state = torch.load(path, weights_only=True, map_location='cpu')
        model = TitanV4Lite(in_channels=6, num_rhythm=9, num_pathology=10).to(device).eval()
        model.load_state_dict(state.get('state_dict', state.get('model_state_dict', state)), strict=True)
        selected = [r for r in rows if r.role == endpoint + '_validation']
        predictions = []
        for start in range(0, len(selected), 8):
            batch = selected[start:start + 8]
            x = torch.from_numpy(np.asarray(cache['student'][[lookup[r.key] for r in batch]]).copy()).to(device)
            with torch.no_grad():
                predictions.append(model(x)[index].cpu().numpy())
        root = output / role / endpoint; root.mkdir(parents=True)
        result = save_predictions(root, selected, np.concatenate(predictions), endpoint)
        result.update(checkpoint_sha256=expected)
        if role == 'warm': results['warm'][endpoint] = result
        else: results[endpoint] = result
        _json(root / 'result.json', result)
        del model
    _json(output / 'controls.json', results); seal_directory(output)
    return results


def _forward(model, values, *, stage, batch=256):
    outputs = []
    with torch.no_grad():
        for start in range(0, len(values), batch):
            x = torch.from_numpy(np.asarray(values[start:start + batch]).copy()).cuda()
            logits = model.dense(x) if stage == 'heads' else founder_features(model, x)[0]
            outputs.append(logits.cpu().numpy())
    return np.concatenate(outputs)


def _delta(model, trainable):
    return {k: p.detach().cpu().clone() for k, p in model.state_dict().items() if k in trainable}


def assert_teacher_frozen(model, initial, trainable):
    for name, value in model.state_dict().items():
        if name not in trainable and (name not in initial or not torch.equal(value.cpu(), initial[name])):
            raise ValueError('Teacher frozen body/buffer changed: ' + name)


def load_teacher_delta(bundle, checkpoint, *, endpoint, seed):
    state = torch.load(checkpoint, weights_only=True, map_location='cpu')
    expected_classes = PRIMARY9_CLASSES if endpoint == 'rhythm' else PATHOLOGY_CLASSES
    if (state['teacher_sha256'] != FOUNDER_SHA256 or state['endpoint'] != endpoint or state['seed'] != seed
            or state.get('paired_manifest_sha256') != MANIFEST_SHA or state.get('contract') not in CONTRACTS
            or tuple(state.get('classes', ())) != tuple(expected_classes)):
        raise ValueError('Teacher delta provenance differs')
    model = load_founder(bundle / 'weights/12_lead_ECGFounder.pth', outputs=9 if endpoint == 'rhythm' else 10, seed=seed)
    allowed = set(configure_teacher(model, state['stage']))
    if set(state['delta']) != allowed:
        raise ValueError('Teacher delta trainable schema differs')
    original = model.state_dict()
    for key, value in state['delta'].items():
        if value.shape != original[key].shape or value.dtype != original[key].dtype or not torch.isfinite(value).all():
            raise ValueError('Teacher delta tensor schema differs')
        original[key] = value
    model.load_state_dict(original, strict=True)
    return model


def fit_teacher(bundle: Path, *, endpoint: str, contract: str, seed: int, stage: str, repeat=False) -> dict:
    device = allocation(seed)
    if contract not in CONTRACTS or seed not in SEEDS or stage not in ('heads', 'adapt'):
        raise ValueError('Unregistered teacher trial')
    rows = paired_rows(bundle)
    train = fitting_rows([r for r in rows if r.role == 'train'], endpoint)
    validation = [r for r in rows if r.role == endpoint + '_validation']
    cache, lookup = open_cache(bundle, contract)
    values = cache[contract + ('-features' if stage == 'heads' else '-signals')]
    train_ids = np.asarray([lookup[r.key] for r in train]); val_ids = np.asarray([lookup[r.key] for r in validation])
    # Validation is compact; fitting tensors remain disk-backed and batch indexed.
    val_values = np.asarray(values[val_ids])
    trial_name = '-'.join((endpoint, contract, stage, str(seed))) + ('-repeat' if repeat else '')
    from .execution import fit_identity, prepare_fit_trial
    identity = fit_identity('teacher', endpoint=endpoint, contract=contract, seed=seed, stage=stage, repeat=repeat)
    output = bundle / 'teacher-fits' / trial_name
    completed = prepare_fit_trial(output, identity)
    if completed is not None: return completed
    output.mkdir(parents=True, exist_ok=False)
    if stage == 'adapt':
        parent = bundle / 'teacher-fits' / '-'.join((endpoint, contract, 'heads', str(seed))) / 'checkpoint.pth'
        model = load_teacher_delta(bundle, parent, endpoint=endpoint, seed=seed)
    else:
        model = load_founder(bundle / 'weights/12_lead_ECGFounder.pth', outputs=9 if endpoint == 'rhythm' else 10, seed=seed)
    trainable = configure_teacher(model, stage)
    frozen = {k: v.clone() for k, v in model.state_dict().items() if k not in trainable}
    model = model.to(device)
    groups = [{'params': [p for n, p in model.named_parameters() if p.requires_grad and n.startswith('dense.')],
               'lr': 1e-3 if stage == 'heads' else 1e-4}]
    body = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith('dense.')]
    if body: groups.append({'params': body, 'lr': 1e-5})
    optimizer = torch.optim.AdamW(groups, weight_decay=1e-4)
    y = np.asarray([r.rhythm_label for r in train], np.int64) if endpoint == 'rhythm' else np.asarray([r.path_targets for r in train], np.float32)
    mask = np.ones(len(train), np.float32) if endpoint == 'rhythm' else np.asarray([r.path_mask for r in train], np.float32)
    if endpoint == 'rhythm':
        sources = np.asarray([r.dataset for r in train])
        weights = fixed_source_weights(y, sources, source_masses(y, sources, role='train'), role='train')
        positive = None
    else:
        positive = torch.from_numpy(np.clip(np.sqrt(((1 - y) * mask).sum(0) / np.maximum(1, (y * mask).sum(0))), 1, 4)).to(device)
    start = time.monotonic(); best_score = -1.; stale = 0; history = []; best = None
    maximum = 20 if stage == 'heads' else 5
    for epoch in range(maximum):
        order = np.random.default_rng(seed + epoch).permutation(len(train))
        losses = []
        for offset in range(0, len(order), 8):
            if time.monotonic() - start > 3 * 3600:
                raise TimeoutError('Teacher fit reached its three-hour ceiling')
            ids = order[offset:offset + 8]
            x = torch.from_numpy(np.asarray(values[train_ids[ids]]).copy()).to(device)
            target = torch.from_numpy(y[ids]).to(device)
            logits = model.dense(x) if stage == 'heads' else founder_features(model, x)[0]
            if endpoint == 'rhythm':
                loss = (F.cross_entropy(logits, target, reduction='none') * torch.from_numpy(weights[ids]).to(device)).mean()
            else:
                loss = teacher_pathology_loss(logits, target, torch.from_numpy(mask[ids]).to(device), positive_weights=positive)
            if not torch.isfinite(loss): raise ValueError('Nonfinite fit loss')
            optimizer.zero_grad(set_to_none=True); loss.backward()
            optimizer.step(); losses.append(float(loss.detach()))
        logits = _forward(model, val_values, stage=stage, batch=8)
        metric = endpoint_metrics(logits,
            np.asarray([r.rhythm_label for r in validation]) if endpoint == 'rhythm' else np.asarray([r.path_targets for r in validation]),
            np.ones(len(validation)) if endpoint == 'rhythm' else np.asarray([r.path_mask for r in validation]),
            np.asarray([r.dataset for r in validation]), np.asarray([r.patient_token or r.key for r in validation]),
            endpoint=endpoint, role='validation')
        history.append({'epoch': epoch, 'loss': float(np.mean(losses)), 'metrics': metric})
        if metric['macro_f1'] > best_score:
            best_score = metric['macro_f1']; stale = 0
            best = _delta(model, trainable)
            torch.save({'delta': best, 'endpoint': endpoint, 'seed': seed, 'stage': stage,
                        'contract': contract, 'teacher_sha256': FOUNDER_SHA256, 'paired_manifest_sha256': MANIFEST_SHA,
                        'classes': list(PRIMARY9_CLASSES if endpoint == 'rhythm' else PATHOLOGY_CLASSES),
                        'epoch': epoch}, output / 'checkpoint.pth')
            torch.save({'optimizer': optimizer.state_dict(), 'torch_rng': torch.get_rng_state(),
                        'cuda_rng': torch.cuda.get_rng_state_all(), 'epoch': epoch}, output / 'training_state.pth')
        else: stale += 1
        print(json.dumps({'trial': trial_name, 'epoch': epoch, 'validation_macro_f1': metric['macro_f1']}), flush=True)
        if stage == 'heads' and stale >= 4: break
    assert_teacher_frozen(model, frozen, set(trainable))
    del model
    model = load_teacher_delta(bundle, output / 'checkpoint.pth', endpoint=endpoint, seed=seed).to(device).eval()
    logits = _forward(model, val_values, stage=stage, batch=8)
    result = save_predictions(output, validation, logits, endpoint)
    result.update(contract=contract, seed=seed, stage=stage, checkpoint=str(output / 'checkpoint.pth'),
                  checkpoint_sha256=file_sha(output / 'checkpoint.pth'), elapsed_seconds=time.monotonic() - start,
                  frozen_state_unchanged=True, epochs=len(history), history=history, fit_identity=identity)
    _json(output / 'result.json', result); seal_directory(output)
    return result


def adaptation_candidates(qualification):
    selected = {}
    for endpoint in ('rhythm', 'pathology'):
        candidates = []
        for result in qualification['decisions']:
            if result['endpoint'] != endpoint: continue
            controls = [result['control']] + ([result['warm_control']] if 'warm_control' in result else [])
            mean = result['mean']
            supported = all(mean['macro_f1'] + 1e-12 >= base['macro_f1'] and (
                mean['accuracy'] + 1e-12 >= base['accuracy'] - .005 if endpoint == 'rhythm'
                else mean['macro_ap'] + 1e-12 >= base['macro_ap']) for base in controls)
            if supported: candidates.append(result)
        if candidates:
            selected[endpoint] = max(candidates, key=lambda r: (r['mean']['macro_f1'], r['contract']))['contract']
    return selected


def run_teachers(bundle):
    allocation()
    control_root = bundle / 'teacher-controls'
    verify_inventory(control_root, json.loads((control_root / 'SHA256SUMS.json').read_text()),
                     exact=True, seal_name='SHA256SUMS.json')
    control = json.loads((control_root / 'controls.json').read_text())
    results = []
    for endpoint in ('rhythm', 'pathology'):
        for contract in CONTRACTS:
            for seed in SEEDS:
                results.append(fit_teacher(bundle, endpoint=endpoint, contract=contract, seed=seed, stage='heads'))
    heads = qualify_teachers(results, control)
    _json(bundle / 'head_qualification.json', heads)
    for endpoint, contract in adaptation_candidates(heads).items():
        for seed in SEEDS:
            results.append(fit_teacher(bundle, endpoint=endpoint, contract=contract, seed=seed, stage='adapt'))
    repeated = fit_teacher(bundle, endpoint='rhythm', contract='canonical_unfiltered',
                           seed=20261014, stage='heads', repeat=True)
    original = next(r for r in results if (r['endpoint'], r['contract'], r['seed'], r['stage']) ==
                    ('rhythm', 'canonical_unfiltered', 20261014, 'heads'))
    a = torch.load(original['checkpoint'], map_location='cpu', weights_only=True)['delta']
    b = torch.load(repeated['checkpoint'], map_location='cpu', weights_only=True)['delta']
    if a.keys() != b.keys() or any(not torch.equal(a[k], b[k]) for k in a):
        raise ValueError('Repeated head fit parameters differ')
    if not np.array_equal(load_predictions(original)['logits'], load_predictions(repeated)['logits']):
        raise ValueError('Repeated head-fit validation logits differ')
    qualification = qualify_teachers(results, control)
    qualification['repeated_fit_identical'] = True
    _json(bundle / 'teacher_results.json', results)
    _json(bundle / 'teacher_qualification.json', qualification)
    return qualification
