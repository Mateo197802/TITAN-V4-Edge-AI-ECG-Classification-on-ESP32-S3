"""Training-only keyed soft targets and reference-agreeing transfer losses."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import expit, softmax
import torch
from torch.nn import functional as F

from .contracts import PRIMARY9_CLASSES, PATHOLOGY_CLASSES, CONTRACTS, SEEDS, FOUNDER_SHA256
from .execution import MANIFEST_SHA, file_sha, verify_inventory, safe_relative, public_paths


def contract_fingerprint(contract):
    if contract not in CONTRACTS:
        raise ValueError('Unregistered teacher preprocessing contract')
    source = Path(__file__).with_name('inputs.py').read_text(encoding='utf-8')
    descriptor = {'contract': contract, 'paired_manifest_sha256': MANIFEST_SHA,
                  'normalized_input_source_sha256': hashlib.sha256(source.encode()).hexdigest()}
    return hashlib.sha256(json.dumps(descriptor, sort_keys=True).encode()).hexdigest()


def _references(records):
    if not records or any(r.role != 'train' or r.window_start != 0 or r.window_samples != 5000 for r in records):
        raise ValueError('Soft targets require the original training allowlist/window')
    keys = [r.key for r in records]
    if len(keys) != len(set(keys)):
        raise ValueError('Duplicate training keys')
    rhythm = np.asarray([r.rhythm_label for r in records])
    path = np.asarray([r.path_targets for r in records], np.float32)
    mask = np.asarray([r.path_mask for r in records], np.float32)
    if (rhythm.dtype.kind not in 'iu' or not np.isin(rhythm, range(-1, 9)).all()
            or path.shape != (len(records), 10) or mask.shape != path.shape
            or not np.isin(path, [0, 1]).all() or not np.isin(mask, [0, 1]).all()):
        raise ValueError('Invalid original references or observed masks')
    return {'record_keys': np.asarray(keys), 'rhythm_labels': rhythm.astype(np.int64),
            'path_targets': path, 'path_mask': mask,
            'header_sha256': np.asarray([r.header_sha256 for r in records]),
            'signal_sha256': np.asarray([r.signal_sha256 for r in records]),
            'student_tensor_sha256': np.asarray([r.student_tensor_sha256 for r in records]),
            'patient_tokens': np.asarray([r.patient_token or '' for r in records]),
            'window_start': np.asarray([r.window_start for r in records]),
            'window_samples': np.asarray([r.window_samples for r in records]),
            'rhythm_classes': np.asarray(PRIMARY9_CLASSES), 'path_classes': np.asarray(PATHOLOGY_CLASSES)}


def _qualification(qualification, bundle):
    if (qualification.get('scope') != 'validation' or qualification.get('selection_uses_tracking') is not False
            or qualification.get('seeds') != list(SEEDS)):
        raise ValueError('Only sealed validation qualification may supply targets')
    checked = {}
    for endpoint, result in qualification['eligible'].items():
        if endpoint not in ('rhythm', 'pathology') or not result.get('eligible') or result['contract'] not in CONTRACTS:
            raise ValueError('Unqualified endpoint or contract')
        checkpoints = result['checkpoints']
        if [c['seed'] for c in checkpoints] != list(SEEDS):
            raise ValueError('All three ordered teacher seeds are required')
        checked[endpoint] = {'contract': result['contract'], 'contract_sha256': contract_fingerprint(result['contract']),
                             'checkpoints': []}
        for c in checkpoints:
            path = Path(c['path'])
            if not path.is_absolute(): path = bundle / safe_relative(c['path'])
            if not path.resolve().is_relative_to(bundle.resolve()) or file_sha(path) != c['sha256']:
                raise ValueError('Teacher checkpoint hash/path differs')
            checked[endpoint]['checkpoints'].append({'seed': c['seed'],
                'path': path.relative_to(bundle).as_posix(), 'sha256': c['sha256']})
        if endpoint == 'pathology':
            thresholds = np.asarray(result['ensemble']['thresholds'], np.float64)
            if thresholds.shape != (10,) or not np.isfinite(thresholds).all() or ((thresholds < 0) | (thresholds > 1)).any():
                raise ValueError('Frozen pathology thresholds differ')
            checked[endpoint]['thresholds'] = thresholds.tolist()
    return checked


def _json(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n')


def write_target_cache(output, records, teacher_logits, qualification, *, producer=None):
    output = Path(output).resolve(); bundle = output.parent
    references = _references(records)
    checked = _qualification(qualification, bundle)
    if set(teacher_logits) != set(checked):
        raise ValueError('Target endpoint coverage differs from qualification')
    arrays = dict(references)
    for endpoint, values in teacher_logits.items():
        values = np.asarray(values, np.float32)
        columns = 9 if endpoint == 'rhythm' else 10
        if values.shape != (len(records), columns) or not np.isfinite(values).all():
            raise ValueError('Invalid teacher logits shape/value')
        prefix = 'rhythm' if endpoint == 'rhythm' else 'path'
        arrays[prefix + '_teacher_logits'] = values
        arrays[prefix + '_teacher_probabilities'] = softmax(values, axis=1) if endpoint == 'rhythm' else expit(values)
    if 'pathology' in checked:
        arrays['path_thresholds'] = np.asarray(checked['pathology']['thresholds'], np.float64)
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / 'targets.npz', **arrays)
    _json(output / 'qualification.json', public_paths(qualification, bundle))
    _json(output / 'metadata.json', {'paired_manifest_sha256': MANIFEST_SHA, 'scope': 'train',
        'records': len(records), 'teachers': checked, 'absent_endpoints': sorted({'rhythm', 'pathology'} - set(checked)),
        'rhythm_classes': list(PRIMARY9_CLASSES), 'pathology_classes': list(PATHOLOGY_CLASSES),
        'rhythm_temperature': 2., 'coefficient': .25, 'aggregation': 'mean_logits',
        'pretrained_teacher_sha256': FOUNDER_SHA256, 'qualification_producer': producer,
        'qualification_sha256': file_sha(output / 'qualification.json')})
    with (output / 'records.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['record_key', 'header_sha256', 'signal_sha256', 'student_tensor_sha256', 'window_start', 'window_samples', 'rhythm_reference']
                        + ['path_target_' + c for c in PATHOLOGY_CLASSES] + ['path_mask_' + c for c in PATHOLOGY_CLASSES])
        for r in records:
            writer.writerow([r.key, r.header_sha256, r.signal_sha256, r.student_tensor_sha256, r.window_start,
                             r.window_samples, r.rhythm_label, *r.path_targets, *r.path_mask])
    _json(output / 'SHA256SUMS.json', {p.name: file_sha(p) for p in output.iterdir() if p.is_file()})
    join_teacher_targets(records, output)
    return output


def join_teacher_targets(records, cache, *, checkpoint_root=None):
    cache = Path(cache).resolve()
    checkpoint_root = cache.parent if checkpoint_root is None else Path(checkpoint_root).resolve()
    requested = _references(records)
    verify_inventory(cache, json.loads((cache / 'SHA256SUMS.json').read_text()), exact=True, seal_name='SHA256SUMS.json')
    metadata = json.loads((cache / 'metadata.json').read_text())
    if (metadata['scope'] != 'train' or metadata['paired_manifest_sha256'] != MANIFEST_SHA
            or metadata['rhythm_classes'] != list(PRIMARY9_CLASSES) or metadata['pathology_classes'] != list(PATHOLOGY_CLASSES)
            or metadata['aggregation'] != 'mean_logits' or metadata['pretrained_teacher_sha256'] != FOUNDER_SHA256):
        raise ValueError('Target scope, input identity or class order differs')
    if file_sha(cache / 'qualification.json') != metadata['qualification_sha256']:
        raise ValueError('Qualification receipt hash differs')
    qualification = json.loads((cache / 'qualification.json').read_text())
    if _qualification(qualification, checkpoint_root) != metadata['teachers']:
        raise ValueError('Teacher metadata differs from qualification')
    for description in metadata['teachers'].values():
        if description['contract_sha256'] != contract_fingerprint(description['contract']):
            raise ValueError('Teacher preprocessing fingerprint differs')
        if [c['seed'] for c in description['checkpoints']] != list(SEEDS):
            raise ValueError('Missing teacher checkpoint seed')
        for c in description['checkpoints']:
            file = checkpoint_root / safe_relative(c['path'])
            if not file.resolve().is_relative_to(checkpoint_root) or file_sha(file) != c['sha256']:
                raise ValueError('Teacher checkpoint hash differs')
    with np.load(cache / 'targets.npz', allow_pickle=False) as archive:
        arrays = {k: archive[k] for k in archive.files}
    keys = arrays['record_keys'].tolist()
    if len(keys) != len(set(keys)) or set(keys) != set(requested['record_keys'].tolist()):
        raise ValueError('Missing/extra/duplicate target record keys')
    lookup = {key: i for i, key in enumerate(keys)}
    order = np.asarray([lookup[r.key] for r in records])
    for key, expected in requested.items():
        actual = arrays[key] if key in ('rhythm_classes', 'path_classes') else arrays[key][order]
        if not np.array_equal(actual, expected):
            raise ValueError('Target reference/source identity differs: ' + key)
    result = dict(requested)
    for endpoint, prefix in (('rhythm', 'rhythm'), ('pathology', 'path')):
        field = prefix + '_teacher_logits'
        if endpoint not in metadata['teachers']:
            if field in arrays: raise ValueError('Unqualified teacher output present')
            result[field] = None
            continue
        columns = 9 if endpoint == 'rhythm' else 10
        values = arrays[field]
        probability = arrays[prefix + '_teacher_probabilities']
        expected = softmax(values, axis=1) if endpoint == 'rhythm' else expit(values)
        if (values.shape != (len(records), columns) or not np.isfinite(values).all()
                or probability.shape != values.shape or not np.isfinite(probability).all()
                or ((probability < 0) | (probability > 1)).any()
                or not np.allclose(probability, expected, atol=1e-7, rtol=1e-6)):
            raise ValueError('Invalid teacher probability/logit semantics')
        result[field] = values[order]
        result[prefix + '_teacher_probabilities'] = probability[order]
    if 'pathology' in metadata['teachers']:
        thresholds = np.asarray(metadata['teachers']['pathology']['thresholds'])
        if not np.array_equal(arrays['path_thresholds'], thresholds):
            raise ValueError('Frozen teacher thresholds differ')
        result['path_thresholds'] = thresholds
    else: result['path_thresholds'] = None
    return result


def transfer_losses(student_logits, targets, *, temperature=2., coefficient=.25):
    if not np.isfinite(temperature) or temperature <= 0 or not np.isfinite(coefficient) or coefficient < 0:
        raise ValueError('Invalid transfer constants')
    rhythm, pathology = student_logits['rhythm'], student_logits['pathology']
    if (rhythm.ndim != 2 or rhythm.shape[1] != 9 or pathology.shape != (len(rhythm), 10)
            or not torch.isfinite(rhythm).all() or not torch.isfinite(pathology).all()):
        raise ValueError('Student transfer output contract differs')
    result = {'rhythm_loss': rhythm.sum() * 0, 'pathology_loss': pathology.sum() * 0,
              'rhythm_support': 0, 'pathology_support': 0}
    if targets.get('rhythm_teacher_logits') is not None:
        teacher = torch.as_tensor(targets['rhythm_teacher_logits'], device=rhythm.device, dtype=rhythm.dtype).detach()
        raw_y = torch.as_tensor(targets['rhythm_labels'], device=rhythm.device)
        if raw_y.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
            raise ValueError('Rhythm references must be integer labels without coercion')
        y = raw_y.long()
        if teacher.shape != rhythm.shape or y.shape != (len(rhythm),) or not torch.isfinite(teacher).all() or not ((y >= -1) & (y < 9)).all():
            raise ValueError('Rhythm soft-target contract differs')
        eligible = (y >= 0) & (teacher.argmax(1) == y)
        divergence = F.kl_div(F.log_softmax(rhythm / temperature, dim=1),
                              F.softmax(teacher / temperature, dim=1), reduction='none').sum(1)
        result['rhythm_loss'] = coefficient * temperature**2 * (divergence * eligible).sum() / eligible.sum().clamp_min(1)
        result['rhythm_support'] = int(eligible.sum())
    if targets.get('path_teacher_logits') is not None:
        teacher = torch.as_tensor(targets['path_teacher_logits'], device=pathology.device, dtype=pathology.dtype).detach()
        y = torch.as_tensor(targets['path_targets'], device=pathology.device, dtype=pathology.dtype)
        mask = torch.as_tensor(targets['path_mask'], device=pathology.device)
        thresholds = torch.as_tensor(targets['path_thresholds'], device=pathology.device, dtype=torch.float64)
        if (teacher.shape != pathology.shape or y.shape != teacher.shape or mask.shape != y.shape
                or thresholds.shape != (10,) or not torch.isfinite(teacher).all() or not torch.isfinite(thresholds).all()
                or not ((y == 0) | (y == 1)).all() or not ((mask == 0) | (mask == 1)).all()
                or not ((thresholds >= 0) & (thresholds <= 1)).all()):
            raise ValueError('Pathology soft-target contract differs')
        probability = teacher.sigmoid() if targets.get('path_teacher_probabilities') is None else torch.as_tensor(
            targets['path_teacher_probabilities'], device=pathology.device, dtype=pathology.dtype).detach()
        if (probability.shape != teacher.shape or not torch.isfinite(probability).all()
                or not ((probability >= 0) & (probability <= 1)).all()
                or not torch.allclose(probability, teacher.sigmoid(), atol=1e-7, rtol=1e-6)):
            raise ValueError('Pathology teacher probability semantics differ')
        eligible = (mask == 1) & ((probability.double() >= thresholds) == (y == 1))
        loss = F.binary_cross_entropy_with_logits(pathology, probability, reduction='none')
        result['pathology_loss'] = coefficient * (loss * eligible).sum() / eligible.sum().clamp_min(1)
        result['pathology_support'] = int(eligible.sum())
    return result


def export_teacher_targets(bundle: Path, qualification: dict) -> Path:
    from .teachers import allocation, paired_rows, open_cache, load_teacher_delta
    device = allocation()
    from .execution import successful_stage, public_paths
    producer = successful_stage(bundle, 'teachers')
    if producer['qualification'] != public_paths(qualification, bundle):
        raise ValueError('Teacher qualification differs from successful sealed producer')
    checked = _qualification(qualification, bundle)
    records = [r for r in paired_rows(bundle) if r.role == 'train']
    predictions = {}
    for endpoint, description in checked.items():
        contract = description['contract']
        cache, lookup = open_cache(bundle, contract)
        indices = np.asarray([lookup[r.key] for r in records])
        logits = []
        for checkpoint in description['checkpoints']:
            path = bundle / checkpoint['path']
            model = load_teacher_delta(bundle, path, endpoint=endpoint, seed=checkpoint['seed']).to(device).eval()
            stored = torch.load(path, weights_only=True, map_location='cpu')
            stage = stored['stage']; values = cache[contract + ('-features' if stage == 'heads' else '-signals')]
            from .founder import founder_features
            batches = []
            for offset in range(0, len(records), 8):
                x = torch.from_numpy(np.asarray(values[indices[offset:offset + 8]]).copy()).to(device)
                with torch.no_grad():
                    output = model.dense(x) if stage == 'heads' else founder_features(model, x)[0]
                    batches.append(output.cpu().numpy())
            logits.append(np.concatenate(batches)); del model
        predictions[endpoint] = np.stack(logits).mean(0)
    from .execution import IDENTITY_FIELDS
    return write_target_cache(bundle / 'teacher-targets', records, predictions, qualification,
                              producer={k: producer[k] for k in IDENTITY_FIELDS})
