"""Outcome-blind source selection and two-task shared-gradient projection."""
import hashlib
import math

import torch

PRIMARY = {'164889003': 0, '426177001': 1, '427084000': 2, '426783006': 3,
           '17338001': 4, '164884008': 4, '427172004': 4, '59118001': 5,
           '713427006': 5, '713426002': 5, '164909002': 6, '284470004': 7,
           '63593006': 7, '270492004': 8}
UNSUPPORTED = {'164890007', '427393009', '426761007', '713422000', '233896004',
               '233897008', '195101003', '195042002', '54016002', '28189009',
               '27885002', '233917008', '10370003', '251173003', '426995002',
               '251164006', '11157007', '75532003', '13640000', '195060002',
               '251180001', '74390002', '698252002'}


def rhythm_reference(codes):
    codes = set(codes)
    if codes & UNSUPPORTED:
        return -1
    labels = {PRIMARY[c] for c in codes & PRIMARY.keys()}
    if len(labels) > 1:
        labels.discard(3)
    return next(iter(labels)) if len(labels) == 1 else -1


def select_records(rows, excluded_ids, excluded_signals, cap=500):
    if cap < 1:
        raise ValueError('Positive per-class cap required')
    eligible = [r for r in rows if r['rhythm_target'] in (4, 5, 6, 7, 8)
                and r['ecg_id'] not in excluded_ids and r['signal_sha256'] not in excluded_signals]
    ordered = sorted(eligible, key=lambda r: hashlib.sha256(('titan-directed-20261017:' + r['ecg_id']).encode()).hexdigest())
    counts, signals, ids, result = {}, set(), set(), []
    for row in ordered:
        label = row['rhythm_target']
        if counts.get(label, 0) >= cap or row['signal_sha256'] in signals or row['ecg_id'] in ids:
            continue
        result.append(row)
        counts[label] = counts.get(label, 0) + 1
        signals.add(row['signal_sha256']); ids.add(row['ecg_id'])
    return result


def project_shared_gradients(first, second, shared):
    if not (len(first) == len(second) == len(shared)):
        raise ValueError('Gradient contract differs')
    pairs = [(a, b) for a, b, s in zip(first, second, shared) if s and a is not None and b is not None]
    if any(not torch.isfinite(g).all() for g in (*first, *second) if g is not None):
        raise ValueError('Non-finite gradients')
    if pairs:
        dot = sum((a * b).sum() for a, b in pairs)
        na = sum((a * a).sum() for a, _ in pairs)
        nb = sum((b * b).sum() for _, b in pairs)
        conflict = bool(dot < 0) and bool(na > 0) and bool(nb > 0)
        cosine = float((dot / (na * nb).sqrt().clamp_min(1e-20)).detach())
    else:
        conflict, cosine = False, 0.
    result = []
    for a, b, s in zip(first, second, shared):
        if a is None or b is None:
            result.append(b if a is None else a)
        elif s and conflict:
            result.append((a - dot / nb * b) + (b - dot / na * a))
        else:
            result.append(a + b)
    if not math.isfinite(cosine):
        raise ValueError('Non-finite gradient cosine')
    return result, {'conflict': conflict, 'cosine': cosine}
