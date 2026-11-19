"""Explicit targets and validation-only calibration for the original four-head CNN."""
from __future__ import annotations

import ast
import numpy as np

PATHOLOGY_CLASSES = ('IMI','ASMI','LVH','ISC_','ISCAL','NST_','ILMI','AMI','ALMI','LAE')
PRIMARY5_INDEX = (0, 8, 6, 9, 3)
SCP_MAPPING = {**{s:i for i,s in enumerate(PATHOLOGY_CLASSES[:-1])}, 'LAO/LAE':9}
# Complete rhythm domain from pinned PTB-XL v1.0.3 scp_statements.csv.
OFFICIAL_RHYTHM_CODES = frozenset(('SR','AFIB','STACH','SARRH','SBRAD','PACE','SVARR','BIGU','AFLT','SVTAC','PSVT','TRIGU'))


def resolve_historical_ptb_ids(ids, metadata_ids):
    # PTB-XL v1.0.3 changelog, duplicate Change 4: 141 was retained as 2509.
    replacements={'141':'2509'}
    resolved=[]
    for rid in ids:
        candidate=rid if rid in metadata_ids else replacements.get(rid)
        if candidate not in metadata_ids:raise ValueError('Unresolved historical PTB-XL alias: '+rid)
        resolved.append(candidate)
    return resolved


def unambiguous_rhythm_target(scores,mapping):
    if any(k in OFFICIAL_RHYTHM_CODES and (k not in mapping or not 0<=int(mapping[k])<9) for k in scores):
        return -1
    present={int(mapping[k]) for k in scores if k in mapping}
    if any(index<0 or index>=9 for index in present):return -1
    abnormal=present-{3}
    if len(abnormal)==1:return next(iter(abnormal))
    if present=={3}:return 3
    return -1


def pathology_targets(dataset, scores):
    y = np.zeros(10, dtype=np.float32)
    mask = np.zeros(10, dtype=np.float32)
    if dataset != 'ptb-xl':
        return y, mask
    mask[:] = 1
    for code, score in scores.items():
        if code in SCP_MAPPING:
            score = float(score)
            if not np.isfinite(score) or not 0 <= score <= 100:
                raise ValueError('Invalid SCP diagnostic likelihood')
            # Match the existing registry: presence, including likelihood zero.
            y[SCP_MAPPING[code]] = 1
    return y, mask


def select_extra_training(metadata, excluded_patients, existing_record_ids):
    selected = []
    for row in metadata:
        rid = str(int(row['ecg_id']))
        if int(row['strat_fold']) > 8 or row['patient_id'] in excluded_patients or rid in existing_record_ids:
            continue
        y, _ = pathology_targets('ptb-xl', ast.literal_eval(row['scp_codes']))
        if y[list(PRIMARY5_INDEX)].any():
            selected.append(row)
    return sorted(selected, key=lambda r:int(r['ecg_id']))


def masked_pathology_loss(logits, y, mask, positive_weights):
    import torch
    from torch.nn import functional as F

    if logits.shape != y.shape or mask.shape != y.shape or logits.shape[-1] != 10:
        raise ValueError('Pathology output/target/mask contract differs')
    if not torch.isfinite(y).all() or not ((y == 0) | (y == 1)).all() or not ((mask == 0) | (mask == 1)).all():
        raise ValueError('Targets and masks must be binary')
    loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=positive_weights, reduction='none')
    return (loss*mask).sum()/mask.sum().clamp_min(1)


def _validate_pathology_arrays(y,p):
    if y.ndim != 2 or y.shape[1] != 10 or p.shape != y.shape or len(y) == 0:
        raise ValueError('Invalid pathology evaluation dimensions')
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any() or not np.isin(y,[0,1]).all():
        raise ValueError('Invalid pathology probabilities/targets')


def calibrate_thresholds(y,p,*,role):
    if role != 'validation':
        raise ValueError('Threshold fitting requires validation, never test')
    _validate_pathology_arrays(y,p)
    if (y.sum(0) == 0).any() or (y.sum(0) == len(y)).any():
        raise ValueError('Calibration requires positive and negative examples for every class')
    thresholds=[]
    for j in range(10):
        options=[]
        for threshold in np.linspace(.05,.95,19):
            prediction=p[:,j]>=threshold
            tp=int((prediction & (y[:,j]==1)).sum())
            f1=2*tp/max(1,int(prediction.sum()+y[:,j].sum()))
            options.append((f1,float(threshold)))
        thresholds.append(max(options)[1])
    return np.asarray(thresholds,dtype=np.float64)


def pathology_metrics(y,p,thresholds):
    _validate_pathology_arrays(y,p)
    thresholds=np.asarray(thresholds)
    if thresholds.shape != (10,) or not np.isfinite(thresholds).all() or ((thresholds<0)|(thresholds>1)).any():
        raise ValueError('Invalid frozen pathology thresholds')
    pred=p>=thresholds
    tp=(pred & (y==1)).sum(0); fp=(pred & (y==0)).sum(0)
    fn=(~pred & (y==1)).sum(0); tn=(~pred & (y==0)).sum(0)
    per_class={}
    for j,name in enumerate(PATHOLOGY_CLASSES):
        per_class[name]={'positive_support':int(tp[j]+fn[j]),'negative_support':int(tn[j]+fp[j]),
                         'tp':int(tp[j]),'fp':int(fp[j]),'fn':int(fn[j]),'tn':int(tn[j]),
                         'f1':float(2*tp[j]/max(1,2*tp[j]+fp[j]+fn[j])),
                         'accuracy':float((tp[j]+tn[j])/len(y)),
                         'recall':float(tp[j]/max(1,tp[j]+fn[j])),
                         'specificity':float(tn[j]/max(1,tn[j]+fp[j])),
                         'threshold':float(thresholds[j])}
    primary=[per_class[PATHOLOGY_CLASSES[j]] for j in PRIMARY5_INDEX]
    return {'records':len(y),'primary5_mean_label_accuracy':float(np.mean([r['accuracy'] for r in primary])),
            'primary5_macro_f1':float(np.mean([r['f1'] for r in primary])),
            'primary5_macro_recall':float(np.mean([r['recall'] for r in primary])), 'per_class':per_class}
