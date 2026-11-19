"""Training-only sampling and masked losses for the unchanged TITAN network."""
from collections import Counter

import numpy as np

try:
    from joint_metrics import PRIMARY5_INDEX
except ModuleNotFoundError:
    from titan_v4.reproducibility.joint_refinement import PRIMARY5_INDEX


def _arrays(y,mask):
    if y.ndim!=2 or y.shape[1]!=10 or y.shape!=mask.shape or not np.isin(y,[0,1]).all() or not np.isin(mask,[0,1]).all():
        raise ValueError('Binary ten-column target/mask contract differs')
    if not np.all(mask==mask[:,0,None]):raise ValueError('Partial source mask is not supported')


def pathology_sampling_probabilities(y,mask,*,role):
    if role!='train':raise ValueError('Sampling requires train targets only')
    _arrays(y,mask)
    primary=y[:,list(PRIMARY5_INDEX)];known=mask[:,0]==1
    positive=known & (primary.sum(1)>0);negative=known & ~positive
    if not positive.any() or not negative.any():raise ValueError('Need positive and negative training groups')
    support=primary[known].sum(0)
    inverse=np.divide(1.,support,out=np.zeros_like(support,dtype=np.float64),where=support>0)
    weights=np.zeros(len(y),np.float64)
    weights[positive]=(primary[positive]*inverse).sum(1)
    weights[positive]*=.5/weights[positive].sum()
    weights[negative]=.5/negative.sum()
    return weights


def rhythm_training_weights(y,dataset,*,role):
    if role!='train':raise ValueError('Rhythm weighting requires train targets only')
    if y.ndim!=1 or dataset.shape!=y.shape or not np.isin(y,range(-1,9)).all() or not (y>=0).any():
        raise ValueError('Invalid rhythm training references')
    known=y>=0;counts=Counter(zip(dataset[known].tolist(),y[known].tolist()))
    w=np.zeros(len(y),np.float32)
    for i in np.flatnonzero(known):w[i]=1/counts[(str(dataset[i]),int(y[i]))]
    w/=w[known].mean()
    return w


def sample_pathology_indices(probabilities,size,*,generator):
    import torch

    if probabilities.device.type!='cpu' or generator.device.type!='cpu':
        raise ValueError('Strict deterministic sampling requires a dedicated CPU generator')
    if probabilities.ndim!=1 or size<1 or not torch.isfinite(probabilities).all() or (probabilities<0).any() or abs(float(probabilities.sum())-1)>2e-6:
        raise ValueError('Invalid sampling probabilities')
    return torch.multinomial(probabilities,size,replacement=True,generator=generator)


def agreement_anchor_loss(logits,reference,targets):
    import torch
    from torch.nn import functional as F

    if logits.shape!=reference.shape or targets.shape!=(len(logits),):raise ValueError('Anchor arrays differ')
    if not torch.isfinite(reference).all() or (reference<0).any() or not torch.allclose(reference.sum(1),torch.ones(len(reference),device=reference.device),atol=2e-6,rtol=0):
        raise ValueError('Invalid reference probabilities')
    keep=(targets>=0) & (reference.argmax(1)==targets)
    if not keep.any():return logits.sum()*0
    return F.kl_div(F.log_softmax(logits[keep],1),reference[keep],reduction='batchmean')


def asymmetric_pathology_loss(logits,y,mask,*,gamma_negative=4.,clip=.05,label_weights=None):
    import math
    import torch
    from torch.nn import functional as F

    if logits.shape!=y.shape or y.shape!=mask.shape or y.ndim!=2 or y.shape[1]!=10:
        raise ValueError('Ten-output pathology contract differs')
    if not torch.isfinite(y).all() or not ((y==0)|(y==1)).all() or not ((mask==0)|(mask==1)).all():
        raise ValueError('Targets and masks must be binary')
    if gamma_negative<0 or not 0<=clip<1:raise ValueError('Invalid asymmetric-loss configuration')
    if gamma_negative==0 and clip==0:
        loss=F.binary_cross_entropy_with_logits(logits,y,reduction='none')
    else:
        # Log-space probabilities preserve gradients for extremely wrong positives.
        log_negative=F.logsigmoid(-logits)
        if clip:log_negative=torch.logaddexp(log_negative,torch.full_like(logits,math.log(clip))).clamp(max=0)
        focal=(1-log_negative.exp()).clamp_min(0).pow(gamma_negative)
        loss=y*F.softplus(-logits)-(1-y)*focal*log_negative
    weight=torch.ones(10,device=logits.device,dtype=logits.dtype) if label_weights is None else label_weights
    if weight.shape!=(10,) or not torch.isfinite(weight).all() or (weight<=0).any():raise ValueError('Invalid label weights')
    effective=mask*weight
    return (loss*effective).sum()/effective.sum().clamp_min(1)


def select_candidates(candidates,baseline,*,role):
    if role!='validation':raise ValueError('Selection requires validation, never follow-up')
    eligible=[r for r in candidates if r['eligible']]
    path=baseline['pathology']['primary5_macro_f1'];rhythm=baseline['rhythm_score']
    joint=[r for r in eligible if r['rhythm_score']>=rhythm-.002 and r['pathology']['primary5_macro_f1']>=path+.005]
    arr=[r for r in eligible if r['rhythm_score']>=rhythm+.002 and r['pathology']['primary5_macro_f1']>=path-.005]
    return {'joint_selected':max(joint,key=lambda r:(r['joint_score'],r['name'])) if joint else None,
            'rhythm_selected':max(arr,key=lambda r:(r['rhythm_score'],r['name'])) if arr else None}
