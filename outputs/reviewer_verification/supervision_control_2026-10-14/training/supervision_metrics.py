"""Training-only source/annotation weights and patient-held-out calibration."""
from collections import Counter
import hashlib

import numpy as np

try:
    from joint_metrics import calibrate_thresholds, PRIMARY5_INDEX
except ModuleNotFoundError:
    from titan_v4.reproducibility.joint_refinement import calibrate_thresholds, PRIMARY5_INDEX


def _rhythm(y, source, role):
    if role != 'train' or y.ndim != 1 or y.shape != source.shape or not np.isin(y, range(-1,9)).all() or not (y>=0).any():
        raise ValueError('Training-only rhythm reference contract differs')


def source_masses(y, source, *, role):
    _rhythm(y, source, role)
    pairs = set(zip(source[y>=0].tolist(), y[y>=0].tolist()))
    groups = Counter(s for s, _ in pairs)
    return {s: groups[s]/len(pairs) for s in sorted(groups)}


def fixed_source_weights(y, source, masses, *, role):
    _rhythm(y, source, role)
    if any(not np.isfinite(v) or v<=0 for v in masses.values()) or abs(sum(masses.values())-1)>1e-10:
        raise ValueError('Invalid frozen source masses')
    counts = Counter(zip(source[y>=0].tolist(), y[y>=0].tolist()))
    classes = Counter(s for s, _ in counts)
    if set(classes) != set(masses):
        raise ValueError('Source population differs from frozen budget')
    weights = np.zeros(len(y),np.float32)
    n = int((y>=0).sum())
    for i in np.flatnonzero(y>=0):
        s = str(source[i]); weights[i] = n*masses[s]/(classes[s]*counts[(s,int(y[i]))])
    return weights


def confidence_weights(targets, mask, human, likelihood, *, role):
    if role != 'train' or targets.ndim!=2 or targets.shape[1]!=10 or mask.shape!=targets.shape or likelihood.shape!=targets.shape or human.shape!=(len(targets),):
        raise ValueError('Training-only annotation weight contract differs')
    if not np.isin(targets,[0,1]).all() or not np.isin(mask,[0,1]).all() or not np.isin(human,[True,False]).all():
        raise ValueError('Invalid source reference/mask/flag')
    if not np.isfinite(likelihood).all() or ((likelihood<0)|(likelihood>100)).any():
        raise ValueError('Invalid source likelihood')
    h = np.where(human,1.,.5).astype(np.float32)[:,None]*mask
    certainty = np.where((targets==1)&(likelihood>0), .5+.5*likelihood/100, 1.)
    return h.astype(np.float32), (h*certainty).astype(np.float32)


def patient_folds(tokens):
    return np.asarray([int(hashlib.sha256(('titan-oof-20261014:'+str(t)).encode()).hexdigest()[:16],16)%5 for t in tokens],np.int64)


def crossfit_pathology(y, p, folds, *, role):
    if role!='validation' or folds.shape!=(len(y),) or set(folds.tolist())!=set(range(5)):
        raise ValueError('Patient-fold validation contract differs')
    predictions = np.zeros(y.shape,bool); records=[]
    for fold in range(5):
        held = folds==fold; fitted=~held
        thresholds = calibrate_thresholds(y[fitted],p[fitted],role='validation')
        predictions[held] = p[held]>=thresholds
        records.append({'fold':fold, 'held_records':int(held.sum()), 'calibration_records':int(fitted.sum()),
                        'calibration_indices_sha256':hashlib.sha256(np.flatnonzero(fitted).astype('<i8').tobytes()).hexdigest(),
                        'thresholds':thresholds.tolist()})
    tp=(predictions&(y==1)).sum(0); fp=(predictions&(y==0)).sum(0); fn=(~predictions&(y==1)).sum(0)
    f1=2*tp/np.maximum(1,2*tp+fp+fn)
    return {'records':len(y), 'primary5_macro_f1':float(f1[list(PRIMARY5_INDEX)].mean()),
            'per_label_f1':f1.tolist(),'folds':records,'selection_scope':'validation patient groups only'}


def configure_role(model, role):
    if role not in ('rhythm','pathology'): raise ValueError('Invalid head role')
    model.eval()
    for name,p in model.named_parameters(): p.requires_grad_(name.startswith('head_'+role+'.'))
    getattr(model,'head_'+role).train()
    return [p for p in model.parameters() if p.requires_grad]


def assert_role_frozen(model, initial, role):
    import torch
    allowed=('head_rhythm.','head_pathology.') if role=='both' else ('head_'+role+'.',)
    state=model.state_dict()
    if set(state)!=set(initial): raise ValueError('State keys changed')
    for name,value in state.items():
        if value.shape!=initial[name].shape or value.dtype!=initial[name].dtype: raise ValueError('Architecture changed')
        if not name.startswith(allowed) and not torch.equal(value.detach().cpu(),initial[name]):
            raise ValueError('Frozen parameter/buffer changed: '+name)


def select_components(records, base, *, role):
    if role!='validation': raise ValueError('Selection requires validation only')
    arr=[r for r in records if r['trial']['task']=='rhythm' and r['eligible']
         and r['metrics']['macro_f1']>=base['metrics']['macro_f1']+.002
         and r['rhythm_score']>=base['rhythm_score']+.002]
    path=[r for r in records if r['trial']['task']=='pathology' and r['eligible']
          and r['pathology_oof']['primary5_macro_f1']>=base['pathology_oof']['primary5_macro_f1']+.005
          and r['pathology_ap']>=base['pathology_ap']-.005
          and r['pathology']['primary5_macro_f1']>=base['pathology']['primary5_macro_f1']]
    return {'rhythm_component':max(arr,key=lambda r:(r['rhythm_score'],r['metrics']['macro_f1'],r['name'])) if arr else None,
            'pathology_component':max(path,key=lambda r:(r['pathology_oof']['primary5_macro_f1'],r['pathology_ap'],r['name'])) if path else None}
