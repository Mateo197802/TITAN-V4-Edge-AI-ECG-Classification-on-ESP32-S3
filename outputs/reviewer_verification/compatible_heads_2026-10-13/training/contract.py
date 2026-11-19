"""Original Gold contract and validation metrics shared by local audit and CEDIA."""
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import socket
import numpy as np
import torch
from titan_v4_net import TitanV4Lite

ROOT=Path(__file__).resolve().parent
CLASSES=['AFIB','SB','STACH','NSR','PVC','RBBB','LBBB','PAC','1AVB']
PARAMETERS=1143751


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def save(path,value):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise ValueError('Preserve existing evidence: '+str(path))
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def parameter_count(model):return sum(p.numel() for p in model.parameters())


def load_model(path):
    state=torch.load(path,map_location='cpu',weights_only=True)
    model=TitanV4Lite(in_channels=6,num_rhythm=9,num_pathology=10,morphology_dim=0)
    if isinstance(state,dict) and 'model_state_dict' in state:state=state['model_state_dict']
    elif isinstance(state,dict) and 'state_dict' in state:state=state['state_dict']
    model.load_state_dict(state,strict=True)
    if parameter_count(model)!=PARAMETERS:raise ValueError('Parameter contract changed')
    return model.eval()


def configure(model):
    model.eval()
    for name,p in model.named_parameters():p.requires_grad_(name.startswith('head_rhythm.'))
    model.head_rhythm.train()


def check_frozen(model,reference):
    state=model.state_dict()
    if set(state)!=set(reference):raise ValueError('State keys changed')
    for key,value in state.items():
        if value.shape!=reference[key].shape:raise ValueError('Shape changed: '+key)
        if not key.startswith('head_rhythm.') and not torch.equal(value.cpu(),reference[key].cpu()):
            raise ValueError('Frozen parameter/buffer changed: '+key)


def require_cedia_gpu():
    if not os.environ.get('SLURM_JOB_ID') or not socket.gethostname().startswith('compute-'):
        raise RuntimeError('Gradient execution requires a CEDIA Slurm compute allocation')
    if not torch.cuda.is_available():raise RuntimeError('No GPU; CPU training fallback prohibited')


def validate_role(role):
    if role not in ('train','validation'):raise ValueError('Evaluation data cannot enter fitting or selection')


def validate_probabilities(p):
    if p.ndim!=2 or p.shape[1]!=9 or not np.isfinite(p).all() or (p<0).any():
        raise ValueError('Invalid nine-class probabilities')
    if not np.allclose(p.sum(1),1,atol=2e-6,rtol=0):raise ValueError('Probabilities do not sum to one')


def sample_weights(data):
    counts=Counter(zip(data['dataset'].tolist(),data['y'].tolist()))
    return np.array([len(data['y'])/(len(counts)*counts[(s,int(y))])
                     for s,y in zip(data['dataset'],data['y'])],dtype=np.float32)


def metrics(y,p):
    validate_probabilities(p)
    cm=np.zeros((9,9),dtype=np.int64)
    np.add.at(cm,(y,p.argmax(1)),1)
    support=cm.sum(1);predicted=cm.sum(0);tp=cm.diagonal()
    recall=np.divide(tp,support,out=np.zeros(9),where=support>0)
    f1=np.divide(2*tp,support+predicted,out=np.zeros(9),where=support+predicted>0)
    return {'n':len(y),'correct':int(tp.sum()),'accuracy':float(tp.sum()/len(y)),
        'macro_f1':float(f1.mean()),'weighted_f1':float((f1*support).sum()/len(y)),
        'confusion_matrix':cm.tolist(),'support':dict(zip(CLASSES,support.tolist())),
        'recall':dict(zip(CLASSES,recall.tolist()))}


def source_metrics(data,p):
    return {str(source):metrics(data['y'][idx],p[idx]) for source in np.unique(data['dataset'])
            if (idx:=data['dataset']==source).any()}


def validation_score(data,p,gold):
    values=source_metrics(data,p);baseline=source_metrics(data,gold)
    # Preserve rare classes and rate classification within five percentage points.
    eligible=all(v['recall'][label]>=baseline[source]['recall'][label]-.05-1e-12
        for source,v in values.items() for label in ('SB','LBBB') if v['support'][label]>0)
    return float(np.mean([v['macro_f1'] for v in values.values()])),bool(eligible),values
