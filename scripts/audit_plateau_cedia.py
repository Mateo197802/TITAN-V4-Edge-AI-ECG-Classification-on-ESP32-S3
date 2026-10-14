"""Read-only CEDIA GPU inference to separate training fit from validation performance."""
import json
import os
import platform
import socket

import numpy as np
import torch
from sklearn.metrics import average_precision_score

import contract as c
from joint_metrics import PRIMARY5_INDEX, calibrate_thresholds, pathology_metrics


def predict(model,x):
    rhythm=[];path=[]
    with torch.inference_mode():
        for start in range(0,len(x),128):
            values=model(torch.from_numpy(x[start:start+128]).cuda())
            rhythm.append(torch.softmax(values[0],1).cpu().numpy())
            path.append(torch.sigmoid(values[3]).cpu().numpy())
    return np.concatenate(rhythm),np.concatenate(path)


def report(y,rp,py,pp,mask,thresholds):
    known=y>=0; observed=mask[:,0]==1
    return {'rhythm':c.metrics(y[known],rp[known]),
            'pathology':pathology_metrics(py[observed],pp[observed],thresholds),
            'pathology_primary5_ap':float(np.mean([average_precision_score(py[observed,j],pp[observed,j]) for j in PRIMARY5_INDEX]))}


def main():
    c.require_cedia_gpu();torch.set_num_threads(2)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    manifest=json.loads((c.ROOT/'INPUT_SHA256.json').read_text())
    for name,digest in manifest.items():
        if c.digest(c.ROOT/name)!=digest:raise ValueError('Frozen diagnostic input differs: '+name)
    protocol=json.loads((c.ROOT/'protocol.json').read_text())
    train,val,extra,pval,directed=[dict(np.load(c.ROOT/name,allow_pickle=False)) for name in ('train.npz','validation.npz','path_extra.npz','path_validation.npz','rhythm_extra.npz')]
    with np.load(c.ROOT/'path_targets.npz',allow_pickle=False) as paths:
        py=np.concatenate((paths['train_targets'],extra['targets'],np.zeros((190,10),np.float32)))
        mask=np.concatenate((paths['train_mask'],extra['mask'],np.zeros((190,10),np.float32)))
    x=np.concatenate((train['x'],extra['x'],directed['x']))
    y=np.concatenate((train['y'],extra['y'],directed['y']))
    source=np.concatenate((train['dataset'],np.full(len(extra['y']),'ptb-xl'),np.full(len(directed['y']),'ecg-arrhythmia')))
    ids=np.concatenate((train['ecg_id'],extra['ecg_id'],directed['ecg_id']))
    out=c.ROOT/'results';out.mkdir(exist_ok=False)
    summary={};arrays={'training_y':y,'training_dataset':source,'training_ecg_id':ids,'training_path_targets':py,'training_path_mask':mask,
                       'validation_y':val['y'],'validation_dataset':val['dataset'],'validation_ecg_id':val['ecg_id'],
                       'validation_path_targets':pval['targets'],'validation_path_ecg_id':pval['ecg_id']}
    for name,digest in protocol['checkpoints'].items():
        checkpoint=c.ROOT/(name+'.pth')
        if c.digest(checkpoint)!=digest:raise ValueError('Frozen diagnostic checkpoint differs')
        model=c.load_model(checkpoint).cuda()
        before={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        tr,tp=predict(model,x);vr,_=predict(model,val['x']);_,vp=predict(model,pval['x'])
        thresholds=calibrate_thresholds(pval['targets'],vp,role='validation')
        for k,v in model.state_dict().items():
            if not torch.equal(v.detach().cpu(),before[k]):raise ValueError('Read-only inference changed state')
        training=report(y,tr,py,tp,mask,thresholds)
        validation={'rhythm':c.metrics(val['y'],vr),'sources':c.source_metrics(val,vr),
                    'pathology':pathology_metrics(pval['targets'],vp,thresholds),
                    'pathology_primary5_ap':float(np.mean([average_precision_score(pval['targets'][:,j],vp[:,j]) for j in PRIMARY5_INDEX]))}
        training['sources']={s:c.metrics(y[(source==s)&(y>=0)],tr[(source==s)&(y>=0)]) for s in sorted(set(source.tolist()))}
        summary[name]={'checkpoint_sha256':digest,'training':training,'validation':validation,'thresholds':thresholds.tolist(),'state_unchanged':True}
        arrays.update({name+'_train_rhythm':tr,name+'_train_path':tp,name+'_val_rhythm':vr,name+'_val_path':vp})
        print(json.dumps({'model':name,'train_rhythm':training['rhythm']['accuracy'],'val_rhythm':validation['rhythm']['accuracy'],
                          'train_path_ap':training['pathology_primary5_ap'],'val_path_ap':validation['pathology_primary5_ap']}),flush=True)
        del model
    np.savez_compressed(out/'diagnostic_probabilities.npz',**arrays)
    c.save(out/'diagnostic.json',{'role':'diagnostic only; no optimizer, fitting or model selection','followup_read':False,'models':summary})
    c.save(out/'execution.json',{'job_id':os.environ['SLURM_JOB_ID'],'hostname':socket.gethostname(),'python':platform.python_version(),'torch':torch.__version__,
                               'numpy':np.__version__,'gpu':torch.cuda.get_device_name(0),'parameters':c.PARAMETERS,'input_manifest_sha256':c.digest(c.ROOT/'INPUT_SHA256.json')})
    c.save(out/'SHA256SUMS.json',{p.name:c.digest(p) for p in sorted(out.iterdir()) if p.is_file()})


if __name__=='__main__':main()
