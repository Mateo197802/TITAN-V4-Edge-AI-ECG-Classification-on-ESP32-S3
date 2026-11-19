"""Fit existing heads independently on frozen CEDIA features and audited references."""
import hashlib
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
from sklearn.metrics import average_precision_score

import contract as c
import baseline_trainer as b
from balanced_metrics import agreement_anchor_loss
from supervision_metrics import configure_role, assert_role_frozen, crossfit_pathology, select_components
from head_composition import compose_head_states

PROTOCOL=json.loads((c.ROOT/'protocol.json').read_text()); b.PROTOCOL=PROTOCOL


def features(model,x):
    model.eval(); parts=[]
    with torch.inference_mode():
        for start in range(0,len(x),128):
            parts.append(model.forward_features(torch.from_numpy(x[start:start+128]).cuda()).cpu().numpy())
    return torch.from_numpy(np.concatenate(parts)).cuda()


def outputs(model,z):
    model.eval(); parts=[[],[],[],[]]
    with torch.inference_mode():
        for start in range(0,len(z),128):
            for p,head in zip(parts,(model.head_rhythm,model.head_quality,model.head_biometrics,model.head_pathology)):
                p.append(head(z[start:start+128]).cpu().numpy())
    return tuple(np.concatenate(p) for p in parts)


def report(val,rv,pval,pv,folds,base=None):
    rp=b.rhythm_prob(rv); sources=c.source_metrics(val,rp); metrics=c.metrics(val['y'],rp)
    pp=b.path_prob(pv); thresholds=b.calibrate_thresholds(pval['targets'],pp,role='validation')
    path=b.pathology_metrics(pval['targets'],pp,thresholds)
    oof=crossfit_pathology(pval['targets'],pp,folds,role='validation')
    ap=float(np.mean([average_precision_score(pval['targets'][:,j],pp[:,j]) for j in b.PRIMARY5_INDEX]))
    source_score=float(np.mean([s['macro_f1'] for s in sources.values()]))
    checks={'protected_recalls':True if base is None else b.protected_recalls(sources,base['sources']),
            'rhythm_macro_preserved':base is None or metrics['macro_f1']>=base['metrics']['macro_f1']-.005-1e-12,
            'rhythm_sources_preserved':base is None or source_score>=base['rhythm_score']-.005-1e-12,
            'quality_biometrics_preserved':True}
    return {'metrics':metrics,'sources':sources,'rhythm_score':source_score,'pathology':path,'thresholds':thresholds.tolist(),
            'joint_score':source_score+.5*path['primary5_macro_f1'],'pathology_oof':oof,'pathology_ap':ap,
            'quality_biometrics_nrmse':{'quality':0.,'biometrics':0.},'preservation_checks':checks,'eligible':all(checks.values())}


def save_validation(out,name,val,rv,pval,pv,folds):
    np.savez_compressed(out/(name+'_validation.npz'),rhythm_probabilities=b.rhythm_prob(rv),pathology_probabilities=b.path_prob(pv),
                        y=val['y'],dataset=val['dataset'],ecg_id=val['ecg_id'],path_targets=pval['targets'],path_mask=pval['mask'],
                        path_ecg_id=pval['ecg_id'],path_patient_fold=folds,quality=rv[1],biometrics=rv[2])


def run_trial(trial,epochs,z,train,targets,mask,controls,val,pval,vz,pz,initial,start_prob,base,out):
    seed=PROTOCOL['optimization']['seed']; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    model=c.load_model(c.ROOT/'start.pth').cuda(); params=configure_role(model,trial['task'])
    if sum(p.numel() for p in params)!=(19401 if trial['task']=='rhythm' else 19498):raise ValueError('Head parameter count differs')
    optimizer=torch.optim.AdamW(params,lr=trial['lr'],weight_decay=.01)
    if trial['task']=='rhythm':
        chosen=np.arange(len(z)); y=torch.from_numpy(controls['training_y']).cuda()
        weights=torch.from_numpy(controls['rhythm_weights']).cuda(); teacher=torch.from_numpy(train['teacher']).cuda()
    else:
        chosen=np.flatnonzero(mask[:,0]==1)
        yy=torch.from_numpy(targets[chosen]).cuda()
        key='certainty_weights' if trial['weighting']=='certainty' else 'human_weights'
        confidence=np.ones_like(yy.cpu().numpy()) if trial['weighting']=='natural' else controls[key][chosen]
        confidence=torch.from_numpy(confidence).cuda()
        label_weight=torch.ones(10,device='cuda'); label_weight[list(b.PRIMARY5_INDEX)]=2
        positive_weight=torch.ones(10,device='cuda')
        if trial['weighting']=='human_positive':
            positives=(confidence*yy).sum(0); negatives=(confidence*(1-yy)).sum(0)
            positive_weight=(negatives/positives.clamp_min(1)).sqrt().clamp(1,4)
    zz=z[torch.from_numpy(chosen).cuda()]
    history=[]; records=[]; state=None
    for epoch in range(1,epochs+1):
        configure_role(model,trial['task'])
        lr=trial['lr']*(.1+.9*.5*(1+math.cos(math.pi*(epoch-1)/PROTOCOL['optimization']['epochs'])))
        for group in optimizer.param_groups: group['lr']=lr
        order=torch.randperm(len(zz),device='cuda'); losses=[]
        for start in range(0,len(order),128):
            ids=order[start:start+128]
            logits=getattr(model,'head_'+trial['task'])(zz[ids])
            if trial['task']=='rhythm':
                known=y[ids]>=0; rid=ids[known]
                loss=logits.sum()*0
                if known.any():
                    loss=(F.cross_entropy(logits[known],y[rid],reduction='none')*weights[rid]).mean()
                    original=rid<len(train['y'])
                    if original.any():loss=loss+.15*agreement_anchor_loss(logits[known][original],teacher[rid[original]],y[rid[original]])
                loss=loss+.1*agreement_anchor_loss(logits,start_prob[ids],y[ids])
            else:
                element=F.binary_cross_entropy_with_logits(logits,yy[ids],pos_weight=positive_weight,reduction='none')
                weight=confidence[ids]*label_weight
                loss=(element*weight).sum()/weight.sum().clamp_min(1)
            if not torch.isfinite(loss):raise ValueError('Non-finite head loss')
            optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(params,1); optimizer.step()
            losses.append(float(loss.detach()))
        assert_role_frozen(model,initial,trial['task'])
        history.append({'epoch':epoch,'lr':lr,'mean_loss':float(np.mean(losses)),'training_records':len(zz)})
        if epoch not in PROTOCOL['optimization']['checkpoints'] and epoch!=epochs:continue
        rv,pv=outputs(model,vz),outputs(model,pz); name=trial['name']+f'_e{epoch:02d}'
        r={'name':name,'trial':trial,'epoch':epoch,**report(val,rv,pval,pv,controls['path_validation_folds'],base)}
        records.append(r); state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        print(json.dumps({'candidate':name,'rhythm_accuracy':r['metrics']['accuracy'],'pathology_macro':r['pathology']['primary5_macro_f1'],
                          'patient_oof_macro':r['pathology_oof']['primary5_macro_f1'],'pathology_ap':r['pathology_ap'],'eligible':r['eligible']}),flush=True)
        if out:
            torch.save({'state_dict':state,'classes':c.CLASSES,'pathology_classes':b.PATHOLOGY_CLASSES,'parameters':c.PARAMETERS},out/(name+'.pth'))
            save_validation(out,name,val,rv,pval,pv,controls['path_validation_folds'])
    return records,history,state


def main():
    c.require_cedia_gpu(); os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'; torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False; torch.set_num_threads(2)
    for name,digest in json.loads((c.ROOT/'INPUT_SHA256.json').read_text()).items():
        if c.digest(c.ROOT/name)!=digest:raise ValueError('Submitted input differs: '+name)
    if c.digest(c.ROOT/'start.pth')!=PROTOCOL['start_sha256']:raise ValueError('Starting checkpoint differs')
    train,val,extra,pval,directed=[dict(np.load(c.ROOT/name,allow_pickle=False)) for name in ('train.npz','validation.npz','path_extra.npz','path_validation.npz','rhythm_extra.npz')]
    controls=dict(np.load(c.ROOT/'supervision_weights.npz',allow_pickle=False))
    for data,count in ((train,8393),(val,1476),(extra,13984),(pval,2027),(directed,190)):
        if data['x'].shape!=(count,6,1250) or data['x'].dtype!=np.float32 or not np.isfinite(data['x']).all():raise ValueError('Source tensor contract differs')
        for x,digest in zip(data['x'],data['tensor_sha256']):
            if hashlib.sha256(x.astype('<f4',copy=False).tobytes()).hexdigest()!=str(digest):raise ValueError('Source tensor hash differs')
    with np.load(c.ROOT/'path_targets.npz',allow_pickle=False) as paths:
        targets=np.concatenate((paths['train_targets'],extra['targets'],np.zeros((190,10),np.float32)))
        mask=np.concatenate((paths['train_mask'],extra['mask'],np.zeros((190,10),np.float32)))
    if not np.array_equal(controls['training_y'],np.concatenate((train['y'],extra['y'],directed['y']))):raise ValueError('Training references differ')
    model=c.load_model(c.ROOT/'start.pth').cuda(); initial={k:v.cpu().clone() for k,v in model.state_dict().items()}
    z=features(model,np.concatenate((train['x'],extra['x'],directed['x']))); vz=features(model,val['x']); pz=features(model,pval['x'])
    rv,pv=outputs(model,vz),outputs(model,pz); rawrv,rawpv=b.outputs(model,val['x']),b.outputs(model,pval['x'])
    if any(not np.array_equal(a,z) for a,z in zip(rv+pv,rawrv+rawpv)):raise ValueError('Cached features do not exactly reproduce raw inference')
    start_prob=torch.from_numpy(b.rhythm_prob(outputs(model,z))).cuda()
    base=report(val,rv,pval,pv,controls['path_validation_folds'])
    out=c.ROOT/'results'; out.mkdir(exist_ok=False)
    c.save(out/'execution.json',{'job_id':os.environ['SLURM_JOB_ID'],'hostname':socket.gethostname(),'python':platform.python_version(),
           'torch':torch.__version__,'numpy':np.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),'parameters':c.PARAMETERS,
           'input_manifest_sha256':c.digest(c.ROOT/'INPUT_SHA256.json'),'features_reproduce_raw_outputs_bitwise':True,
           'feature_sha256':{name:hashlib.sha256(a.cpu().numpy().astype('<f4').tobytes()).hexdigest() for name,a in [('train',z),('validation',vz),('path_validation',pz)]}})
    save_validation(out,'baseline',val,rv,pval,pv,controls['path_validation_folds'])
    records=[]; histories={}
    for trial in PROTOCOL['optimization']['trials']:
        r,h,_=run_trial(trial,20,z,train,targets,mask,controls,val,pval,vz,pz,initial,start_prob,base,out)
        records.extend(r); histories[trial['name']]=h
    chosen=select_components(records,base,role='validation'); refits={}; states={}
    for task,record in chosen.items():
        if record is None:states[task]=initial; refits[task]={'needed':False}
        else:
            _,_,state=run_trial(record['trial'],record['epoch'],z,train,targets,mask,controls,val,pval,vz,pz,initial,start_prob,base,None)
            old=torch.load(out/(record['name']+'.pth'),map_location='cpu',weights_only=True)['state_dict']
            if any(not torch.equal(v,state[k]) for k,v in old.items()):raise ValueError('Selected head refit differs')
            states[task]=state; refits[task]={'needed':True,'candidate':record['name'],'all_state_tensors_bitwise_equal':True}
    composed=compose_head_states(states['rhythm_component'],states['pathology_component'])
    for name,state in [('joint_selected',composed),('rhythm_selected',states['rhythm_component']),('composed_selected',composed)]:
        torch.save({'state_dict':state,'classes':c.CLASSES,'pathology_classes':b.PATHOLOGY_CLASSES,'parameters':c.PARAMETERS},out/(name+'.pth'))
    model=c.load_model(out/'joint_selected.pth').cuda(); assert_role_frozen(model,initial,'both')
    crv,cpv=outputs(model,vz),outputs(model,pz)
    combined={'name':'composed_selected','kind':'composition','trial':{'name':'composition','task':'both'},'epoch':0,
              'components':{k:r['name'] if r else 'start' for k,r in chosen.items()},**report(val,crv,pval,cpv,controls['path_validation_folds'],base)}
    save_validation(out,'composed_selected',val,crv,pval,cpv,controls['path_validation_folds'])
    selection={'baseline':base,'candidates':records+[combined],**chosen,'joint_selected':combined,'rhythm_selected':chosen['rhythm_component'],
               'selection_data':'validation only','thresholds_fitted_on_validation_only':True,'evaluation_labels_loaded_after_selection':True,
               'references_changed':False,'checkpoints':{n:c.digest(out/(n+'.pth')) for n in ('joint_selected','rhythm_selected')}}
    c.save(out/'selection.json',selection); c.save(out/'training_histories.json',histories); c.save(out/'refit.json',refits)
    data=dict(np.load(c.ROOT/'evaluation.npz',allow_pickle=False))
    with np.load(c.ROOT/'path_targets.npz',allow_pickle=False) as paths:ey,em=paths['followup_targets'],paths['followup_mask']
    for name,record in [('baseline',base),('joint_selected',combined),('rhythm_selected',chosen['rhythm_component'] or base)]:
        model=c.load_model(c.ROOT/'start.pth' if name=='baseline' else out/(name+'.pth')).cuda()
        b.export_predictions(out,name,model,data,ey,em,np.asarray(record['thresholds']))
    c.save(out/'SHA256SUMS.json',{p.name:c.digest(p) for p in sorted(out.iterdir()) if p.is_file()})
    print(json.dumps({'COMPLETED':True,'components':combined['components']}),flush=True)


if __name__=='__main__':main()
