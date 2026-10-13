"""Controlled continued fitting of the unchanged four-head TITAN, on CEDIA only."""
import json
import math
import os
from pathlib import Path
import platform
import random
import socket
import time

import numpy as np
import torch
from torch.nn import functional as F

import contract as c
import baseline_trainer as b
from balanced_metrics import pathology_sampling_probabilities, rhythm_training_weights, agreement_anchor_loss, asymmetric_pathology_loss, select_candidates, sample_pathology_indices

b.PROTOCOL=json.loads((c.ROOT/'protocol.json').read_text())


def starting_anchor(streams):
    coefficient=b.PROTOCOL['optimization']['conditional_start_kl_weight']
    return coefficient*torch.stack([agreement_anchor_loss(*stream) for stream in streams]).mean()


def run_trial(trial,epochs,train,extra,validation,paths,initial,train_ref,extra_ref,val_ref,base,out):
    if trial['mode']=='legacy':
        return b.run_trial(trial,epochs,train,extra,validation,paths,initial,train_ref,extra_ref,val_ref,base,out)
    protocol=b.PROTOCOL;seed=protocol['optimization']['seed']
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    model=c.load_model(c.ROOT/'start.pth').cuda();params=b.configure(model,trial['blocks'],True)
    expected={0:19498,2:778943,5:1118787}[trial['blocks']]
    if sum(p.numel() for p in params)!=expected:raise ValueError('Trainable parameter contract changed')
    optimizer=torch.optim.AdamW(params,lr=trial['lr'],weight_decay=.01)
    x=torch.from_numpy(np.concatenate((train['x'],extra['x']))).cuda()
    yy=np.concatenate((train['y'],extra['y']));y=torch.from_numpy(yy).cuda();n=len(train['y'])
    py=np.concatenate((paths['train_targets'],extra['targets']));pm=np.concatenate((paths['train_mask'],extra['mask']))
    target=torch.from_numpy(py).cuda();mask=torch.from_numpy(pm).cuda()
    sampler=torch.from_numpy(pathology_sampling_probabilities(py,pm,role='train'))
    sampling_rng=torch.Generator(device='cpu').manual_seed(seed)
    source=np.concatenate((train['dataset'],np.full(len(extra['x']),'ptb-xl')))
    weights=torch.from_numpy(rhythm_training_weights(yy,source,role='train')).cuda()
    reference=tuple(np.concatenate((a,z)) for a,z in zip(train_ref,extra_ref))
    start_prob=torch.from_numpy(b.rhythm_prob(reference)).cuda();teacher=torch.from_numpy(train['teacher']).cuda()
    auxiliary=tuple(torch.from_numpy(a).cuda() for a in reference[1:3])
    scales=tuple(a.std(0,unbiased=False).clamp_min(.05) for a in auxiliary)
    label_weights=torch.ones(10,device='cuda');label_weights[list(b.PRIMARY5_INDEX)]=2
    pw=(((1-target)*mask).sum(0)/(target*mask).sum(0)).sqrt().clamp(1,10)
    known_ids=torch.nonzero(y>=0).flatten();observed=int(pm[:,0].sum())
    shared=[p for name,p in model.named_parameters() if p.requires_grad and name.startswith(('stem.','res_layers.'))]
    history=[];records=[];state=None
    for epoch in range(1,epochs+1):
        b.configure(model,trial['blocks'],True)
        # Refit always uses the originally declared 40-epoch schedule.
        factor=.1+.9*.5*(1+math.cos(math.pi*(epoch-1)/protocol['optimization']['epochs']))
        for group in optimizer.param_groups:group['lr']=trial['lr']*factor
        if trial['mode']=='agreement':order=torch.randperm(len(y),device='cuda')
        elif trial['blocks']:order=known_ids[torch.randperm(len(known_ids),device='cuda')]
        else:order=torch.zeros(observed,device='cuda',dtype=torch.long)
        totals=[];diagnostic=None
        for offset in range(0,len(order),128):
            ri=order[offset:offset+128]
            pi=ri if trial['mode']=='agreement' else sample_pathology_indices(sampler,128,generator=sampling_rng).to('cuda')
            rv=model(x[ri]) if trial['blocks'] else None
            pv=rv if trial['mode']=='agreement' else model(x[pi])
            path_loss=(b.masked_pathology_loss(pv[3],target[pi],mask[pi],pw) if trial['path_loss']=='legacy_bce'
                       else asymmetric_pathology_loss(pv[3],target[pi],mask[pi],gamma_negative=4 if trial['path_loss']=='asl' else 0,
                                                     clip=.05 if trial['path_loss']=='asl' else 0,label_weights=label_weights))
            rhythm_loss=path_loss*0
            if trial['blocks']:
                known=y[ri]>=0;ids=ri[known]
                if known.any():
                    ce=F.cross_entropy(rv[0][known],y[ids],reduction='none')
                    rhythm_loss=(ce*weights[ids]).mean()
                    original=ids<n
                    if original.any():
                        oid=ids[original]
                        rhythm_loss=rhythm_loss+.15*agreement_anchor_loss(rv[0][known][original],teacher[oid],y[oid])
                streams=[(rv[0],start_prob[ri],y[ri])]
                if trial['mode']=='balanced':streams.append((pv[0],start_prob[pi],y[pi]))
                rhythm_loss=rhythm_loss+starting_anchor(streams)
                values=[(rv,ri)] if trial['mode']=='agreement' else [(rv,ri),(pv,pi)]
                for value,index in values:
                    rhythm_loss=rhythm_loss+sum((((v-t[index])/s)**2).mean() for v,t,s in zip(value[1:3],auxiliary,scales))/len(values)
                if offset==0 and shared and trial['mode']=='balanced':
                    gr=torch.autograd.grad(rhythm_loss,shared,retain_graph=True,allow_unused=True)
                    gp=torch.autograd.grad(path_loss,shared,retain_graph=True,allow_unused=True)
                    dot=sum((a*z).sum() for a,z in zip(gr,gp) if a is not None and z is not None)
                    rn=sum(a.square().sum() for a in gr if a is not None).sqrt()
                    pn=sum(a.square().sum() for a in gp if a is not None).sqrt()
                    diagnostic={'cosine':float((dot/(rn*pn).clamp_min(1e-12)).detach()),'rhythm_norm':float(rn.detach()),'pathology_norm':float(pn.detach())}
            loss=rhythm_loss+path_loss*trial['path_weight']
            if not torch.isfinite(loss):raise ValueError('Non-finite loss')
            optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(params,1);optimizer.step()
            totals.append([float(loss.detach()),float(rhythm_loss.detach()),float(path_loss.detach())])
        b.assert_frozen(model,initial,trial['blocks'],True)
        history.append({'epoch':epoch,'mean_loss':float(np.mean(totals,0)[0]),'rhythm_loss':float(np.mean(totals,0)[1]),
                        'pathology_loss':float(np.mean(totals,0)[2]),'lr':trial['lr']*factor,'training_gradient':diagnostic})
        if epoch not in protocol['optimization']['checkpoints'] and epoch!=epochs:continue
        values=b.outputs(model,validation['x']);name=trial['name']+f'_e{epoch:02d}'
        record={'name':name,'trial':trial,'epoch':epoch,**b.validation_report(validation,values,val_ref,paths['validation_targets'],paths['validation_mask'],base)}
        records.append(record);state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        print(json.dumps({'candidate':name,'rhythm_accuracy':record['metrics']['accuracy'],'rhythm_macro_f1':record['metrics']['macro_f1'],
                         'pathology_macro_f1':record['pathology']['primary5_macro_f1'],'eligible':record['eligible']}),flush=True)
        if out:
            torch.save({'state_dict':state,'classes':c.CLASSES,'pathology_classes':b.PATHOLOGY_CLASSES,'parameters':c.PARAMETERS},out/(name+'.pth'))
            np.savez_compressed(out/(name+'_validation.npz'),rhythm_probabilities=b.rhythm_prob(values),pathology_probabilities=b.path_prob(values),
                                y=validation['y'],dataset=validation['dataset'],ecg_id=validation['ecg_id'],
                                path_targets=paths['validation_targets'],path_mask=paths['validation_mask'])
    return records,history,state


def main():
    c.require_cedia_gpu();os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8';torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.set_num_threads(2)
    for name,h in json.loads((c.ROOT/'INPUT_SHA256.json').read_text()).items():
        if c.digest(c.ROOT/name)!=h:raise ValueError('Input hash changed: '+name)
    if c.digest(c.ROOT/'start.pth')!=b.PROTOCOL['start_sha256']:raise ValueError('Warm start differs')
    train=dict(np.load(c.ROOT/'train.npz',allow_pickle=False));val=dict(np.load(c.ROOT/'validation.npz',allow_pickle=False))
    extra=dict(np.load(c.ROOT/'path_extra.npz',allow_pickle=False))
    with np.load(c.ROOT/'path_targets.npz',allow_pickle=False) as f:paths={k:f[k] for k in ('train_targets','train_mask','validation_targets','validation_mask')}
    for data,count in [(train,8393),(val,1476),(extra,3083)]:
        if data['x'].shape!=(count,6,1250) or data['x'].dtype!=np.float32 or not np.isfinite(data['x']).all():raise ValueError('Input contract differs')
        import hashlib
        for x,h in zip(data['x'],data['tensor_sha256']):
            if hashlib.sha256(x.astype('<f4',copy=False).tobytes()).hexdigest()!=str(h):raise ValueError('Tensor hash differs')
    model=c.load_model(c.ROOT/'start.pth').cuda();initial={k:v.cpu().clone() for k,v in model.state_dict().items()}
    tr=b.outputs(model,train['x']);er=b.outputs(model,extra['x']);vr=b.outputs(model,val['x'])
    prob=b.rhythm_prob(vr);sources=c.source_metrics(val,prob)
    observed=paths['validation_mask'][:,0]==1;thresholds=b.calibrate_thresholds(paths['validation_targets'][observed],b.path_prob(vr)[observed],role='validation')
    pathology=b.pathology_metrics(paths['validation_targets'][observed],b.path_prob(vr)[observed],thresholds)
    base={'metrics':c.metrics(val['y'],prob),'sources':sources,'rhythm_score':float(np.mean([m['macro_f1'] for m in sources.values()])),
          'pathology':pathology,'thresholds':thresholds.tolist()};base['joint_score']=base['rhythm_score']+.5*pathology['primary5_macro_f1']
    out=c.ROOT/'results';out.mkdir(exist_ok=False)
    c.save(out/'execution.json',{'job_id':os.environ['SLURM_JOB_ID'],'hostname':socket.gethostname(),'python':platform.python_version(),'platform':platform.platform(),
           'torch':torch.__version__,'numpy':np.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),'parameters':c.PARAMETERS,
           'input_manifest_sha256':c.digest(c.ROOT/'INPUT_SHA256.json'),'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})
    np.savez_compressed(out/'baseline_validation.npz',rhythm_probabilities=prob,pathology_probabilities=b.path_prob(vr),y=val['y'],dataset=val['dataset'],ecg_id=val['ecg_id'],
                        path_targets=paths['validation_targets'],path_mask=paths['validation_mask'])
    print(json.dumps({'BASELINE':base}),flush=True);reports=[];histories={}
    for trial in b.PROTOCOL['optimization']['trials']:
        records,history,_=run_trial(trial,40,train,extra,val,paths,initial,tr,er,vr,base,out)
        reports.extend(records);histories[trial['name']]=history
    selected=select_candidates(reports,base,role='validation');c.save(out/'training_histories.json',histories)
    refits={}
    for role,record in selected.items():
        if record is None:state=initial;refits[role]={'needed':False}
        else:
            _,_,state=run_trial(record['trial'],record['epoch'],train,extra,val,paths,initial,tr,er,vr,base,None)
            original=torch.load(out/(record['name']+'.pth'),map_location='cpu',weights_only=True)['state_dict']
            if not all(torch.equal(original[k],state[k]) for k in original):raise ValueError('Deterministic refit differs')
            refits[role]={'needed':True,'candidate':record['name'],'all_state_tensors_bitwise_equal':True}
        torch.save({'state_dict':state,'classes':c.CLASSES,'pathology_classes':b.PATHOLOGY_CLASSES,'parameters':c.PARAMETERS},out/(role+'.pth'))
    selection={'baseline':base,'candidates':reports,**selected,'selection_data':'validation only','thresholds_fitted_on_validation_only':True,
               'evaluation_labels_loaded_after_selection':True,'references_changed':False,'checkpoints':{role:c.digest(out/(role+'.pth')) for role in selected}}
    c.save(out/'selection.json',selection);c.save(out/'refit.json',refits)
    # No follow-up targets are opened until checkpoint selection is sealed above.
    data=dict(np.load(c.ROOT/'evaluation.npz',allow_pickle=False))
    with np.load(c.ROOT/'path_targets.npz',allow_pickle=False) as f:ey=f['followup_targets'];em=f['followup_mask']
    for role,record in [('baseline',None),*selected.items()]:
        model=c.load_model(c.ROOT/'start.pth' if role=='baseline' else out/(role+'.pth')).cuda()
        b.export_predictions(out,role,model,data,ey,em,np.asarray((record or base)['thresholds']))
    c.save(out/'SHA256SUMS.json',{p.name:c.digest(p) for p in sorted(out.iterdir()) if p.is_file()})
    print(json.dumps({'COMPLETED':True,**{role:r['name'] if r else 'start' for role,r in selected.items()}}),flush=True)


if __name__=='__main__':main()
