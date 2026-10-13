"""Continue the two original heads on frozen original-network features, on CEDIA."""
import json
import os
import platform
import random
import socket
import time

import numpy as np
import torch
from torch.nn import functional as F

import contract as c
import baseline_trainer as b
from balanced_metrics import rhythm_training_weights, agreement_anchor_loss, select_candidates
from head_metrics import configure_heads, assert_heads_frozen

b.PROTOCOL=json.loads((c.ROOT/'protocol.json').read_text())


def features(model,x):
    model.eval();parts=[]
    with torch.no_grad():
        for i in range(0,len(x),128):parts.append(model.forward_features(torch.from_numpy(x[i:i+128]).cuda()))
    value=torch.cat(parts)
    if value.shape!=(len(x),192) or value.requires_grad:raise ValueError('Frozen feature contract differs')
    return value


def head_values(model,x):
    return tuple(getattr(model,name)(x) for name in ('head_rhythm','head_quality','head_biometrics','head_pathology'))


def predict(model,x):
    model.eval();parts=[[],[],[],[]]
    with torch.inference_mode():
        for i in range(0,len(x),128):
            for bucket,value in zip(parts,head_values(model,x[i:i+128])):bucket.append(value.cpu().numpy())
    return tuple(np.concatenate(p) for p in parts)


def run_trial(trial,epochs,train,extra,val,paths,cache,reference,base,initial,out):
    seed=b.PROTOCOL['optimization']['seed'];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    model=c.load_model(c.ROOT/'start.pth').cuda();rhythm=trial['rhythm_head_train'];params=configure_heads(model,rhythm=rhythm)
    if sum(p.numel() for p in params)!=(38899 if rhythm else 19498):raise ValueError('Head parameter count changed')
    groups=[{'params':list(model.head_pathology.parameters()),'lr':trial['path_lr']}]
    if rhythm:groups.append({'params':list(model.head_rhythm.parameters()),'lr':trial['rhythm_lr']})
    optimizer=torch.optim.AdamW(groups,weight_decay=.01)
    yy=np.concatenate((train['y'],extra['y']));y=torch.from_numpy(yy).cuda();n=len(train['y'])
    target=torch.from_numpy(np.concatenate((paths['train_targets'],extra['targets']))).cuda()
    mask=torch.from_numpy(np.concatenate((paths['train_mask'],extra['mask']))).cuda()
    positive=(target*mask).sum(0);negative=((1-target)*mask).sum(0)
    if (positive==0).any() or (negative==0).any():raise ValueError('Training support missing')
    pw=(negative/positive).sqrt().clamp(1,10)
    dataset=np.concatenate((train['dataset'],np.full(len(extra['x']),'ptb-xl')))
    weights=torch.from_numpy(rhythm_training_weights(yy,dataset,role='train')).cuda()
    focus=torch.ones_like(weights);focus[y==c.CLASSES.index('SB')]=trial['sb_focus']
    for label,multiplier in b.PROTOCOL['optimization']['other_rhythm_focus'].items():focus[y==c.CLASSES.index(label)]=multiplier
    weights=weights*focus;weights/=weights[y>=0].mean()
    start_prob=torch.from_numpy(b.rhythm_prob(reference['train'])).cuda();teacher=torch.from_numpy(train['teacher']).cuda()
    history=[];records=[];state=None
    for epoch in range(1,epochs+1):
        configure_heads(model,rhythm=rhythm);order=torch.randperm(len(y),device='cuda');totals=[]
        for i in range(0,len(y),128):
            ids=order[i:i+128];value=head_values(model,cache['train'][ids])
            loss=b.masked_pathology_loss(value[3],target[ids],mask[ids],pw)
            if rhythm:
                known=y[ids]>=0;ki=ids[known]
                if known.any():
                    loss=loss+(F.cross_entropy(value[0][known],y[ki],reduction='none')*weights[ki]).mean()
                    loss=loss+.1*agreement_anchor_loss(value[0],start_prob[ids],y[ids])
                    original=ki<n
                    if original.any():loss=loss+.15*agreement_anchor_loss(value[0][known][original],teacher[ki[original]],y[ki[original]])
            if not torch.isfinite(loss):raise ValueError('Non-finite loss')
            optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(params,1);optimizer.step()
            totals.append(float(loss.detach()))
        assert_heads_frozen(model,initial,rhythm=rhythm);history.append({'epoch':epoch,'mean_loss':float(np.mean(totals))})
        if epoch not in b.PROTOCOL['optimization']['checkpoints'] and epoch!=epochs:continue
        values=predict(model,cache['validation']);name=trial['name']+f'_e{epoch:02d}'
        record={'name':name,'trial':trial,'epoch':epoch,**b.validation_report(val,values,reference['validation'],paths['validation_targets'],paths['validation_mask'],base)}
        records.append(record);state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        print(json.dumps({'candidate':name,'rhythm_accuracy':record['metrics']['accuracy'],'rhythm_macro_f1':record['metrics']['macro_f1'],
                         'pathology_macro_f1':record['pathology']['primary5_macro_f1'],'eligible':record['eligible']}),flush=True)
        if out:
            torch.save({'state_dict':state,'classes':c.CLASSES,'pathology_classes':b.PATHOLOGY_CLASSES,'parameters':c.PARAMETERS},out/(name+'.pth'))
            np.savez_compressed(out/(name+'_validation.npz'),rhythm_probabilities=b.rhythm_prob(values),pathology_probabilities=b.path_prob(values),
                                y=val['y'],dataset=val['dataset'],ecg_id=val['ecg_id'],path_targets=paths['validation_targets'],path_mask=paths['validation_mask'])
    return records,history,state


def main():
    c.require_cedia_gpu();os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8';torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.set_num_threads(2)
    for name,h in json.loads((c.ROOT/'INPUT_SHA256.json').read_text()).items():
        if c.digest(c.ROOT/name)!=h:raise ValueError('Input hash differs: '+name)
    for name,key in [('start.pth','start_sha256'),('reference.pth','reference_sha256')]:
        if c.digest(c.ROOT/name)!=b.PROTOCOL[key]:raise ValueError('Initialization/reference differs')
    train=dict(np.load(c.ROOT/'train.npz',allow_pickle=False));val=dict(np.load(c.ROOT/'validation.npz',allow_pickle=False));extra=dict(np.load(c.ROOT/'path_extra.npz',allow_pickle=False))
    with np.load(c.ROOT/'path_targets.npz',allow_pickle=False) as f:paths={k:f[k] for k in ('train_targets','train_mask','validation_targets','validation_mask')}
    for data,count in [(train,8393),(val,1476),(extra,3083)]:
        if data['x'].shape!=(count,6,1250) or data['x'].dtype!=np.float32 or not np.isfinite(data['x']).all():raise ValueError('Input contract differs')
        import hashlib
        for x,h in zip(data['x'],data['tensor_sha256']):
            if hashlib.sha256(x.astype('<f4',copy=False).tobytes()).hexdigest()!=str(h):raise ValueError('Tensor hash differs')
    model=c.load_model(c.ROOT/'start.pth').cuda();initial={k:v.cpu().clone() for k,v in model.state_dict().items()}
    x=np.concatenate((train['x'],extra['x']));cache={'train':features(model,x),'validation':features(model,val['x'])}
    raw=b.outputs(model,val['x']);cached=predict(model,cache['validation'])
    if not all(np.array_equal(a,z) for a,z in zip(raw,cached)):raise ValueError('Cached-head outputs differ from full network')
    reference_model=c.load_model(c.ROOT/'reference.pth').cuda();fallback={k:v.cpu().clone() for k,v in reference_model.state_dict().items()}
    reference={'train':b.outputs(reference_model,x),'validation':b.outputs(reference_model,val['x'])}
    prob=b.rhythm_prob(reference['validation']);sources=c.source_metrics(val,prob);observed=paths['validation_mask'][:,0]==1
    thresholds=b.calibrate_thresholds(paths['validation_targets'][observed],b.path_prob(reference['validation'])[observed],role='validation')
    path=b.pathology_metrics(paths['validation_targets'][observed],b.path_prob(reference['validation'])[observed],thresholds)
    base={'metrics':c.metrics(val['y'],prob),'sources':sources,'rhythm_score':float(np.mean([v['macro_f1'] for v in sources.values()])),
          'pathology':path,'thresholds':thresholds.tolist()};base['joint_score']=base['rhythm_score']+.5*path['primary5_macro_f1']
    out=c.ROOT/'results';out.mkdir(exist_ok=False)
    c.save(out/'execution.json',{'job_id':os.environ['SLURM_JOB_ID'],'hostname':socket.gethostname(),'python':platform.python_version(),'platform':platform.platform(),
           'torch':torch.__version__,'numpy':np.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),'parameters':c.PARAMETERS,
           'input_manifest_sha256':c.digest(c.ROOT/'INPUT_SHA256.json'),'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
           'cached_validation_outputs_bitwise_equal_to_full_network':True,'cached_features':192})
    np.savez_compressed(out/'baseline_validation.npz',rhythm_probabilities=prob,pathology_probabilities=b.path_prob(reference['validation']),y=val['y'],dataset=val['dataset'],ecg_id=val['ecg_id'],
                        path_targets=paths['validation_targets'],path_mask=paths['validation_mask'])
    print(json.dumps({'BASELINE':base}),flush=True);records=[];histories={}
    for trial in b.PROTOCOL['optimization']['trials']:
        rr,hh,_=run_trial(trial,32,train,extra,val,paths,cache,reference,base,initial,out);records.extend(rr);histories[trial['name']]=hh
    selected=select_candidates(records,base,role='validation');c.save(out/'training_histories.json',histories);refits={}
    for role,record in selected.items():
        if record is None:state=fallback;refits[role]={'needed':False,'fallback':'phase6 reference.pth'}
        else:
            _,_,state=run_trial(record['trial'],record['epoch'],train,extra,val,paths,cache,reference,base,initial,None)
            saved=torch.load(out/(record['name']+'.pth'),map_location='cpu',weights_only=True)['state_dict']
            if not all(torch.equal(saved[k],state[k]) for k in saved):raise ValueError('Refit differs')
            refits[role]={'needed':True,'candidate':record['name'],'all_state_tensors_bitwise_equal':True}
        torch.save({'state_dict':state,'classes':c.CLASSES,'pathology_classes':b.PATHOLOGY_CLASSES,'parameters':c.PARAMETERS},out/(role+'.pth'))
    selection={'baseline':base,'candidates':records,**selected,'selection_data':'validation only','thresholds_fitted_on_validation_only':True,
               'evaluation_labels_loaded_after_selection':True,'references_changed':False,'checkpoints':{role:c.digest(out/(role+'.pth')) for role in selected}}
    c.save(out/'selection.json',selection);c.save(out/'refit.json',refits)
    data=dict(np.load(c.ROOT/'evaluation.npz',allow_pickle=False))
    with np.load(c.ROOT/'path_targets.npz',allow_pickle=False) as f:ey=f['followup_targets'];em=f['followup_mask']
    for role,record in [('baseline',None),*selected.items()]:
        model=c.load_model(c.ROOT/'reference.pth' if role=='baseline' else out/(role+'.pth')).cuda()
        b.export_predictions(out,role,model,data,ey,em,np.asarray((record or base)['thresholds']))
    c.save(out/'SHA256SUMS.json',{p.name:c.digest(p) for p in sorted(out.iterdir()) if p.is_file()})
    print(json.dumps({'COMPLETED':True,**{role:r['name'] if r else 'phase6_reference' for role,r in selected.items()}}),flush=True)


if __name__=='__main__':main()
