"""Fine-tune the existing CNN with observed pathology targets, never test targets."""
import csv
import json
import os
import platform
from pathlib import Path
import random
import socket
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import contract as c
from joint_metrics import PATHOLOGY_CLASSES, PRIMARY5_INDEX, masked_pathology_loss, calibrate_thresholds, pathology_metrics

PROTOCOL=json.loads((Path(__file__).resolve().parent/'protocol.json').read_text())


def prefixes(blocks,path_train):
    if blocks not in (0,2,5):raise ValueError('Invalid existing-block configuration')
    backbone=() if blocks==0 else ('stem.','res_layers.','head_rhythm.') if blocks==5 else ('res_layers.3.','res_layers.4.','head_rhythm.')
    return backbone+(('head_pathology.',) if path_train else ())


def configure(model,blocks,path_train):
    model.eval()
    for name,p in model.named_parameters():p.requires_grad_(name.startswith(prefixes(blocks,path_train)))
    if blocks:
        if blocks==5:
            model.stem.train()
            for module in model.stem.modules():
                if isinstance(module,nn.BatchNorm1d):module.eval()
        for index in (range(5) if blocks==5 else (3,4)):
            model.res_layers[index].train()
            for module in model.res_layers[index].modules():
                if isinstance(module,nn.BatchNorm1d):module.eval()
        model.head_rhythm.train()
    if path_train:model.head_pathology.train()
    return [p for p in model.parameters() if p.requires_grad]


def assert_frozen(model,initial,blocks,path_train):
    allowed={n for n,_ in model.named_parameters() if n.startswith(prefixes(blocks,path_train))}
    current=model.state_dict()
    if set(current)!=set(initial):raise ValueError('State keys changed')
    changed=[]
    for name,value in current.items():
        if value.shape!=initial[name].shape:raise ValueError('State shape changed')
        if not torch.equal(value.detach().cpu(),initial[name]):
            if name not in allowed:raise ValueError('Frozen parameter/buffer changed: '+name)
            changed.append(name)
    return changed


def outputs(model,x):
    model.eval();parts=[[],[],[],[]]
    with torch.inference_mode():
        for start in range(0,len(x),128):
            value=model(torch.from_numpy(x[start:start+128]).cuda())
            if len(value)!=4:raise ValueError('Four-head contract changed')
            for bucket,tensor in zip(parts,value):bucket.append(tensor.cpu().numpy())
    return tuple(np.concatenate(p) for p in parts)


def rhythm_prob(values):return torch.softmax(torch.from_numpy(values[0]),1).numpy()
def path_prob(values):return torch.sigmoid(torch.from_numpy(values[3])).numpy()


def protected_recalls(sources,baseline):
    policy=PROTOCOL['selection']
    return all(m['recall'][label]>=baseline[source]['recall'][label]-policy['maximum_recall_drop']-1e-12
               for source,m in sources.items() for label in policy['protected_classes']
               if m['support'].get(label,0)>=policy['minimum_class_support'])


def quality_bio_errors(values,reference):
    return {name:float(np.sqrt(np.mean(((v-t)/np.maximum(t.std(0),.05))**2)))
            for name,v,t in zip(('quality','biometrics'),values[1:3],reference[1:3])}


def validation_report(data,values,baseline,targets,mask,base_report):
    p=rhythm_prob(values);sources=c.source_metrics(data,p);score=float(np.mean([m['macro_f1'] for m in sources.values()]))
    metrics=c.metrics(data['y'],p);errors=quality_bio_errors(values,baseline)
    idx=mask[:,0]==1;pp=path_prob(values)[idx];y=targets[idx]
    thresholds=calibrate_thresholds(y,pp,role='validation');path=pathology_metrics(y,pp,thresholds)
    policy=PROTOCOL['selection'];checks={'protected_recalls':protected_recalls(sources,base_report['sources']),
        'rhythm_macro_preserved':metrics['macro_f1']>=base_report['metrics']['macro_f1']-policy['maximum_rhythm_macro_f1_drop']-1e-12,
        'rhythm_sources_preserved':score>=base_report['rhythm_score']-policy['maximum_source_mean_f1_drop']-1e-12,
        'quality_biometrics_preserved':max(errors.values())<=policy['maximum_quality_biometrics_nrmse']}
    return {'metrics':metrics,'sources':sources,'rhythm_score':score,'pathology':path,'thresholds':thresholds.tolist(),
            'joint_score':score+.5*path['primary5_macro_f1'],'quality_biometrics_nrmse':errors,
            'preservation_checks':checks,'eligible':all(checks.values())}


def run_trial(trial,epochs,train,extra,validation,path_data,initial,train_ref,extra_ref,val_ref,base_report,out):
    seed=PROTOCOL['optimization']['seed'];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    model=c.load_model(c.ROOT/'start.pth').cuda();params=configure(model,trial['blocks'],trial['path_train'])
    expected=(1099289 if trial['blocks']==5 else 759445 if trial['blocks'] else 0)+(19498 if trial['path_train'] else 0)
    if sum(p.numel() for p in params)!=expected:raise ValueError('Trainable parameter contract changed')
    opt=torch.optim.AdamW(params,lr=trial['lr'],weight_decay=.01)
    use_extra=trial['path_train'] or trial['expanded_rhythm'];n=len(train['y'])
    xx=np.concatenate((train['x'],extra['x'])) if use_extra else train['x']
    extra_y=extra['y'] if trial['expanded_rhythm'] else np.full(len(extra['x']),-1,dtype=np.int64)
    yy=np.concatenate((train['y'],extra_y)) if use_extra else train['y']
    py=np.concatenate((path_data['train_targets'],extra['targets'])) if use_extra else path_data['train_targets']
    pm=np.concatenate((path_data['train_mask'],extra['mask'])) if use_extra else path_data['train_mask']
    ref=tuple(np.concatenate((a,b)) if use_extra else a for a,b in zip(train_ref,extra_ref))
    x=torch.from_numpy(xx).cuda();y=torch.from_numpy(yy).cuda();target=torch.from_numpy(py).cuda();mask=torch.from_numpy(pm).cuda()
    positive=(target*mask).sum(0);negative=((1-target)*mask).sum(0)
    if (positive==0).any() or (negative==0).any():raise ValueError('Pathology class lacks training support')
    pw=(negative/positive).sqrt().clamp(1,10)
    start_prob=torch.from_numpy(rhythm_prob(ref)).cuda()
    teacher=torch.from_numpy(train['teacher']).cuda()
    datasets=np.concatenate((train['dataset'],np.full(len(extra['x']),'ptb-xl'))) if use_extra else train['dataset']
    weights=torch.from_numpy(c.sample_weights({'y':yy,'dataset':datasets})).cuda()
    focus=torch.ones_like(weights)
    for label in ('SB','LBBB'):focus[y==c.CLASSES.index(label)]=3
    for label in ('PAC','RBBB','NSR','PVC'):focus[y==c.CLASSES.index(label)]=2
    weights=weights*focus;weights=weights/weights[y>=0].mean()
    aux=tuple(torch.from_numpy(a).cuda() for a in ref[1:3]);scales=tuple(a.std(0,unbiased=False).clamp_min(.05) for a in aux)
    history=[];records=[];last_state=None
    for epoch in range(1,epochs+1):
        configure(model,trial['blocks'],trial['path_train']);order=torch.randperm(len(y),device='cuda');total=0
        for start in range(0,len(y),128):
            index=order[start:start+128];value=model(x[index]);logp=F.log_softmax(value[0],1)
            loss=value[0].sum()*0
            known=y[index]>=0
            if trial['blocks'] and known.any():
                ids=index[known];ce=F.cross_entropy(value[0][known],y[ids],reduction='none')
                kl=F.kl_div(logp[known],start_prob[ids],reduction='none').sum(1)
                tkl=torch.zeros_like(ce);original=ids<n
                if original.any():
                    original_ids=ids[original]
                    divergence=F.kl_div(logp[known][original],teacher[original_ids],reduction='none').sum(1)
                    tkl[original]=divergence*(teacher[original_ids].argmax(1)==y[original_ids])
                loss=loss+((ce+.2*kl+.15*tkl)*weights[ids]).mean()
            if trial['blocks']:
                loss=loss+sum((((v-t[index])/s)**2).mean() for v,t,s in zip(value[1:3],aux,scales))
                if (~known).any():
                    loss=loss+.2*F.kl_div(logp[~known],start_prob[index[~known]],reduction='batchmean')
            if trial['path_train']:
                loss=loss+trial['path_weight']*masked_pathology_loss(value[3],target[index],mask[index],pw)
            if not torch.isfinite(loss):raise ValueError('Non-finite loss')
            opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(params,1);opt.step();total+=loss.detach().item()*len(index)
        assert_frozen(model,initial,trial['blocks'],trial['path_train']);history.append({'epoch':epoch,'mean_loss':total/len(y)})
        if epoch not in PROTOCOL['optimization']['checkpoints'] and epoch!=epochs:continue
        values=outputs(model,validation['x']);name=trial['name']+f'_e{epoch:02d}'
        record={'name':name,'trial':trial,'epoch':epoch,**validation_report(validation,values,val_ref,path_data['validation_targets'],path_data['validation_mask'],base_report)}
        records.append(record);last_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        print(json.dumps({'candidate':name,'rhythm_accuracy':record['metrics']['accuracy'],'rhythm_macro_f1':record['metrics']['macro_f1'],
                         'pathology_macro_f1':record['pathology']['primary5_macro_f1'],'eligible':record['eligible'],'checks':record['preservation_checks']}),flush=True)
        if out:
            torch.save({'state_dict':last_state,'classes':c.CLASSES,'pathology_classes':PATHOLOGY_CLASSES,'parameters':c.PARAMETERS},out/(name+'.pth'))
            np.savez_compressed(out/(name+'_validation.npz'),rhythm_probabilities=rhythm_prob(values),pathology_probabilities=path_prob(values),
                                y=validation['y'],dataset=validation['dataset'],ecg_id=validation['ecg_id'],
                                path_targets=path_data['validation_targets'],path_mask=path_data['validation_mask'])
    return records,history,last_state


def export_predictions(out,name,model,data,targets,mask,thresholds):
    values=outputs(model,data['x']);repeat=outputs(model,data['x'])
    if not all(np.array_equal(a,b) for a,b in zip(values,repeat)):raise ValueError('Repeated inference differs')
    p=rhythm_prob(values);pp=path_prob(values);rows=[]
    for i in range(len(p)):
        row={'cohort':str(data['cohort'][i]),'dataset':str(data['dataset'][i]),'ecg_id':str(data['ecg_id'][i]),
             'reference':c.CLASSES[int(data['y'][i])],'prediction':c.CLASSES[int(p[i].argmax())],
             'tensor_sha256':str(data['tensor_sha256'][i]),'pathology_observed':int(mask[i,0])}
        row.update({f'prob_{label}':float(p[i,j]) for j,label in enumerate(c.CLASSES)})
        for j,label in enumerate(PATHOLOGY_CLASSES):
            row['target_'+label]=int(targets[i,j]) if mask[i,j] else ''
            row['score_'+label]=float(pp[i,j]);row['pred_'+label]=int(pp[i,j]>=thresholds[j])
        rows.append(row)
    with (out/(name+'_predictions.csv')).open('x',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    cohort_metrics={str(cohort):c.metrics(data['y'][idx],p[idx]) for cohort in np.unique(data['cohort']) if (idx:=data['cohort']==cohort).any()}
    observed=mask[:,0]==1
    path=pathology_metrics(targets[observed],pp[observed],thresholds)
    path_cohorts={str(cohort):pathology_metrics(targets[idx],pp[idx],thresholds)
                 for cohort in np.unique(data['cohort'][observed]) if (idx:=((data['cohort']==cohort)&observed)).any()}
    c.save(out/(name+'_metrics.json'),{'rhythm':c.metrics(data['y'],p),'cohorts':cohort_metrics,'pathology':path,
                                    'pathology_cohorts':path_cohorts,'repeat_outputs_bitwise_equal':True})


def main():
    c.require_cedia_gpu();os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8';torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.set_num_threads(2)
    for name,h in json.loads((c.ROOT/'INPUT_SHA256.json').read_text()).items():
        if c.digest(c.ROOT/name)!=h:raise ValueError('Input hash changed: '+name)
    if c.digest(c.ROOT/'start.pth')!=PROTOCOL['start_sha256']:raise ValueError('Warm start changed')
    train=dict(np.load(c.ROOT/'train.npz',allow_pickle=False));validation=dict(np.load(c.ROOT/'validation.npz',allow_pickle=False))
    extra=dict(np.load(c.ROOT/'path_extra.npz',allow_pickle=False))
    # Followup targets remain on disk, unopened until checkpoint and thresholds are selected.
    with np.load(c.ROOT/'path_targets.npz',allow_pickle=False) as f:
        paths={k:f[k] for k in ('train_targets','train_mask','validation_targets','validation_mask')}
    for data,count in [(train,8393),(validation,1476),(extra,PROTOCOL['data']['pathology_extra_train'])]:
        if data['x'].shape!=(count,6,1250) or data['x'].dtype!=np.float32 or not np.isfinite(data['x']).all():raise ValueError('Input contract changed')
        import hashlib
        for x,h in zip(data['x'],data['tensor_sha256']):
            if hashlib.sha256(x.astype('<f4',copy=False).tobytes()).hexdigest()!=str(h):raise ValueError('Tensor hash changed')
    model=c.load_model(c.ROOT/'start.pth').cuda();initial={k:v.cpu().clone() for k,v in model.state_dict().items()}
    train_ref=outputs(model,train['x']);extra_ref=outputs(model,extra['x']);val_ref=outputs(model,validation['x'])
    bp=rhythm_prob(val_ref);bs=c.source_metrics(validation,bp);bm=c.metrics(validation['y'],bp)
    idx=paths['validation_mask'][:,0]==1;thresholds=calibrate_thresholds(paths['validation_targets'][idx],path_prob(val_ref)[idx],role='validation')
    bpath=pathology_metrics(paths['validation_targets'][idx],path_prob(val_ref)[idx],thresholds)
    base={'metrics':bm,'sources':bs,'rhythm_score':float(np.mean([m['macro_f1'] for m in bs.values()])),'pathology':bpath,'thresholds':thresholds.tolist()}
    base['joint_score']=base['rhythm_score']+.5*bpath['primary5_macro_f1']
    out=c.ROOT/'results';out.mkdir(exist_ok=False)
    c.save(out/'execution.json',{'job_id':os.environ['SLURM_JOB_ID'],'hostname':socket.gethostname(),'python':platform.python_version(),'platform':platform.platform(),'torch':torch.__version__,
           'numpy':np.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),'parameters':c.PARAMETERS,
           'input_manifest_sha256':c.digest(c.ROOT/'INPUT_SHA256.json'),'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})
    np.savez_compressed(out/'baseline_validation.npz',rhythm_probabilities=bp,pathology_probabilities=path_prob(val_ref),
                        y=validation['y'],dataset=validation['dataset'],ecg_id=validation['ecg_id'],
                        path_targets=paths['validation_targets'],path_mask=paths['validation_mask'])
    print(json.dumps({'BASELINE':base}),flush=True)
    reports=[];histories={}
    for trial in PROTOCOL['optimization']['trials']:
        records,history,_=run_trial(trial,32,train,extra,validation,paths,initial,train_ref,extra_ref,val_ref,base,out)
        reports.extend(records);histories[trial['name']]=history
    eligible=[r for r in reports if r['eligible']]
    joint=[r for r in eligible if r['trial']['path_train'] and r['pathology']['primary5_macro_f1']>=bpath['primary5_macro_f1']+.02]
    selected=max(joint,key=lambda r:(r['joint_score'],r['name'])) if joint else None
    rhythm=[r for r in eligible if r['rhythm_score']>base['rhythm_score']+.002]
    best_rhythm=max(rhythm,key=lambda r:(r['rhythm_score'],r['name'])) if rhythm else None
    c.save(out/'training_histories.json',histories)
    refits={};selected_states={}
    for name,record in [('joint_selected',selected),('rhythm_selected',best_rhythm)]:
        if record is None:state=initial;refits[name]={'needed':False}
        else:
            _,_,state=run_trial(record['trial'],record['epoch'],train,extra,validation,paths,initial,train_ref,extra_ref,val_ref,base,None)
            saved=torch.load(out/(record['name']+'.pth'),map_location='cpu',weights_only=True)['state_dict']
            if not all(torch.equal(saved[k],state[k]) for k in saved):raise ValueError('Deterministic refit differs')
            refits[name]={'needed':True,'candidate':record['name'],'all_state_tensors_bitwise_equal':True}
        torch.save({'state_dict':state,'classes':c.CLASSES,'pathology_classes':PATHOLOGY_CLASSES,'parameters':c.PARAMETERS},out/(name+'.pth'))
        selected_states[name]=state
    selection={'baseline':base,'candidates':reports,'joint_selected':selected,'rhythm_selected':best_rhythm,
               'selection_data':'validation only','thresholds_fitted_on_validation_only':True,'evaluation_labels_loaded_after_selection':True,
               'references_changed':False,'checkpoints':{name:c.digest(out/(name+'.pth')) for name in selected_states}}
    c.save(out/'selection.json',selection);c.save(out/'refit.json',refits)
    data=dict(np.load(c.ROOT/'evaluation.npz',allow_pickle=False))
    with np.load(c.ROOT/'path_targets.npz',allow_pickle=False) as f:ey=f['followup_targets'];em=f['followup_mask']
    for name,record in [('baseline',None),('joint_selected',selected),('rhythm_selected',best_rhythm)]:
        if name=='baseline':model=c.load_model(c.ROOT/'start.pth').cuda();threshold=base['thresholds']
        else:model=c.load_model(out/(name+'.pth')).cuda();threshold=record['thresholds'] if record else base['thresholds']
        export_predictions(out,name,model,data,ey,em,np.asarray(threshold))
    c.save(out/'SHA256SUMS.json',{p.name:c.digest(p) for p in sorted(out.iterdir()) if p.is_file()})
    print(json.dumps({'COMPLETED':True,'joint_selected':selected['name'] if selected else 'start','rhythm_selected':best_rhythm['name'] if best_rhythm else 'start'}),flush=True)


if __name__=='__main__':main()
