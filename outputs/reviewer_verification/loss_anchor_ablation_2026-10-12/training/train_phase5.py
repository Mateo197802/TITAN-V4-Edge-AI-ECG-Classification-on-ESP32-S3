"""Compare training-only agreement anchoring on the existing CEDIA CNN."""
import csv
import json
import os
from pathlib import Path
import random
import socket
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import contract as c

PROTOCOL=json.loads((Path(__file__).resolve().parent/'protocol.json').read_text())


def prefixes(blocks):
    if blocks not in (1,2):raise ValueError('Invalid partial-backbone configuration')
    return ('res_layers.4.','head_rhythm.') if blocks==1 else ('res_layers.3.','res_layers.4.','head_rhythm.')


def configure(model,blocks):
    model.eval()
    for name,p in model.named_parameters():p.requires_grad_(name.startswith(prefixes(blocks)))
    for index in range(5-blocks,5):
        model.res_layers[index].train()
        for module in model.res_layers[index].modules():
            if isinstance(module,nn.BatchNorm1d):module.eval()
    model.head_rhythm.train()
    return [p for p in model.parameters() if p.requires_grad]


def agreement_weights(y,gold,teacher,gold_weight,teacher_weight,gold_gate,teacher_gate):
    g=torch.full_like(y,float(gold_weight),dtype=gold.dtype)
    t=torch.full_like(y,float(teacher_weight),dtype=teacher.dtype)
    if gold_gate:g=g*(gold.argmax(1)==y)
    if teacher_gate:t=t*(teacher.argmax(1)==y)
    return g,t


def assert_frozen(model,initial,blocks):
    allowed={n for n,_ in model.named_parameters() if n.startswith(prefixes(blocks))}
    current=model.state_dict()
    if set(current)!=set(initial):raise ValueError('State keys changed')
    changed=[]
    for name,value in current.items():
        if not torch.equal(value.detach().cpu(),initial[name]):
            if name not in allowed:raise ValueError('Frozen parameter/buffer changed: '+name)
            changed.append(name)
    return changed


def outputs(model,data):
    model.eval();parts=[[],[],[],[]]
    with torch.inference_mode():
        for start in range(0,len(data['x']),128):
            value=model(torch.from_numpy(data['x'][start:start+128]).cuda())
            if len(value)!=4:raise ValueError('Four-head contract changed')
            for bucket,tensor in zip(parts,value,strict=True):bucket.append(tensor.cpu().numpy())
    return tuple(np.concatenate(p) for p in parts)


def probabilities(values):return torch.softmax(torch.from_numpy(values[0]),1).numpy()


def aux_errors(values,reference):
    return {name:float(np.sqrt(np.mean(((value-target)/np.maximum(target.std(0),.05))**2)))
            for name,value,target in zip(('quality','biometrics','pathology'),values[1:],reference[1:],strict=True)}


def validate_data(data,role,count):
    if data['x'].shape!=(count,6,1250) or data['x'].dtype!=np.float32 or not np.isfinite(data['x']).all():
        raise ValueError('Input contract differs: '+role)
    if data['y'].shape!=(count,) or not np.issubdtype(data['y'].dtype,np.integer) or np.any(data['y']<0) or np.any(data['y']>=9):
        raise ValueError('Target contract differs: '+role)
    for x,expected in zip(data['x'],data['tensor_sha256'],strict=True):
        import hashlib
        if hashlib.sha256(x.astype('<f4',copy=False).tobytes()).hexdigest()!=str(expected):
            raise ValueError('Frozen tensor changed: '+role)


def validation_report(data,p,base_p,values,base_values):
    score,gold_ok,sources=c.validation_score(data,p,data['gold'])
    baseline=c.source_metrics(data,base_p);policy=PROTOCOL['selection'];checks={}
    for source,m in sources.items():
        for label in policy['protected_classes']:
            if m['support'][label]>=policy['minimum_class_support']:
                checks[source+':'+label]=m['recall'][label]>=baseline[source]['recall'][label]-policy['maximum_recall_drop']-1e-12
    errors=aux_errors(values,base_values)
    return {'score':score,'eligible':bool(gold_ok and all(checks.values()) and max(errors.values())<=policy['maximum_auxiliary_nrmse']),
            'metrics':c.metrics(data['y'],p),'sources':sources,'auxiliary_nrmse':errors,'protected_recall_checks':checks,'gold_sb_lbbb_preserved':gold_ok}


def run_trial(trial,epochs,train,validation,initial,train_reference,val_reference,base_p,out):
    seed=PROTOCOL['optimization']['seed'];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    model=c.load_model(c.ROOT/'start.pth').cuda();params=configure(model,trial['blocks'])
    expected=460065 if trial['blocks']==1 else 759445
    if sum(p.numel() for p in params)!=expected:raise ValueError('Trainable parameter contract changed')
    opt=torch.optim.AdamW(params,lr=trial['lr'],weight_decay=PROTOCOL['optimization']['weight_decay'])
    x=torch.from_numpy(train['x']).cuda();y=torch.from_numpy(train['y']).cuda()
    gold=torch.from_numpy(train['gold']).cuda();teacher=torch.from_numpy(train['teacher']).cuda()
    gw,tw=agreement_weights(y,gold,teacher,trial['gold_weight'],.15,trial['gold_agreement'],trial['teacher_agreement'])
    weights=torch.from_numpy(c.sample_weights(train)).cuda();focus=torch.ones_like(weights)
    for label in ('SB','LBBB'):focus[y==c.CLASSES.index(label)]=trial['sb_lbbb_weight']
    for label in ('PAC','RBBB','NSR','PVC'):focus[y==c.CLASSES.index(label)]=trial['weak_weight']
    weights=weights*focus;weights=weights/weights.mean()
    targets=tuple(torch.from_numpy(a).cuda() for a in train_reference[1:]);scales=tuple(a.std(0,unbiased=False).clamp_min(.05) for a in targets)
    history=[];records=[];last_state=None
    for epoch in range(1,epochs+1):
        configure(model,trial['blocks']);order=torch.randperm(len(y),device='cuda');total=0
        for start in range(0,len(y),128):
            index=order[start:start+128];value=model(x[index]);logp=F.log_softmax(value[0],1)
            ce=F.cross_entropy(value[0],y[index],reduction='none')
            gkl=F.kl_div(logp,gold[index],reduction='none').sum(1);tkl=F.kl_div(logp,teacher[index],reduction='none').sum(1)
            aux=sum((((v-t[index])/s)**2).mean() for v,t,s in zip(value[1:],targets,scales,strict=True))
            loss=((ce+gw[index]*gkl+tw[index]*tkl)*weights[index]).mean()+PROTOCOL['optimization']['auxiliary_weight']*aux
            if not torch.isfinite(loss):raise ValueError('Non-finite training loss')
            opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(params,1.0);opt.step();total+=loss.item()*len(index)
        assert_frozen(model,initial,trial['blocks']);history.append({'epoch':epoch,'mean_loss':total/len(y)})
        if epoch not in PROTOCOL['optimization']['checkpoints'] and epoch!=epochs:continue
        value=outputs(model,validation);p=probabilities(value);name=trial['name']+f'_e{epoch:02d}'
        record={'name':name,'trial':trial,'epoch':epoch,**validation_report(validation,p,base_p,value,val_reference)};records.append(record)
        last_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        print(json.dumps({'candidate':name,'accuracy':record['metrics']['accuracy'],'macro_f1':record['metrics']['macro_f1'],'score':record['score'],'eligible':record['eligible'],'auxiliary_nrmse':record['auxiliary_nrmse']}),flush=True)
        if out is not None:
            torch.save({'state_dict':last_state,'classes':c.CLASSES,'parameters':c.PARAMETERS},out/(name+'.pth'))
            np.savez_compressed(out/(name+'_validation.npz'),probabilities=p,y=validation['y'],dataset=validation['dataset'],ecg_id=validation['ecg_id'])
    return records,history,last_state


def main():
    c.require_cedia_gpu();os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8';torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.set_num_threads(2)
    for name,h in json.loads((c.ROOT/'INPUT_SHA256.json').read_text()).items():
        if c.digest(c.ROOT/name)!=h:raise ValueError('Input hash changed: '+name)
    if c.digest(c.ROOT/'start.pth')!=PROTOCOL['start_sha256']:raise ValueError('Warm start changed')
    train=dict(np.load(c.ROOT/'train.npz',allow_pickle=False));validation=dict(np.load(c.ROOT/'validation.npz',allow_pickle=False))
    validate_data(train,'train',8393);validate_data(validation,'validation',1476)
    c.validate_probabilities(train['gold']);c.validate_probabilities(train['teacher']);c.validate_probabilities(validation['gold'])
    model=c.load_model(c.ROOT/'start.pth').cuda();initial={k:v.cpu().clone() for k,v in model.state_dict().items()}
    train_ref=outputs(model,train);val_ref=outputs(model,validation);base_p=probabilities(val_ref)
    score,_,_=c.validation_score(validation,base_p,validation['gold']);out=c.ROOT/'results';out.mkdir(exist_ok=False)
    c.save(out/'execution.json',{'job_id':os.environ['SLURM_JOB_ID'],'hostname':socket.gethostname(),'torch':torch.__version__,'numpy':np.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'parameters':c.PARAMETERS,'input_manifest_sha256':c.digest(c.ROOT/'INPUT_SHA256.json')})
    records=[];histories={};best=None
    for trial in PROTOCOL['optimization']['trials']:
        reports,history,_=run_trial(trial,12,train,validation,initial,train_ref,val_ref,base_p,out)
        records.extend(reports);histories[trial['name']]=history
        for r in reports:
            if r['eligible'] and r['score']>score+PROTOCOL['selection']['minimum_gain'] and (best is None or r['score']>best['score']):best=r
    c.save(out/'training_histories.json',histories)
    if best is None:
        state=initial;refit={'needed':False,'selected':'start'}
    else:
        _,_,state=run_trial(best['trial'],best['epoch'],train,validation,initial,train_ref,val_ref,base_p,None)
        saved=torch.load(out/(best['name']+'.pth'),map_location='cpu',weights_only=True)['state_dict']
        if set(saved)!=set(state) or not all(torch.equal(saved[k],state[k]) for k in state):raise ValueError('Deterministic refit differs')
        refit={'needed':True,'selected':best['name'],'all_state_tensors_bitwise_equal':True}
    torch.save({'state_dict':state,'classes':c.CLASSES,'parameters':c.PARAMETERS},out/'selected.pth')
    selected=c.load_model(out/'selected.pth').cuda();changed=assert_frozen(selected,initial,1 if best is None else best['trial']['blocks'])
    selection={'selected':'start' if best is None else best['name'],'starting_metrics':c.metrics(validation['y'],base_p),'starting_score':score,
               'selected_metrics':c.metrics(validation['y'],base_p) if best is None else best['metrics'],'candidates':records,'selection_data':'validation only','selected_checkpoint_sha256':c.digest(out/'selected.pth'),'changed_state_tensors':changed,'labels_changed':False,'evaluation_labels_loaded_after_selection':True}
    c.save(out/'selection.json',selection);c.save(out/'refit.json',refit)
    data=dict(np.load(c.ROOT/'evaluation.npz',allow_pickle=False));validate_data(data,'followup',2171)
    p=probabilities(outputs(selected,data));repeat=probabilities(outputs(selected,data))
    if not np.array_equal(p,repeat):raise ValueError('Repeated inference differed')
    rows=[]
    for i in range(len(p)):
        row={'cohort':str(data['cohort'][i]),'dataset':str(data['dataset'][i]),'ecg_id':str(data['ecg_id'][i]),'reference':c.CLASSES[int(data['y'][i])],'prediction':c.CLASSES[int(p[i].argmax())],'tensor_sha256':str(data['tensor_sha256'][i])}
        row.update({f'prob_{label}':float(p[i,j]) for j,label in enumerate(c.CLASSES)});rows.append(row)
    with (out/'followup_predictions.csv').open('x',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    metrics={str(name):c.metrics(data['y'][idx],p[idx]) for name in np.unique(data['cohort']) if (idx:=data['cohort']==name).any()}
    c.save(out/'followup_metrics.json',{'records':len(rows),'cohorts':metrics,'repeat_probabilities_bitwise_equal':True})
    c.save(out/'SHA256SUMS.json',{path.name:c.digest(path) for path in sorted(out.iterdir()) if path.is_file()})
    print(json.dumps({'COMPLETED':selection['selected'],'parameters':c.PARAMETERS,'followup_records':len(rows)}),flush=True)


if __name__=='__main__':main()
