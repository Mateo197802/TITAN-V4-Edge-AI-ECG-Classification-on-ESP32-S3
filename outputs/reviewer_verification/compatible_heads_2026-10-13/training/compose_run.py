"""Validate one physically reconstructed PTH, never combine two reported metrics."""
import json
import os
import platform
import socket
import time

import numpy as np
import torch

import contract as c
import baseline_trainer as b
from composition_metrics import compose_head_states
from balanced_metrics import select_candidates

b.PROTOCOL=json.loads((c.ROOT/'protocol.json').read_text())


def main():
    c.require_cedia_gpu();os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8';torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.set_num_threads(2)
    for name,h in json.loads((c.ROOT/'INPUT_SHA256.json').read_text()).items():
        if c.digest(c.ROOT/name)!=h:raise ValueError('Input hash differs: '+name)
    protocol=b.PROTOCOL
    for name,key in [('start.pth','start_sha256'),('path_source.pth','path_source_sha256'),('reference.pth','reference_sha256')]:
        if c.digest(c.ROOT/name)!=protocol[key]:raise ValueError('Source checkpoint differs')
    source=json.loads((c.ROOT/'source_selection.json').read_text());inventory=json.loads((c.ROOT/'source_results_SHA256.json').read_text())
    eligible=[r for r in source['candidates'] if r['eligible']]
    r=max(eligible,key=lambda v:(v['rhythm_score'],v['name']));p=max(source['candidates'],key=lambda v:(v['pathology']['primary5_macro_f1'],v['name']))
    if r['name']!=protocol['head_selection']['chosen_rhythm'] or p['name']!=protocol['head_selection']['chosen_pathology']:
        raise ValueError('Head choice differs from validation-only rule')
    if inventory[r['name']+'.pth']!=protocol['start_sha256'] or inventory[p['name']+'.pth']!=protocol['path_source_sha256']:
        raise ValueError('Source execution inventory differs')
    rm=c.load_model(c.ROOT/'start.pth');pm=c.load_model(c.ROOT/'path_source.pth')
    rs=rm.state_dict();ps=pm.state_dict();state=compose_head_states(rs,ps)
    rebuilt=compose_head_states(rs,ps)
    if not all(torch.equal(state[k],rebuilt[k]) for k in state):raise ValueError('Reconstruction differs')
    model=c.load_model(c.ROOT/'start.pth');model.load_state_dict(state,strict=True);model=model.cuda()
    reference_model=c.load_model(c.ROOT/'reference.pth').cuda();fallback={k:v.cpu().clone() for k,v in reference_model.state_dict().items()}
    val=dict(np.load(c.ROOT/'validation.npz',allow_pickle=False))
    with np.load(c.ROOT/'path_targets.npz',allow_pickle=False) as f:targets=f['validation_targets'];mask=f['validation_mask']
    actual=b.outputs(model,val['x']);rv=b.outputs(rm.cuda(),val['x']);pv=b.outputs(pm.cuda(),val['x']);reference=b.outputs(reference_model,val['x'])
    if not np.array_equal(actual[0],rv[0]) or not np.array_equal(actual[3],pv[3]) or any(not np.array_equal(a,z) for a,z in zip(actual[1:3],rv[1:3])):
        raise ValueError('Actual composed network outputs do not match selected head sources')
    bp=b.rhythm_prob(reference);sources=c.source_metrics(val,bp);observed=mask[:,0]==1
    threshold=b.calibrate_thresholds(targets[observed],b.path_prob(reference)[observed],role='validation')
    path=b.pathology_metrics(targets[observed],b.path_prob(reference)[observed],threshold)
    base={'metrics':c.metrics(val['y'],bp),'sources':sources,'rhythm_score':float(np.mean([z['macro_f1'] for z in sources.values()])),
          'pathology':path,'thresholds':threshold.tolist()};base['joint_score']=base['rhythm_score']+.5*path['primary5_macro_f1']
    candidate={'name':'compatible_heads_e00','trial':protocol['optimization']['trials'][0],'epoch':0,**b.validation_report(val,actual,reference,targets,mask,base)}
    selected=select_candidates([candidate],base,role='validation');out=c.ROOT/'results';out.mkdir(exist_ok=False)
    c.save(out/'execution.json',{'job_id':os.environ['SLURM_JOB_ID'],'hostname':socket.gethostname(),'python':platform.python_version(),'platform':platform.platform(),
           'torch':torch.__version__,'numpy':np.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),'parameters':c.PARAMETERS,
           'input_manifest_sha256':c.digest(c.ROOT/'INPUT_SHA256.json'),'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'gradient_fitting':False})
    for name,value in [('baseline',reference),('compatible_heads_e00',actual)]:
        np.savez_compressed(out/(name+'_validation.npz'),rhythm_probabilities=b.rhythm_prob(value),pathology_probabilities=b.path_prob(value),
                            y=val['y'],dataset=val['dataset'],ecg_id=val['ecg_id'],path_targets=targets,path_mask=mask)
    torch.save({'state_dict':state,'classes':c.CLASSES,'pathology_classes':b.PATHOLOGY_CLASSES,'parameters':c.PARAMETERS},out/'compatible_heads_e00.pth')
    refits={}
    for role,record in selected.items():
        chosen=state if record else fallback
        torch.save({'state_dict':chosen,'classes':c.CLASSES,'pathology_classes':b.PATHOLOGY_CLASSES,'parameters':c.PARAMETERS},out/(role+'.pth'))
        refits[role]={'needed':False,'operation':'deterministic head reconstruction, not fitting','all_state_tensors_bitwise_equal':True}
    c.save(out/'refit.json',refits)
    c.save(out/'composition.json',{'rhythm_source':r['name'],'pathology_source':p['name'],'shared_state_bitwise_equal':True,
           'actual_validation_rhythm_matches_source':True,'actual_validation_pathology_matches_source':True,'actual_auxiliary_matches_source':True,
           'deterministic_reconstruction':True,'labels_changed':False,'selection_role':'validation only'})
    selection={'baseline':base,'candidates':[candidate],**selected,'selection_data':'validation only','thresholds_fitted_on_validation_only':True,
               'evaluation_labels_loaded_after_selection':True,'references_changed':False,'checkpoints':{role:c.digest(out/(role+'.pth')) for role in selected}}
    c.save(out/'selection.json',selection)
    # Composed state, thresholds, validation and selection are persisted before evaluation.
    data=dict(np.load(c.ROOT/'evaluation.npz',allow_pickle=False))
    with np.load(c.ROOT/'path_targets.npz',allow_pickle=False) as f:ey=f['followup_targets'];em=f['followup_mask']
    for role,record in [('baseline',None),*selected.items()]:
        model=c.load_model(c.ROOT/'reference.pth' if role=='baseline' else out/(role+'.pth')).cuda()
        b.export_predictions(out,role,model,data,ey,em,np.asarray((record or base)['thresholds']))
    c.save(out/'SHA256SUMS.json',{z.name:c.digest(z) for z in sorted(out.iterdir()) if z.is_file()})
    print(json.dumps({'COMPLETED':True,'candidate':candidate,'joint_selected':selected['joint_selected'] is not None,'rhythm_selected':selected['rhythm_selected'] is not None}),flush=True)


if __name__=='__main__':main()
