"""Collect a sealed diagnostic and recompute all fit/error accounting from probabilities."""
import argparse
import json
from pathlib import Path
import re
import shutil
import sys

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from titan_v4.reproducibility.refinement import sha256, read_manifest
from titan_v4.reproducibility.plateau_audit import error_decomposition, shared_errors
from titan_v4.reproducibility.physionet import preprocess_primary9_signal
from titan_v4.reproducibility.joint_refinement import PRIMARY5_INDEX, PATHOLOGY_CLASSES, calibrate_thresholds, pathology_metrics
from verify_joint_evidence import c

BUNDLE=ROOT/'outputs/reviewer_verification/plateau_audit_2026-10-14'
PHASES=['joint_refinement_2026-10-12','balanced_joint_2026-10-13','sequential_heads_2026-10-13','compatible_heads_2026-10-13',
        'expanded_population_2026-10-13','directed_pcgrad_2026-10-13','supervision_control_2026-10-14']


def equal(a,b):
    if isinstance(a,dict):return isinstance(b,dict) and set(a)==set(b) and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,list):return isinstance(b,list) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    if isinstance(a,float):return isinstance(b,(int,float)) and abs(a-b)<1e-12
    return a==b


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--collect',action='store_true');parser.add_argument('--write',action='store_true')
    args=parser.parse_args();state=ROOT/'data/cache/plateau_audit_cedia';out=BUNDLE/'results'
    if args.collect:
        if out.exists():raise FileExistsError('Preserve collected results')
        shutil.copytree(state/'results',out)
        shutil.copyfile(state/'INPUT_SHA256.json',BUNDLE/'training/INPUT_SHA256.json')
        raw=json.loads((state/'submission.json').read_text());job=raw['stdout'].strip()
        data=BUNDLE/'logs';data.mkdir()
        log_receipt={}
        for suffix in ('out','err'):
            source=state/f'slurm-{job}.{suffix}';target=data/source.name
            text=re.sub(r'/home/[^/\s]+','/home/REDACTED_USER',source.read_text(errors='replace'))
            text=text.replace('/Mateo Gavilanes/','/REDACTED_PROJECT/')
            target.write_text(text,encoding='utf-8')
            log_receipt[target.name]={'original_sha256':sha256(source),'published_sha256':sha256(target)}
        (data/'redaction_receipt.json').write_text(json.dumps(log_receipt,indent=2)+'\n',encoding='utf-8')
        (data/'submission.json').write_text(json.dumps({'job_id':job,'input_manifest_sha256':raw['input_manifest_sha256'],'exit_code':raw['exit_code']},indent=2)+'\n',encoding='utf-8')
    for name,digest in json.loads((out/'SHA256SUMS.json').read_text()).items():
        if Path(name).name!=name or sha256(out/name)!=digest:raise ValueError('Sealed diagnostic result differs')
    inputs=json.loads((BUNDLE/'training/INPUT_SHA256.json').read_text());execution=json.loads((out/'execution.json').read_text())
    if execution['input_manifest_sha256']!=sha256(BUNDLE/'training/INPUT_SHA256.json') or execution['parameters']!=1143751:raise ValueError('Execution receipt differs')
    for name,digest in inputs.items():
        path=BUNDLE/'training'/name
        if path.is_file() and sha256(path)!=digest:raise ValueError('Executed input differs: '+name)
    upstream=ROOT/'outputs/reviewer_verification/supervision_control_2026-10-14/training/INPUT_SHA256.json'
    previous=json.loads(upstream.read_text())
    for name in ('train.npz','validation.npz','evaluation.npz','path_extra.npz','path_targets.npz','path_validation.npz','rhythm_extra.npz'):
        if inputs[name]!=previous[name]:raise ValueError('Diagnosis data differs from preceding run')
    saved=json.loads((out/'diagnostic.json').read_text());protocol=json.loads((BUNDLE/'training/protocol.json').read_text())
    if saved['followup_read']:raise ValueError('Diagnostic should not open followup')
    a=dict(np.load(out/'diagnostic_probabilities.npz',allow_pickle=False))
    original=ROOT/'outputs/reviewer_verification/partial_backbone_refinement_2026-10-12/manifests'
    data=ROOT/'outputs/reviewer_verification/supervision_control_2026-10-14/data'
    train=read_manifest(original/'train_records.csv')+read_manifest(data/'path_extra_records.csv')+read_manifest(data/'rhythm_extra_records.csv')
    val=read_manifest(original/'validation_records.csv');pval=read_manifest(data/'path_validation_records.csv')
    for prefix,rows in [('training',train),('validation',val)]:
        y=np.asarray([int(r['rhythm_target']) if 'rhythm_target' in r else c.CLASSES.index(r['reference_label']) for r in rows])
        if not np.array_equal(a[prefix+'_y'],y) or list(a[prefix+'_ecg_id'])!=[r['ecg_id'] for r in rows] or list(a[prefix+'_dataset'])!=[r['dataset'] for r in rows]:raise ValueError('Reference row identity differs')
    py=np.asarray([[int(r['target_'+label]) for label in PATHOLOGY_CLASSES] for r in pval],np.float32)
    if not np.array_equal(a['validation_path_targets'],py) or list(a['validation_path_ecg_id'])!=[r['ecg_id'] for r in pval]:raise ValueError('Pathology validation identity differs')
    from titan_v4.reproducibility.joint_refinement import pathology_targets
    import ast
    meta={str(int(r['ecg_id'])):r for r in read_manifest(data/'ptbxl_database.csv')}
    target_mask=[pathology_targets(r['dataset'],ast.literal_eval(meta[str(int(r['ecg_id']))]['scp_codes']) if r['dataset']=='ptb-xl' else {}) for r in train]
    if not np.array_equal(a['training_path_targets'],np.stack([x[0] for x in target_mask])) or not np.array_equal(a['training_path_mask'],np.stack([x[1] for x in target_mask])):raise ValueError('Pathology training references differ')
    reports={};prediction_rows=[]
    for name,digest in protocol['checkpoints'].items():
        if sha256(BUNDLE/'training'/(name+'.pth'))!=digest or saved['models'][name]['checkpoint_sha256']!=digest or not saved['models'][name]['state_unchanged']:raise ValueError('Read-only model receipt differs')
        thresholds=calibrate_thresholds(py,a[name+'_val_path'],role='validation')
        reports[name]={}
        for split in ('training','validation'):
            y=a[split+'_y'];known=y>=0;rp=a[name+('_train_rhythm' if split=='training' else '_val_rhythm')]
            observed=a['training_path_mask'][:,0]==1 if split=='training' else np.ones(len(py),bool)
            yy=a['training_path_targets'][observed] if split=='training' else py
            pp=a[name+('_train_path' if split=='training' else '_val_path')][observed]
            score={'rhythm':c.metrics(y[known],rp[known]),'pathology':pathology_metrics(yy,pp,thresholds),
                   'pathology_primary5_ap':float(np.mean([average_precision_score(yy[:,j],pp[:,j]) for j in PRIMARY5_INDEX]))}
            sources=a[split+'_dataset']
            score['sources']={s:c.metrics(y[(sources==s)&known],rp[(sources==s)&known]) for s in sorted(set(sources.tolist()))}
            if not equal(score,saved['models'][name][split]):raise ValueError('GPU/local diagnostic score differs')
            reports[name][split]={'accuracy':score['rhythm']['accuracy'],'macro_f1':score['rhythm']['macro_f1'],
                                 'pathology_macro_f1':score['pathology']['primary5_macro_f1'],'pathology_ap':score['pathology_primary5_ap'],
                                 'pathology_auroc':float(np.mean([roc_auc_score(yy[:,j],pp[:,j]) for j in PRIMARY5_INDEX])),
                                 'pathology_prevalence':yy.mean(0).tolist(),
                                 'rhythm_errors':error_decomposition(y[known],rp[known].argmax(1)),'sources':score['sources']}
        total=sum(v['n'] for v in reports[name]['validation']['sources'].values())
        standardized=0
        for source,metrics in reports[name]['validation']['sources'].items():
            train_metrics=reports[name]['training']['sources'][source]
            for label,support in metrics['support'].items():
                if support and not train_metrics['support'][label]:raise ValueError('Validation stratum lacks training support')
                standardized+=support*train_metrics['recall'][label]
        reports[name]['training_accuracy_at_validation_source_class_mix']=standardized/total
        reports[name]['native_rbbb_subtypes']={}
        for split,short in [('training','train'),('validation','val')]:
            y=a[split+'_y'];sources=a[split+'_dataset'];ids=a[split+'_ecg_id'];pred=a[name+'_'+short+'_rhythm'].argmax(1)
            groups={}
            for i in np.flatnonzero((sources=='ptb-xl')&(y==5)):
                codes=ast.literal_eval(meta[str(int(ids[i]))]['scp_codes'])
                group='both' if 'IRBBB' in codes and 'CRBBB' in codes else 'IRBBB' if 'IRBBB' in codes else 'CRBBB' if 'CRBBB' in codes else 'other'
                item=groups.setdefault(group,{'support':0,'correct':0});item['support']+=1;item['correct']+=int(pred[i]==5)
            reports[name]['native_rbbb_subtypes'][split]=groups
        prediction_rows.append(a[name+'_val_rhythm'].argmax(1))
    candidates=[]
    for phase in PHASES:
        path=ROOT/'outputs/reviewer_verification'/phase/'results/selection.json';selection=json.loads(path.read_text())
        for r in selection['candidates']:
            candidates.append({'phase':phase,'name':r['name'],'accuracy':r['metrics']['accuracy'],'macro_f1':r['metrics']['macro_f1'],
                               'pathology_macro_f1':r['pathology']['primary5_macro_f1'],'eligible':r['eligible'],
                               'failed_protection_checks':[k for k,v in r['preservation_checks'].items() if not v]})
    t=np.arange(5000)/500;raw=np.column_stack([np.sin((i+1)*2*np.pi*t)+.2*np.cos(3*np.pi*t) for i in range(6)])
    leads=['I','II','III','aVR','aVL','aVF'];delta=float(np.max(np.abs(preprocess_primary9_signal(raw,leads,500)-preprocess_primary9_signal(raw*np.arange(2,8)[None,:],leads,500))))
    summary={'diagnostic_only':True,'model_fitting_performed':False,'followup_read':False,'models':reports,'candidate_entries':len(candidates),
             'candidate_landscape':candidates,'shared_validation_errors':shared_errors(a['validation_y'],np.stack(prediction_rows)),
             'positive_channel_gain_invariance':{'synthetic_control_only':True,'maximum_tensor_difference':delta,'threshold':1e-4},
             'source_sha256':{phase:sha256(ROOT/'outputs/reviewer_verification'/phase/'results/selection.json') for phase in PHASES}}
    if delta>=1e-4:raise ValueError('Amplitude invariance control did not reproduce')
    target=BUNDLE/'audit.json'
    if args.write:target.write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    elif not equal(summary,json.loads(target.read_text())):raise ValueError('Audit report differs')
    sums=BUNDLE/'SHA256SUMS.json';inventory={p.relative_to(BUNDLE).as_posix():sha256(p) for p in sorted(BUNDLE.rglob('*')) if p.is_file() and p!=sums and '__pycache__' not in p.parts}
    if args.write:sums.write_text(json.dumps(inventory,indent=2)+'\n',encoding='utf-8')
    elif inventory!=json.loads(sums.read_text()):raise ValueError('Audit artifact hashes differ')
    print(json.dumps({'artifacts':len(inventory),'models':{k:{s:({n:v for n,v in r.items() if n not in ('rhythm_errors','sources')} if isinstance(r,dict) else r) for s,r in reports[k].items()} for k in reports},'common_errors':summary['shared_validation_errors'],'gain_invariance':delta},indent=2))


if __name__=='__main__':main()
