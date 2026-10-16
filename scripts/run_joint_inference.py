"""Reproduce both heads of one selected checkpoint directly from original WFDB signals."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import json
from pathlib import Path
import platform
import sys

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from titan_v4.reproducibility.inference import load_primary9_model,primary9_metrics
from titan_v4.data.primary9_registry import PRIMARY9_CLASSES
from titan_v4.reproducibility.refinement import read_manifest,load_tensor,sha256
from titan_v4.reproducibility.joint_refinement import PATHOLOGY_CLASSES,pathology_targets,pathology_metrics

BUNDLE=ROOT/'outputs/reviewer_verification/joint_refinement_2026-10-12'


def main():
    global BUNDLE
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',type=Path,default=BUNDLE)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--data-root',type=Path,default=ROOT/'data/cache/physionet')
    parser.add_argument('--model',choices=['joint_selected','rhythm_selected'],default='joint_selected')
    parser.add_argument('--offline',action='store_true')
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args();BUNDLE=args.bundle
    if args.output.exists():raise FileExistsError('Use a new output directory')
    if args.workers<1:parser.error('workers must be positive')
    manifest=ROOT/'outputs/reviewer_verification/partial_backbone_refinement_2026-10-12/manifests/followup_records.csv'
    rows=read_manifest(manifest);key=lambda r:(r['cohort'],r['dataset'],r['ecg_id'])
    expected={key(r):r for r in read_manifest(BUNDLE/'results'/(args.model+'_predictions.csv'))}
    labels={key(r):r for r in read_manifest(BUNDLE/'data/path_targets.csv') if r['role']=='followup'}
    if len(expected)!=len(rows) or set(expected)!=set(map(key,rows)) or set(labels)!=set(expected):raise ValueError('Record sets differ')
    checkpoint=BUNDLE/'model'/(args.model+'.pth')
    selection=json.loads((BUNDLE/'results/selection.json').read_text())
    if sha256(checkpoint)!=selection['checkpoints'][args.model]:raise ValueError('Selected checkpoint hash differs')
    record=selection[args.model]
    thresholds=np.asarray((record or selection['baseline'])['thresholds'])
    torch.set_num_threads(2);model=load_primary9_model(checkpoint)
    if sum(p.numel() for p in model.parameters())!=1143751:raise ValueError('Architecture changed')
    def read(row):return load_tensor(row,args.data_root,download=not args.offline)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:x=np.stack(list(executor.map(read,rows)))
    rhythm=[];path=[]
    with torch.inference_mode():
        for start in range(0,len(x),128):
            values=model(torch.from_numpy(x[start:start+128]))
            if [v.shape[1] for v in values]!=[9,1,3,10]:raise ValueError('Four-head output contract changed')
            rhythm.append(torch.softmax(values[0],1).numpy());path.append(torch.sigmoid(values[3]).numpy())
    rp=np.concatenate(rhythm);pp=np.concatenate(path);targets=[];observed=[];output=[]
    for i,row in enumerate(rows):
        label=labels[key(row)];old=expected[key(row)]
        if row['reference_label']!=old['reference'] or row['tensor_sha256']!=old['tensor_sha256'] or row['tensor_sha256']!=label['tensor_sha256']:raise ValueError('Reference or signal hash changed')
        y,mask=pathology_targets(row['dataset'],json.loads(label['source_scp_codes']))
        if int(mask[0])!=int(old['pathology_observed']) or any(mask[j] and int(y[j])!=int(old['target_'+name]) for j,name in enumerate(PATHOLOGY_CLASSES)):
            raise ValueError('Pathology reference changed')
        targets.append(y);observed.append(bool(mask[0]))
        item={**{k:row[k] for k in ['cohort','dataset','ecg_id']},'reference':row['reference_label'],
              'prediction':PRIMARY9_CLASSES[int(rp[i].argmax())],'pathology_observed':int(mask[0])}
        item.update({'prob_'+name:float(rp[i,j]) for j,name in enumerate(PRIMARY9_CLASSES)})
        item.update({'score_'+name:float(pp[i,j]) for j,name in enumerate(PATHOLOGY_CLASSES)})
        output.append(item)
    reference_rhythm=np.asarray([[float(expected[key(r)]['prob_'+name]) for name in PRIMARY9_CLASSES] for r in rows])
    reference_path=np.asarray([[float(expected[key(r)]['score_'+name]) for name in PATHOLOGY_CLASSES] for r in rows])
    max_r=float(np.max(np.abs(rp-reference_rhythm)));max_p=float(np.max(np.abs(pp-reference_path)))
    same_r=int((rp.argmax(1)==reference_rhythm.argmax(1)).sum());same_p=int(np.all((pp>=thresholds)==(reference_path>=thresholds),axis=1).sum())
    idx=np.asarray(observed);y=np.asarray(targets)
    report={'model':args.model,'parameters':1143751,'records':len(rows),'pathology_records':int(idx.sum()),
            'checkpoint_sha256':sha256(checkpoint),'source_hashes_verified':len(rows),
            'comparison':{'identical_rhythm_top1':same_r,'identical_pathology_decisions':same_p,
                          'max_rhythm_probability_difference':max_r,'max_pathology_probability_difference':max_p,'tolerance':2e-5},
            'rhythm':primary9_metrics([r['reference_label'] for r in rows],[r['prediction'] for r in output]),
            'pathology':pathology_metrics(y[idx],pp[idx],thresholds),
            'environment':{'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,'device':'cpu'}}
    passed=same_r==len(rows) and same_p==len(rows) and max(max_r,max_p)<=2e-5
    report['passed']=passed;args.output.mkdir(parents=True)
    with (args.output/'predictions.csv').open('x',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(output[0]),lineterminator='\n');writer.writeheader();writer.writerows(output)
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'model':args.model,'passed':passed,'comparison':report['comparison'],'rhythm_accuracy':report['rhythm']['accuracy'],
                      'pathology_accuracy':report['pathology']['primary5_mean_label_accuracy'],'pathology_f1':report['pathology']['primary5_macro_f1']},indent=2))
    return 0 if passed else 1


if __name__=='__main__':raise SystemExit(main())
