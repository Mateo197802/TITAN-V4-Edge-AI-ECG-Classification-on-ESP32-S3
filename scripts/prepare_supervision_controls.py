"""Audit training source budgets and freeze metadata-only supervision weights."""
import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from prepare_joint_pathology_inputs import write_csv, META_SHA
from titan_v4.reproducibility.refinement import read_manifest, sha256
from titan_v4.reproducibility.supervision_control import source_masses, fixed_source_weights, confidence_weights, patient_folds
from titan_v4.reproducibility.joint_refinement import SCP_MAPPING, PATHOLOGY_CLASSES, pathology_targets
from titan_v4.data.primary9_registry import PRIMARY9_CLASSES


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--bundle',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Use fresh controls directory')
    data=args.bundle/'data'
    meta=data/'ptbxl_database.csv'
    if sha256(meta)!=META_SHA:raise ValueError('Pinned metadata differs')
    byid={str(int(r['ecg_id'])):r for r in read_manifest(meta)}
    original=read_manifest(ROOT/'outputs/reviewer_verification/partial_backbone_refinement_2026-10-12/manifests/train_records.csv')
    extra=read_manifest(data/'path_extra_records.csv'); directed=read_manifest(data/'rhythm_extra_records.csv')
    rows=original+extra+directed
    y=np.asarray([int(r['rhythm_target']) if 'rhythm_target' in r else PRIMARY9_CLASSES.index(r['reference_label']) for r in rows],np.int64)
    source=np.asarray([r['dataset'] for r in rows])
    before=len(original)+len(extra)
    mass=source_masses(y[:before],source[:before],role='train')
    weights=fixed_source_weights(y,source,mass,role='train')
    labels=[]; masks=[]; human=[]; likelihood=[]; entries=[]
    support={label:Counter() for label in PATHOLOGY_CLASSES}
    for index,row in enumerate(rows):
        known=row['dataset']=='ptb-xl'
        native=byid[str(int(row['ecg_id']))] if known else None
        scores=ast.literal_eval(native['scp_codes']) if known else {}
        target,mask=pathology_targets(row['dataset'],scores)
        flag=native['validated_by_human'] if known else 'False'
        if flag not in ('True','False'):raise ValueError('Unrecognized human-validation flag')
        certainty=np.full(10,100,np.float32)
        for code,value in scores.items():
            if code in SCP_MAPPING:
                j=SCP_MAPPING[code]; certainty[j]=value
                support[PATHOLOGY_CLASSES[j]]['unknown' if value==0 else 'weak' if value<100 else '100']=support[PATHOLOGY_CLASSES[j]].get('unknown' if value==0 else 'weak' if value<100 else '100',0)+1
        labels.append(target); masks.append(mask); human.append(flag=='True'); likelihood.append(certainty)
        entries.append({'row_index':index,'dataset':row['dataset'],'ecg_id':row['ecg_id'],'rhythm_target':int(y[index]),
                        'rhythm_training_weight':float(weights[index]),'pathology_observed':int(known),'source_validated_by_human':flag})
    targets=np.stack(labels); mask=np.stack(masks); human=np.asarray(human); likelihood=np.stack(likelihood)
    human_weight, certainty_weight=confidence_weights(targets,mask,human,likelihood,role='train')
    pval=read_manifest(data/'path_validation_records.csv')
    folds=patient_folds([r['patient_token'] for r in pval])
    args.output.mkdir(parents=True)
    np.savez_compressed(args.output/'supervision_weights.npz',rhythm_weights=weights,human_weights=human_weight,
                        certainty_weights=certainty_weight,path_validation_folds=folds,
                        training_y=y,training_dataset=source,training_ecg_id=np.asarray([r['ecg_id'] for r in rows]))
    write_csv(args.output/'training_controls_records.csv',entries)
    audit={'metadata_sha256':META_SHA,'training_records':len(rows),'native_pathology_records':int(mask[:,0].sum()),
           'native_human_validation':dict(Counter(str(v) for v in human[mask[:,0]==1])),
           'source_mass_before':mass,'equal_pair_source_mass_after':source_masses(y,source,role='train'),
           'fixed_source_mass_measured':{s:float(weights[source==s].sum()/weights.sum()) for s in mass},
           'source_positive_likelihoods':{k:dict(v) for k,v in support.items()},
           'calibration_patient_groups':5,'calibration_records_per_group':np.bincount(folds,minlength=5).tolist(),
           'reference_labels_changed':False,'evaluation_data_used_for_weights':False,
           'unknown_likelihood_policy':'Zero is unknown, not a negative or zero-weight positive',
           'source_files':{p.name:sha256(p) for p in (meta,data/'path_extra_records.csv',data/'rhythm_extra_records.csv')},
           'files':{p.name:sha256(p) for p in args.output.iterdir() if p.is_file()}}
    (args.output/'audit.json').write_text(json.dumps(audit,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(audit,indent=2))


if __name__=='__main__':main()
