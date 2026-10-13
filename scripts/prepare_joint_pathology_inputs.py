"""Add patient-disjoint, training-only Primary-5 examples from pinned PTB-XL metadata."""
import argparse
import ast
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from titan_v4.reproducibility.joint_refinement import PATHOLOGY_CLASSES, OFFICIAL_RHYTHM_CODES, pathology_targets, select_extra_training, resolve_historical_ptb_ids, unambiguous_rhythm_target
from titan_v4.data.label_registry import SCP_TO_RHYTHM
from titan_v4.data.primary9_registry import PRIMARY9_CLASSES
from titan_v4.reproducibility.refinement import _ensure_file, read_manifest, sha256
from titan_v4.reproducibility.physionet import preprocess_primary9_signal

META_SHA='7600de9c1b27d181d850b3c6038a35d7c3ddb6bb33b702e3a20252a6859d216b'
SCP_SHA='ad05b0b1fcae83bb1230755ad9cfc7c96f303feddc08a4a9ad5bdc9ca63bac8f'
INDEX_SHA='b7224b92b341511ec3ceb13dc6652079b2c36a06504bcb49506f157f51dc695d'
BASE=ROOT/'outputs/reviewer_verification/partial_backbone_refinement_2026-10-12/manifests'


def write_csv(path,rows):
    with path.open('x',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n')
        writer.writeheader();writer.writerows(rows)


def fetch_pinned_file(path,url,expected,download,*,sleep=time.sleep):
    for attempt in range(3):
        try:
            _ensure_file(path,url,expected,download)
            return
        except OSError:
            if not download or attempt==2:raise
            print(f'Transport retry {attempt+1}/2 for {path.name}; expected SHA-256 unchanged',flush=True)
            sleep(2**(attempt+1))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--metadata',type=Path,default=ROOT/'data/cache/physionet/ptb-xl/1.0.3/ptbxl_database.csv')
    parser.add_argument('--data-root',type=Path,default=ROOT/'data/cache/physionet')
    parser.add_argument('--source-index',type=Path,default=ROOT/'data/cache/refinement_source_indexes/ptb-xl/1.0.3/SHA256SUMS.txt')
    parser.add_argument('--scp-statements',type=Path,default=ROOT/'data/cache/physionet/ptb-xl/1.0.3/scp_statements.csv')
    parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--offline',action='store_true')
    args=parser.parse_args()
    if args.workers<1:parser.error('workers must be positive')
    if args.output.exists():raise FileExistsError('Preserve completed evidence; use a new output directory')
    _ensure_file(args.metadata,'https://physionet.org/files/ptb-xl/1.0.3/ptbxl_database.csv',META_SHA,not args.offline)
    _ensure_file(args.source_index,'https://physionet.org/files/ptb-xl/1.0.3/SHA256SUMS.txt',INDEX_SHA,not args.offline)
    _ensure_file(args.scp_statements,'https://physionet.org/files/ptb-xl/1.0.3/scp_statements.csv',SCP_SHA,not args.offline)
    official_rhythms={r[''] for r in read_manifest(args.scp_statements) if float(r['rhythm'] or 0)>0}
    if official_rhythms!=OFFICIAL_RHYTHM_CODES:raise ValueError('Official rhythm taxonomy differs from the pinned domain')
    official={}
    for line in args.source_index.read_text().splitlines():
        digest,name=line.split(maxsplit=1);official[name.lstrip('*').removeprefix('./')]=digest
    if official['ptbxl_database.csv']!=META_SHA:raise ValueError('Official index/metadata hash differs')
    metadata=read_manifest(args.metadata);by_id={str(int(r['ecg_id'])):r for r in metadata}
    existing={role:read_manifest(BASE/(role+'_records.csv')) for role in ('train','validation','followup')}
    patient_sets={role:{by_id[str(int(r['ecg_id']))]['patient_id'] for r in rows if r['dataset']=='ptb-xl'} for role,rows in existing.items()}
    if patient_sets['train'] & (patient_sets['validation']|patient_sets['followup']):raise ValueError('Existing PTB-XL patient overlap')
    if patient_sets['validation'] & patient_sets['followup']:raise ValueError('Existing validation/followup patient overlap')
    existing_ids={str(int(r['ecg_id'])) for r in existing['train'] if r['dataset']=='ptb-xl'}
    historical=read_manifest(ROOT/'outputs/reviewer_verification/arrhythmia_primary9/cedia_gold_2026-05-26/primary9_label_history_672.csv')
    aliases=[str(int(r['record_id'][2:])) for r in historical if r['record_id'].upper().startswith('HR')]
    resolved_aliases=resolve_historical_ptb_ids(aliases,set(by_id))
    historical_patients={by_id[rid]['patient_id'] for rid in resolved_aliases}
    if patient_sets['train'] & historical_patients:raise ValueError('Existing rhythm training overlaps historical PTB-XL patients')
    extra=select_extra_training(metadata,patient_sets['validation']|patient_sets['followup']|historical_patients,existing_ids)
    args.output.mkdir(parents=True)
    target_csv=[];packed={};receipt={}
    for role,rows in existing.items():
        targets=[];masks=[]
        for row in rows:
            is_ptb=row['dataset']=='ptb-xl'
            meta=by_id[str(int(row['ecg_id']))] if is_ptb else None
            scores=ast.literal_eval(meta['scp_codes']) if is_ptb else {}
            y,mask=pathology_targets(row['dataset'],scores);targets.append(y);masks.append(mask)
            entry={'role':role,'cohort':row.get('cohort',''),'dataset':row['dataset'],'ecg_id':row['ecg_id'],
                   'tensor_sha256':row['tensor_sha256'],'pathology_observed':int(is_ptb),
                   'strat_fold':meta['strat_fold'] if is_ptb else '',
                   'patient_token':hashlib.sha256(('ptb-xl:'+meta['patient_id']).encode()).hexdigest() if is_ptb else '',
                   'source_scp_codes':json.dumps(scores,sort_keys=True)}
            entry.update({'target_'+name:int(y[j]) for j,name in enumerate(PATHOLOGY_CLASSES)})
            target_csv.append(entry)
        packed[role+'_targets']=np.stack(targets);packed[role+'_mask']=np.stack(masks)
        receipt[role]={'records':len(rows),'pathology_observed':int(np.stack(masks)[:,0].sum()),
                       'positive_support':dict(zip(PATHOLOGY_CLASSES,np.stack(targets).sum(0).astype(int).tolist()))}
    np.savez_compressed(args.output/'path_targets.npz',**packed)
    write_csv(args.output/'path_targets.csv',target_csv)
    def read_extra(meta):
        import wfdb
        record_path=meta['filename_hr'];root=args.data_root/'ptb-xl/1.0.3'
        row={'ecg_id':str(int(meta['ecg_id'])),'dataset':'ptb-xl','version':'1.0.3','doi':'10.13026/kfzx-aw45',
             'source_release':'ptb-xl','record_path':record_path,'signal_filename':Path(record_path).name+'.dat',
             'header_sha256':official[record_path+'.hea'],'signal_sha256':official[record_path+'.dat']}
        url='https://physionet.org/files/ptb-xl/1.0.3/'
        for suffix in ('.hea','.dat'):
            fetch_pinned_file(root/(record_path+suffix),url+record_path+suffix,official[record_path+suffix],not args.offline)
        record=wfdb.rdrecord(str(root/record_path),physical=True)
        if record.fs!=500:raise ValueError('Unexpected source sampling rate')
        tensor=preprocess_primary9_signal(record.p_signal,record.sig_name,record.fs)
        row['tensor_sha256']=hashlib.sha256(tensor.astype('<f4',copy=False).tobytes()).hexdigest()
        row['strat_fold']=meta['strat_fold'];row['patient_token']=hashlib.sha256(('ptb-xl:'+meta['patient_id']).encode()).hexdigest()
        scores=ast.literal_eval(meta['scp_codes']);y,mask=pathology_targets('ptb-xl',scores)
        rhythm=unambiguous_rhythm_target(scores,SCP_TO_RHYTHM)
        row['rhythm_target']=rhythm;row['reference_label']=PRIMARY9_CLASSES[rhythm] if rhythm>=0 else 'unknown'
        row['source_scp_codes']=json.dumps(scores,sort_keys=True)
        row.update({'target_'+name:int(y[j]) for j,name in enumerate(PATHOLOGY_CLASSES)})
        return tensor,y,mask,row
    tensors=[];targets=[];masks=[];rows=[]
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for i,(x,y,mask,row) in enumerate(executor.map(read_extra,extra),1):
            tensors.append(x);targets.append(y);masks.append(mask);rows.append(row)
            if i%100==0:print(f'Official hashes and source tensors verified: {i}/{len(extra)}',flush=True)
    np.savez_compressed(args.output/'path_extra.npz',x=np.stack(tensors),targets=np.stack(targets),mask=np.stack(masks),
                        y=np.asarray([r['rhythm_target'] for r in rows],dtype=np.int64),
                        ecg_id=np.asarray([r['ecg_id'] for r in rows]),tensor_sha256=np.asarray([r['tensor_sha256'] for r in rows]))
    write_csv(args.output/'path_extra_records.csv',rows)
    receipt['expansion']={'records':len(rows),'positive_support':dict(zip(PATHOLOGY_CLASSES,np.stack(targets).sum(0).astype(int).tolist())),
                          'selection':'All remaining Primary-5-positive fold 1-8 records, excluding validation/followup/historical patients',
                          'unknown_rhythm_targets_not_used':True,'overlapping_validation_followup_patients':0,
                          'rhythm_positive_support':{name:sum(r['reference_label']==name for r in rows) for name in PRIMARY9_CLASSES},
                          'rhythm_unknown_records':sum(r['rhythm_target']<0 for r in rows),
                          'historical_ptb_aliases':len(aliases),'historical_aliases_matched':len(resolved_aliases),
                          'historical_alias_replacements':{a:b for a,b in zip(aliases,resolved_aliases) if a!=b},
                          'overlapping_historical_patients':0,
                          'metadata_sha256':META_SHA,'scp_statements_sha256':SCP_SHA,
                          'official_rhythm_codes':sorted(official_rhythms),'official_index_sha256':sha256(args.source_index)}
    receipt['files']={p.name:sha256(p) for p in sorted(args.output.iterdir()) if p.is_file()}
    (args.output/'preparation.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
