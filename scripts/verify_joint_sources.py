"""Independently compare every new target and patient exclusion with official metadata."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from titan_v4.reproducibility.refinement import read_manifest,_ensure_file,sha256
from titan_v4.reproducibility.joint_refinement import PATHOLOGY_CLASSES,OFFICIAL_RHYTHM_CODES,pathology_targets,select_extra_training,resolve_historical_ptb_ids,unambiguous_rhythm_target
from titan_v4.data.label_registry import SCP_TO_RHYTHM

BUNDLE=ROOT/'outputs/reviewer_verification/joint_refinement_2026-10-12'


def main():
    global BUNDLE
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',type=Path,default=BUNDLE)
    parser.add_argument('--metadata',type=Path,default=ROOT/'data/cache/physionet/ptb-xl/1.0.3/ptbxl_database.csv')
    parser.add_argument('--source-index',type=Path,default=ROOT/'data/cache/refinement_source_indexes/ptb-xl/1.0.3/SHA256SUMS.txt')
    parser.add_argument('--scp-statements',type=Path,default=ROOT/'data/cache/physionet/ptb-xl/1.0.3/scp_statements.csv')
    parser.add_argument('--offline',action='store_true')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args();BUNDLE=args.bundle
    receipt=json.loads((BUNDLE/'data/preparation.json').read_text())['expansion']
    base='https://physionet.org/files/ptb-xl/1.0.3/'
    _ensure_file(args.metadata,base+'ptbxl_database.csv',receipt['metadata_sha256'],not args.offline)
    _ensure_file(args.source_index,base+'SHA256SUMS.txt',receipt['official_index_sha256'],not args.offline)
    _ensure_file(args.scp_statements,base+'scp_statements.csv',receipt['scp_statements_sha256'],not args.offline)
    if {r[''] for r in read_manifest(args.scp_statements) if float(r['rhythm'] or 0)>0}!=OFFICIAL_RHYTHM_CODES:
        raise ValueError('Complete official rhythm taxonomy differs')
    meta={str(int(r['ecg_id'])):r for r in read_manifest(args.metadata)}
    index={}
    for line in args.source_index.read_text().splitlines():
        h,name=line.split(maxsplit=1);index[name.lstrip('*').removeprefix('./')]=h
    targets=read_manifest(BUNDLE/'data/path_targets.csv');patients={role:set() for role in ['train','validation','followup']}
    original_train=set();checked=0
    for r in targets:
        if r['dataset']!='ptb-xl':
            if r['pathology_observed']!='0':raise ValueError('Unknown source was assigned pathology references')
            continue
        rid=str(int(r['ecg_id']));row=meta[rid];scores=ast.literal_eval(row['scp_codes'])
        y,_=pathology_targets('ptb-xl',scores)
        if json.loads(r['source_scp_codes'])!=scores or int(r['strat_fold'])!=int(row['strat_fold']):raise ValueError('Published metadata differs from official source')
        token=hashlib.sha256(('ptb-xl:'+row['patient_id']).encode()).hexdigest()
        if r['patient_token']!=token or any(int(r['target_'+name])!=int(y[j]) for j,name in enumerate(PATHOLOGY_CLASSES)):raise ValueError('Target/patient derivation differs')
        patients[r['role']].add(row['patient_id']);checked+=1
        if r['role']=='train':original_train.add(rid)
    h=read_manifest(ROOT/'outputs/reviewer_verification/arrhythmia_primary9/cedia_gold_2026-05-26/primary9_label_history_672.csv')
    historical_ids=resolve_historical_ptb_ids([str(int(r['record_id'][2:])) for r in h if r['record_id'].upper().startswith('HR')],set(meta))
    hp={meta[rid]['patient_id'] for rid in historical_ids}
    blocked=patients['validation']|patients['followup']|hp
    expected=select_extra_training(list(meta.values()),blocked,original_train)
    extra=read_manifest(BUNDLE/'data/path_extra_records.csv')
    if len(extra)!=len(expected) or {r['ecg_id'] for r in extra}!={str(int(r['ecg_id'])) for r in expected}:raise ValueError('Training expansion is not the complete predeclared set')
    extra_patients=set()
    for r in extra:
        row=meta[r['ecg_id']];scores=ast.literal_eval(row['scp_codes']);y,_=pathology_targets('ptb-xl',scores)
        if r['record_path']!=row['filename_hr'] or json.loads(r['source_scp_codes'])!=scores:raise ValueError('Additional source metadata differs')
        if any(r[field]!=index[r['record_path']+suffix] for field,suffix in [('header_sha256','.hea'),('signal_sha256','.dat')]):raise ValueError('Additional source hash differs from official index')
        if any(int(r['target_'+name])!=int(y[j]) for j,name in enumerate(PATHOLOGY_CLASSES)):raise ValueError('Additional pathology target differs')
        if int(r['rhythm_target'])!=unambiguous_rhythm_target(scores,SCP_TO_RHYTHM):raise ValueError('Additional rhythm target differs')
        if r['patient_token']!=hashlib.sha256(('ptb-xl:'+row['patient_id']).encode()).hexdigest():raise ValueError('Additional patient token differs')
        extra_patients.add(row['patient_id'])
    if extra_patients&blocked or patients['train']&blocked or patients['validation']&patients['followup']:raise ValueError('PTB-XL patient partitions overlap')
    report={'metadata_sha256':sha256(args.metadata),'scp_statements_sha256':sha256(args.scp_statements),
            'official_rhythm_codes_verified':sorted(OFFICIAL_RHYTHM_CODES),'official_index_sha256':sha256(args.source_index),
            'existing_ptbxl_targets_verified':checked,'additional_records_verified':len(extra),'official_additional_file_hashes_verified':2*len(extra),
            'historical_aliases_resolved':len(historical_ids),'extra_patients':len(extra_patients),'overlapping_comparison_patients':0,'complete_predeclared_expansion_verified':True}
    if args.output:
        if args.output.exists():raise FileExistsError('Preserve the earlier verification receipt')
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
