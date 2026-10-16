"""Authenticate head controls, patient-fold selection, exact state composition and replays."""
import argparse
import ast
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score

from verify_joint_evidence import ROOT, c, recalc_validation, read_manifest, sha256, pathology_targets, PATHOLOGY_CLASSES, pathology_metrics
from titan_v4.reproducibility.supervision_control import source_masses, fixed_source_weights, confidence_weights, crossfit_pathology, verify_fold_alignment, select_components, assert_role_frozen
from titan_v4.reproducibility.head_composition import compose_head_states
from titan_v4.reproducibility.joint_refinement import SCP_MAPPING, PRIMARY5_INDEX


def same(a,b):
    if isinstance(a,dict): return isinstance(b,dict) and set(a)==set(b) and all(same(a[k],b[k]) for k in a)
    if isinstance(a,list): return isinstance(b,list) and len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
    if isinstance(a,float): return isinstance(b,(int,float)) and abs(a-b)<=1e-12
    return a==b


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',type=Path,required=True); parser.add_argument('--write',action='store_true')
    args=parser.parse_args(); bundle=args.bundle; data=bundle/'data'; out=bundle/'results'; torch.set_num_threads(2)
    prior=ROOT/'outputs/reviewer_verification/directed_pcgrad_2026-10-13'
    older=json.loads((prior/'SHA256SUMS.json').read_text())
    for name in ('ptbxl_database.csv','scp_statements.csv','official_SHA256SUMS.txt','directed_official_SHA256SUMS.txt',
                 'ConditionNames_SNOMED-CT.csv','path_extra_records.csv','path_validation_records.csv','rhythm_extra_records.csv',
                 'header_census.csv','path_targets.csv','path_targets.npz','preparation.json','directed_preparation.json','source_rebuild_receipt.json'):
        if sha256(data/name)!=older['data/'+name]:raise ValueError('Inherited source population differs: '+name)
    original=ROOT/'outputs/reviewer_verification/partial_backbone_refinement_2026-10-12'
    train=read_manifest(original/'manifests/train_records.csv'); extra=read_manifest(data/'path_extra_records.csv'); directed=read_manifest(data/'rhythm_extra_records.csv')
    records=train+extra+directed; meta={str(int(r['ecg_id'])):r for r in read_manifest(data/'ptbxl_database.csv')}
    controls=dict(np.load(data/'supervision_weights.npz',allow_pickle=False))
    y=np.asarray([int(r['rhythm_target']) if 'rhythm_target' in r else c.CLASSES.index(r['reference_label']) for r in records],np.int64)
    sources=np.asarray([r['dataset'] for r in records]); ids=np.asarray([r['ecg_id'] for r in records])
    if not all(np.array_equal(controls[k],v) for k,v in [('training_y',y),('training_dataset',sources),('training_ecg_id',ids)]):raise ValueError('Training control row order differs')
    masses=source_masses(y[:-190],sources[:-190],role='train')
    if not np.array_equal(controls['rhythm_weights'],fixed_source_weights(y,sources,masses,role='train')):raise ValueError('Frozen CE source budget differs')
    targets=[]; masks=[]; human=[]; certainty=[]
    for row in records:
        native=meta[str(int(row['ecg_id']))] if row['dataset']=='ptb-xl' else None
        scores=ast.literal_eval(native['scp_codes']) if native else {}
        yy,mask=pathology_targets(row['dataset'],scores); l=np.full(10,100,np.float32)
        for code,value in scores.items():
            if code in SCP_MAPPING:l[SCP_MAPPING[code]]=value
        targets.append(yy); masks.append(mask); certainty.append(l); human.append(bool(native and native['validated_by_human']=='True'))
    hw,cw=confidence_weights(np.stack(targets),np.stack(masks),np.asarray(human),np.stack(certainty),role='train')
    if not np.array_equal(hw,controls['human_weights']) or not np.array_equal(cw,controls['certainty_weights']):raise ValueError('Metadata-derived supervision weights differ')
    audit=json.loads((data/'audit.json').read_text())
    if audit['reference_labels_changed'] or audit['evaluation_data_used_for_weights'] or audit['training_records']!=22567:raise ValueError('Control audit scope differs')
    for name,digest in audit['files'].items():
        if sha256(data/name)!=digest:raise ValueError('Prepared control receipt differs')
    protocol=json.loads((bundle/'training/protocol.json').read_text()); inputs=json.loads((bundle/'training/INPUT_SHA256.json').read_text())
    execution=json.loads((out/'execution.json').read_text()); submission=json.loads((data/'submission.json').read_text())
    if sha256(bundle/'training/INPUT_SHA256.json')!=execution['input_manifest_sha256'] or execution['input_manifest_sha256']!=submission['input_manifest_sha256']:raise ValueError('Submitted/executed input manifest differs')
    for name,digest in inputs.items():
        if (bundle/'training'/name).is_file() and sha256(bundle/'training'/name)!=digest:raise ValueError('Executed source differs: '+name)
    if inputs['supervision_weights.npz']!=sha256(data/'supervision_weights.npz') or inputs['supervision_weights.npz']!=sha256(bundle/'training/supervision_weights.npz'):raise ValueError('Executed controls differ')
    prepared=json.loads((data/'preparation.json').read_text()); directed_prep=json.loads((data/'directed_preparation.json').read_text())
    for name in ('path_targets.npz','path_extra.npz','path_validation.npz'):
        if inputs[name]!=prepared['files'][name]:raise ValueError('Executed native input differs')
    if inputs['rhythm_extra.npz']!=directed_prep['files']['rhythm_extra.npz']:raise ValueError('Executed directed input differs')
    if sha256(bundle/'training/start.pth')!=protocol['start_sha256'] or execution['parameters']!=1143751 or not execution['features_reproduce_raw_outputs_bitwise']:raise ValueError('Architecture/features contract differs')
    selection=json.loads((out/'selection.json').read_text()); inventory=json.loads((out/'SHA256SUMS.json').read_text())
    for name,digest in inventory.items():
        if Path(name).name!=name or sha256(out/name)!=digest:raise ValueError('Sealed result differs')
    pval=read_manifest(data/'path_validation_records.csv')
    py=np.asarray([[int(r['target_'+label]) for label in PATHOLOGY_CLASSES] for r in pval],np.float32)
    reference=np.load(original/'training/phase2_reference/validation_features.npz',allow_pickle=False)
    baseline_raw=dict(np.load(out/'baseline_validation.npz',allow_pickle=False)); initial=c.load_model(bundle/'training/start.pth').state_dict()

    def recalc(raw):
        if not all(np.array_equal(raw[k],reference[k]) for k in ('y','dataset','ecg_id')):raise ValueError('Rhythm validation reference differs')
        if not np.array_equal(raw['path_targets'],py) or not (raw['path_mask']==1).all():raise ValueError('Pathology validation reference differs')
        verify_fold_alignment(raw['path_patient_fold'],raw['path_ecg_id'],pval)
        if not np.array_equal(raw['path_patient_fold'],controls['path_validation_folds']):raise ValueError('Executed fold vector differs')
        report=recalc_validation(raw)
        report['pathology_oof']=crossfit_pathology(py,raw['pathology_probabilities'],raw['path_patient_fold'],role='validation')
        report['pathology_ap']=float(np.mean([average_precision_score(py[:,j],raw['pathology_probabilities'][:,j]) for j in PRIMARY5_INDEX]))
        return report

    base=recalc(baseline_raw)
    if any(not same(v,selection['baseline'][k]) for k,v in base.items()):raise ValueError('Baseline validation score differs')
    fitted=[]; composed=None
    for row in selection['candidates']:
        raw=dict(np.load(out/(row['name']+'_validation.npz'),allow_pickle=False)); actual=recalc(raw)
        if any(not same(v,row[k]) for k,v in actual.items()):raise ValueError('Candidate validation/OOF score differs: '+row['name'])
        if not np.array_equal(raw['quality'],baseline_raw['quality']) or not np.array_equal(raw['biometrics'],baseline_raw['biometrics']):raise ValueError('Frozen auxiliary predictions changed')
        checks={'protected_recalls':all(v['recall'][label]>=base['sources'][s]['recall'][label]-.05-1e-12 for s,v in actual['sources'].items() for label in protocol['selection']['protected_classes'] if v['support'][label]>=20),
                'rhythm_macro_preserved':actual['metrics']['macro_f1']>=base['metrics']['macro_f1']-.005-1e-12,
                'rhythm_sources_preserved':actual['rhythm_score']>=base['rhythm_score']-.005-1e-12,'quality_biometrics_preserved':True}
        if row['preservation_checks']!=checks or row['eligible']!=all(checks.values()):raise ValueError('Preservation screening differs')
        model=c.load_model(out/(row['name']+'.pth')); assert_role_frozen(model,initial,'both' if row.get('kind')=='composition' else row['trial']['task'])
        if row.get('kind')=='composition':composed=row; continue
        if row['trial'] not in protocol['optimization']['trials']:raise ValueError('Trial differs from frozen protocol')
        unchanged='pathology_probabilities' if row['trial']['task']=='rhythm' else 'rhythm_probabilities'
        if not np.array_equal(raw[unchanged],baseline_raw[unchanged]):raise ValueError('Independent head fit changed other task predictions')
        fitted.append(row)
    if {(r['trial']['name'],r['epoch']) for r in fitted}!={(r['name'],epoch) for r in protocol['optimization']['trials'] for epoch in protocol['optimization']['checkpoints']} or composed is None:raise ValueError('Trial/composition coverage differs')
    chosen=select_components(fitted,base,role='validation')
    if any(not same(record,selection[key]) for key,record in chosen.items()):raise ValueError('Independent component selection differs')
    component_states={}; refits=json.loads((out/'refit.json').read_text())
    for key,row in chosen.items():
        component_states[key]=c.load_model(out/(row['name']+'.pth')).state_dict() if row else initial
        if row and (not refits[key].get('all_state_tensors_bitwise_equal') or refits[key]['candidate']!=row['name']):raise ValueError('Selected component refit differs')
    combined=compose_head_states(component_states['rhythm_component'],component_states['pathology_component'])
    for role,expected in [('joint_selected',combined),('rhythm_selected',component_states['rhythm_component'])]:
        checkpoint=bundle/'model'/(role+'.pth')
        if sha256(checkpoint)!=selection['checkpoints'][role] or sha256(checkpoint)!=sha256(out/(role+'.pth')):raise ValueError('Published model hash differs')
        if any(not torch.equal(v,expected[k]) for k,v in c.load_model(checkpoint).state_dict().items()):raise ValueError('Published head composition differs')
    if not same(composed,selection['joint_selected']) or not same(chosen['rhythm_component'],selection['rhythm_selected']):raise ValueError('Final-role selection differs')
    if selection['references_changed'] or not selection['evaluation_labels_loaded_after_selection'] or not selection['thresholds_fitted_on_validation_only']:raise ValueError('Selection boundary changed')
    follow=read_manifest(original/'manifests/followup_records.csv'); native_targets={ (r['dataset'],r['ecg_id']):r for r in read_manifest(data/'path_targets.csv') if r['role']=='followup'}
    calculated={}
    for name in ('baseline','joint_selected','rhythm_selected'):
        rows=read_manifest(out/(name+'_predictions.csv'))
        if [(r['cohort'],r['dataset'],r['ecg_id']) for r in rows]!=[(r['cohort'],r['dataset'],r['ecg_id']) for r in follow]:raise ValueError('Followup population differs')
        if any(r['reference']!=f['reference_label'] or r['tensor_sha256']!=f['tensor_sha256'] for r,f in zip(rows,follow)):raise ValueError('Followup references differ')
        pp=[]; yy=[]
        for row in rows:
            reference_row=native_targets[(row['dataset'],row['ecg_id'])]
            if row['pathology_observed']!=reference_row['pathology_observed']:raise ValueError('Unknown pathology mask differs')
            if row['pathology_observed']=='1':
                yrow=[int(row['target_'+label]) for label in PATHOLOGY_CLASSES]
                if yrow!=[int(reference_row['target_'+label]) for label in PATHOLOGY_CLASSES]:raise ValueError('Pathology reference changed')
                yy.append(yrow); pp.append([float(row['score_'+label]) for label in PATHOLOGY_CLASSES])
        selected=selection.get(name) if name!='baseline' else None; thresholds=np.asarray((selected or base)['thresholds'])
        probabilities=np.asarray([[float(r['prob_'+label]) for label in c.CLASSES] for r in rows])
        if any(r['prediction']!=c.CLASSES[int(p.argmax())] for r,p in zip(rows,probabilities)):raise ValueError('Rhythm decision differs')
        if any(int(r['pred_'+label])!=int(float(r['score_'+label])>=thresholds[j]) for r in rows for j,label in enumerate(PATHOLOGY_CLASSES)):raise ValueError('Pathology threshold decision differs')
        actual={'rhythm':c.metrics(np.asarray([c.CLASSES.index(r['reference']) for r in rows]),probabilities),'pathology':pathology_metrics(np.asarray(yy),np.asarray(pp),thresholds)}
        saved=json.loads((out/(name+'_metrics.json')).read_text())
        if any(not same(v,saved[k]) for k,v in actual.items()):raise ValueError('Followup score differs')
        calculated[name]=actual
    replay=json.loads((bundle/'replay/validation_source_replay.json').read_text())
    if not replay['all_passed'] or replay['checkpoints']!=len(fitted)+2:raise ValueError('Candidate source replay incomplete')
    historical_reference=read_manifest(ROOT/'outputs/reviewer_verification/arrhythmia_primary9/cedia_gold_2026-05-26/primary9_label_history_672.csv')
    history_lookup={(r['source'],r['record_id']):r for r in historical_reference}
    for role in ('joint_selected','rhythm_selected'):
        r=json.loads((bundle/'replay'/(role+'_cpu_signal_replay.json')).read_text())
        if not r['passed'] or r['records']!=2171 or r['checkpoint_sha256']!=selection['checkpoints'][role]:raise ValueError('Final source replay incomplete')
        history=json.loads((bundle/'historical672'/(role+'_report.json')).read_text())
        rows=read_manifest(bundle/'historical672'/(role+'_paired_predictions.csv'))
        if history['model']!=role or history['records']!=672 or len(rows)!=672 or history['checkpoint_sha256']!=selection['checkpoints'][role]:raise ValueError('Historical source run incomplete')
        signal_run=json.loads((bundle/'historical672'/(role+'_run_manifest.json')).read_text())
        if signal_run['model']['sha256'].lower()!=selection['checkpoints'][role] or signal_run['record_count']!=672:raise ValueError('Historical executed checkpoint differs')
        if len(history_lookup)!=672 or {(r['source'],r['record_id']) for r in rows}!=set(history_lookup):raise ValueError('Historical population differs')
        probabilities=np.asarray([[float(r['prob_'+label]) for label in c.CLASSES] for r in rows])
        if not np.isfinite(probabilities).all() or (probabilities<0).any() or not np.allclose(probabilities.sum(1),1,atol=2e-6,rtol=0):raise ValueError('Invalid historical probabilities')
        if any(r['prediction']!=c.CLASSES[int(p.argmax())] for r,p in zip(rows,probabilities)):raise ValueError('Historical prediction differs')
        for column in ('original_rhythm_label','final_rhythm_label'):
            if any(r[column]!=history_lookup[(r['source'],r['record_id'])][column] for r in rows):raise ValueError('Historical reference changed')
            metrics=c.metrics(np.asarray([c.CLASSES.index(r[column]) for r in rows]),probabilities)
            saved=history['results'][column]
            if metrics['correct']!=saved['correct_predictions'] or abs(metrics['accuracy']-saved['accuracy'])>1e-12 or abs(metrics['macro_f1']-saved['macro_f1'])>1e-12:raise ValueError('Historical metric differs')
        if history['reference_labels_changed'] or history['reference_file_sha256']!=sha256(ROOT/'outputs/reviewer_verification/arrhythmia_primary9/cedia_gold_2026-05-26/primary9_label_history_672.csv'):raise ValueError('Historical reference receipt differs')
        for suffix,receipt_key in [('run_manifest.json','original_signal_run_manifest_sha256'),('source_files_sha256.csv','source_file_hashes_sha256')]:
            if sha256(bundle/'historical672'/(role+'_'+suffix))!=history[receipt_key]:raise ValueError('Historical signal receipt differs')
    summary={'inherited_source_hashes_verified':True,'training_controls_recomputed':True,'canonical_patient_folds_verified':True,
             'fitted_candidates':len(fitted),'shared_extractor_and_auxiliary_tensors_unchanged':True,'models':calculated}
    target=bundle/'verification_summary.json'
    if args.write:target.write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    elif not same(summary,json.loads(target.read_text())):raise ValueError('Verification summary differs')
    sums=bundle/'SHA256SUMS.json'; hashes={p.relative_to(bundle).as_posix():sha256(p) for p in sorted(bundle.rglob('*')) if p.is_file() and p!=sums and '__pycache__' not in p.parts and p.suffix!='.pyc'}
    if args.write:sums.write_text(json.dumps(hashes,indent=2)+'\n',encoding='utf-8')
    elif hashes!=json.loads(sums.read_text()):raise ValueError('Bundle byte hashes differ')
    print(json.dumps({'artifacts':len(hashes),'selected_components':{k:r['name'] if r else 'start' for k,r in chosen.items()},
                      'models':{n:{'accuracy':v['rhythm']['accuracy'],'macro_f1':v['rhythm']['macro_f1'],'pathology_accuracy':v['pathology']['primary5_mean_label_accuracy'],'pathology_f1':v['pathology']['primary5_macro_f1']} for n,v in calculated.items()}},indent=2))


if __name__=='__main__':main()
