"""Recalculate validation selection, both task endpoints and original-network invariants."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
BUNDLE=ROOT/'outputs/reviewer_verification/joint_refinement_2026-10-12'
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(BUNDLE/'training'))
import contract as c
from train_joint import prefixes,protected_recalls
from titan_v4.reproducibility.refinement import read_manifest,sha256
from titan_v4.reproducibility.joint_refinement import PATHOLOGY_CLASSES,pathology_targets,calibrate_thresholds,pathology_metrics,unambiguous_rhythm_target
from titan_v4.data.label_registry import SCP_TO_RHYTHM


def key(r):return r.get('cohort',''),r['dataset'],r['ecg_id']


def recalc_validation(npz):
    p=npz['rhythm_probabilities'];c.validate_probabilities(p)
    sources=c.source_metrics(npz,p);metrics=c.metrics(npz['y'],p)
    idx=npz['path_mask'][:,0]==1;y=npz['path_targets'][idx];pp=npz['pathology_probabilities'][idx]
    thresholds=calibrate_thresholds(y,pp,role='validation')
    path=pathology_metrics(y,pp,thresholds)
    score=float(np.mean([m['macro_f1'] for m in sources.values()]))
    return {'metrics':metrics,'sources':sources,'rhythm_score':score,'pathology':path,
            'thresholds':thresholds.tolist(),'joint_score':score+.5*path['primary5_macro_f1']}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--write',action='store_true');args=parser.parse_args()
    torch.set_num_threads(2)
    protocol=json.loads((BUNDLE/'training/protocol.json').read_text())
    selection=json.loads((BUNDLE/'results/selection.json').read_text())
    input_manifest=BUNDLE/'training/INPUT_SHA256.json'
    execution=json.loads((BUNDLE/'results/execution.json').read_text())
    submission=json.loads((BUNDLE/'data/submission.json').read_text())
    if sha256(input_manifest)!=execution['input_manifest_sha256'] or execution['input_manifest_sha256']!=submission['input_manifest_sha256']:
        raise ValueError('Submitted input manifest is not byte-identical to the execution receipt')
    for name,h in json.loads(input_manifest.read_text()).items():
        published=BUNDLE/'training'/name
        if published.is_file() and sha256(published)!=h:raise ValueError('Executed training source differs: '+name)
    if sha256(BUNDLE/'training/start.pth')!=protocol['start_sha256']:raise ValueError('Warm start differs')
    if selection['references_changed'] or not selection['evaluation_labels_loaded_after_selection'] or not selection['thresholds_fitted_on_validation_only']:
        raise ValueError('Split or selection contract differs')
    targets=read_manifest(BUNDLE/'data/path_targets.csv')
    lookup={(r['role'],*key(r)):r for r in targets}
    if len(lookup)!=len(targets):raise ValueError('Duplicate pathology reference keys')
    for r in targets:
        y,mask=pathology_targets(r['dataset'],json.loads(r['source_scp_codes']))
        if int(mask[0])!=int(r['pathology_observed']) or any(int(y[j])!=int(r['target_'+name]) for j,name in enumerate(PATHOLOGY_CLASSES)):
            raise ValueError('Pathology reference differs from explicit SCP mapping')
    train_patients={r['patient_token'] for r in targets if r['role']=='train' and r['pathology_observed']=='1'}
    validation_patients={r['patient_token'] for r in targets if r['role']=='validation' and r['pathology_observed']=='1'}
    followup_patients={r['patient_token'] for r in targets if r['role']=='followup' and r['pathology_observed']=='1'}
    extra=read_manifest(BUNDLE/'data/path_extra_records.csv');ep={r['patient_token'] for r in extra}
    if len(extra)!=3083 or any(int(r['strat_fold'])>8 for r in extra) or (train_patients|ep)&(validation_patients|followup_patients):
        raise ValueError('Expanded training partition overlaps evaluation patients')
    for row in extra:
        y,_=pathology_targets('ptb-xl',json.loads(row['source_scp_codes']))
        if not y[[0,8,6,9,3]].any() or any(int(y[j])!=int(row['target_'+name]) for j,name in enumerate(PATHOLOGY_CLASSES)):
            raise ValueError('Extra record does not match training-only selection rule')
        rhythm=unambiguous_rhythm_target(json.loads(row['source_scp_codes']),SCP_TO_RHYTHM)
        if int(row['rhythm_target'])!=rhythm or row['reference_label']!=(c.CLASSES[rhythm] if rhythm>=0 else 'unknown'):
            raise ValueError('Added rhythm reference differs from official singleton SCP mapping')
    preparation=json.loads((BUNDLE/'data/preparation.json').read_text())
    if preparation['expansion']['overlapping_historical_patients']!=0 or preparation['expansion']['historical_aliases_matched']!=378:
        raise ValueError('Historical patient exclusion not established')
    base_npz=dict(np.load(BUNDLE/'results/baseline_validation.npz',allow_pickle=False));base=recalc_validation(base_npz)
    for name,value in base.items():
        if selection['baseline'][name]!=value:raise ValueError('Baseline validation metrics differ')
    original=np.load(ROOT/'outputs/reviewer_verification/partial_backbone_refinement_2026-10-12/training/phase2_reference/validation_features.npz',allow_pickle=False)
    if any(not np.array_equal(base_npz[k],original[k]) for k in ['y','dataset','ecg_id']):raise ValueError('Validation targets changed')
    candidates=[]
    policy=protocol['selection']
    aux=np.load(BUNDLE/'replay/validation_auxiliary.npz',allow_pickle=False)
    replay=json.loads((BUNDLE/'replay/validation_auxiliary_replay.json').read_text())
    if not replay['all_passed'] or replay['candidates']!=len(selection['candidates']):raise ValueError('Candidate replay incomplete')
    if any(not np.array_equal(aux[k],base_npz[k]) for k in ['dataset','ecg_id']):raise ValueError('Auxiliary replay record set differs')
    auxiliary_names=list(aux['candidate_names'])
    for candidate in selection['candidates']:
        raw=dict(np.load(BUNDLE/'results'/(candidate['name']+'_validation.npz'),allow_pickle=False))
        if any(not np.array_equal(raw[k],base_npz[k]) for k in ['y','dataset','ecg_id','path_targets','path_mask']):raise ValueError('Validation reference arrays changed')
        actual=recalc_validation(raw)
        for name,value in actual.items():
            if candidate[name]!=value:raise ValueError('Validation metric or calibration differs: '+candidate['name'])
        i=auxiliary_names.index(candidate['name'])
        cpu_errors={label:float(np.sqrt(np.mean(((aux['candidate_'+label][i]-aux['baseline_'+label])/np.maximum(aux['baseline_'+label].std(0),.05))**2)))
                    for label in ('quality','biometrics')}
        if any(abs(cpu_errors[k]-candidate['quality_biometrics_nrmse'][k])>2e-5 for k in cpu_errors):raise ValueError('Independently recalculated auxiliary preservation differs')
        checks={'protected_recalls':protected_recalls(actual['sources'],base['sources']),
                'rhythm_macro_preserved':actual['metrics']['macro_f1']>=base['metrics']['macro_f1']-.005-1e-12,
                'rhythm_sources_preserved':actual['rhythm_score']>=base['rhythm_score']-.005-1e-12,
                'quality_biometrics_preserved':max(candidate['quality_biometrics_nrmse'].values())<=.15}
        if checks!=candidate['preservation_checks'] or all(checks.values())!=candidate['eligible']:raise ValueError('Preservation decisions differ')
        candidates.append(candidate)
    eligible=[r for r in candidates if r['eligible']]
    joint=[r for r in eligible if r['trial']['path_train'] and r['pathology']['primary5_macro_f1']>=base['pathology']['primary5_macro_f1']+.02]
    rhythm=[r for r in eligible if r['rhythm_score']>base['rhythm_score']+.002]
    expected={'joint_selected':max(joint,key=lambda r:(r['joint_score'],r['name'])) if joint else None,
              'rhythm_selected':max(rhythm,key=lambda r:(r['rhythm_score'],r['name'])) if rhythm else None}
    if any(selection[name]!=value for name,value in expected.items()):raise ValueError('Validation-only selection differs')
    refit=json.loads((BUNDLE/'results/refit.json').read_text());start=c.load_model(BUNDLE/'training/start.pth');initial=start.state_dict()
    delta={}
    for name,record in expected.items():
        model=c.load_model(BUNDLE/'model'/(name+'.pth'))
        if sha256(BUNDLE/'model'/(name+'.pth'))!=selection['checkpoints'][name]:raise ValueError('Checkpoint differs')
        state=model.state_dict();allowed=prefixes(record['trial']['blocks'],record['trial']['path_train']) if record else ()
        parameters=dict(model.named_parameters());changed=[k for k in initial if not torch.equal(initial[k],state[k])]
        if set(initial)!=set(state) or any(k not in parameters or not k.startswith(allowed) or state[k].shape!=initial[k].shape for k in changed):
            raise ValueError('Frozen parameter, buffer, or architecture changed')
        if record and not refit[name].get('all_state_tensors_bitwise_equal'):raise ValueError('Refit not verified')
        with torch.inference_mode():outputs=model(torch.zeros(2,6,1250))
        if [tuple(v.shape) for v in outputs]!=[(2,9),(2,1),(2,3),(2,10)]:raise ValueError('Four-head contract changed')
        delta[name]={'changed_parameters':sum(parameters[k].numel() for k in changed),'changed_state_tensors':changed}
    manifest=read_manifest(ROOT/'outputs/reviewer_verification/partial_backbone_refinement_2026-10-12/manifests/followup_records.csv')
    frozen={key(r):r for r in manifest};recomputed={}
    for name in ['baseline','joint_selected','rhythm_selected']:
        rows=read_manifest(BUNDLE/'results'/(name+'_predictions.csv'))
        if len(rows)!=2171 or len({key(r) for r in rows})!=2171 or set(map(key,rows))!=set(frozen):raise ValueError('Followup record set changed')
        record=expected.get(name);thresholds=np.asarray((record or base)['thresholds'])
        p=[];pp=[];y=[];py=[];mask=[]
        for r in rows:
            source=frozen[key(r)];reference=lookup[('followup',*key(r))]
            if r['reference']!=source['reference_label'] or r['tensor_sha256']!=source['tensor_sha256']:raise ValueError('Followup reference/signal changed')
            prob=np.array([float(r['prob_'+v]) for v in c.CLASSES]);path=np.array([float(r['score_'+v]) for v in PATHOLOGY_CLASSES])
            if c.CLASSES[prob.argmax()]!=r['prediction']:raise ValueError('Rhythm prediction differs from probabilities')
            target,observed=pathology_targets(r['dataset'],json.loads(reference['source_scp_codes']))
            for j,label in enumerate(PATHOLOGY_CLASSES):
                if int(r['pred_'+label])!=int(path[j]>=thresholds[j]) or (observed[j] and int(r['target_'+label])!=int(target[j])):
                    raise ValueError('Pathology target or thresholded prediction differs')
            p.append(prob);pp.append(path);y.append(c.CLASSES.index(r['reference']));py.append(target);mask.append(bool(observed[0]))
        p=np.asarray(p);pp=np.asarray(pp);y=np.asarray(y);py=np.asarray(py);mask=np.asarray(mask)
        metrics={'rhythm':c.metrics(y,p),'pathology':pathology_metrics(py[mask],pp[mask],thresholds)}
        saved=json.loads((BUNDLE/'results'/(name+'_metrics.json')).read_text())
        if any(metrics[k]!=saved[k] for k in metrics) or not saved['repeat_outputs_bitwise_equal']:raise ValueError('Followup metrics or repeat differs')
        recomputed[name]=metrics
    report={'parameters':1143751,'validation':{name:(r or base)['metrics'] for name,r in expected.items()},
            'pathology_validation':{name:(r or base)['pathology'] for name,r in expected.items()},
            'followup':recomputed,'deltas':delta,'extra_training_records':3083,'labels_changed':False,'candidate_count':len(candidates)}
    history=read_manifest(ROOT/'outputs/reviewer_verification/arrhythmia_primary9/cedia_gold_2026-05-26/primary9_label_history_672.csv')
    frozen_history={(r['source'].casefold(),r['record_id'].casefold()):r for r in history}
    historical={}
    for name in ['joint_selected','rhythm_selected']:
        rows=read_manifest(BUNDLE/'historical672'/(name+'_paired_predictions.csv'))
        if len(rows)!=672 or len({(r['source'].casefold(),r['record_id'].casefold()) for r in rows})!=672:raise ValueError('Historical cohort incomplete')
        for row in rows:
            old=frozen_history[(row['source'].casefold(),row['record_id'].casefold())]
            if any(row[k]!=old[k] for k in ['original_rhythm_label','final_rhythm_label']):raise ValueError('Historical reference changed')
            p=np.array([float(row['prob_'+k]) for k in c.CLASSES])
            if not np.isfinite(p).all() or np.any(p<0) or abs(p.sum()-1)>2e-6 or c.CLASSES[p.argmax()]!=row['prediction']:raise ValueError('Historical probability/label differs')
        saved=json.loads((BUNDLE/'historical672'/(name+'_report.json')).read_text())
        if saved['checkpoint_sha256']!=selection['checkpoints'][name]:raise ValueError('Historical checkpoint differs from validation selection')
        calculated={k:c.metrics(np.asarray([c.CLASSES.index(r[k]) for r in rows]),np.asarray([[float(r['prob_'+v]) for v in c.CLASSES] for r in rows])) for k in ['original_rhythm_label','final_rhythm_label']}
        for reference,m in calculated.items():
            stored=saved['results'][reference]
            if m['correct']!=stored['correct_predictions'] or abs(m['macro_f1']-stored['macro_f1'])>1e-12:raise ValueError('Historical score differs')
        historical[name]=calculated
    report['historical672']=historical
    summary=BUNDLE/'verification_summary.json'
    if args.write:summary.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    elif json.loads(summary.read_text())!=report:raise ValueError('Verification summary differs')
    checksum=BUNDLE/'SHA256SUMS.json'
    inventory={p.relative_to(BUNDLE).as_posix():sha256(p) for p in sorted(BUNDLE.rglob('*')) if p.is_file() and p!=checksum and '__pycache__' not in p.parts and p.suffix!='.pyc'}
    if args.write:checksum.write_text(json.dumps(inventory,indent=2)+'\n',encoding='utf-8')
    elif json.loads(checksum.read_text())!=inventory:raise ValueError('Bundle file hashes differ')
    print(json.dumps({'verified_artifacts':len(inventory),'models':{name:{'rhythm_correct':m['rhythm']['correct'],
          'rhythm_accuracy':m['rhythm']['accuracy'],'rhythm_macro_f1':m['rhythm']['macro_f1'],
          'pathology_mean_label_accuracy':m['pathology']['primary5_mean_label_accuracy'],
          'pathology_macro_f1':m['pathology']['primary5_macro_f1']} for name,m in recomputed.items()}},indent=2))


if __name__=='__main__':main()
