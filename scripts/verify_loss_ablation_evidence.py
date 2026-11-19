"""Verify the CEDIA loss ablation and recalculate both frozen reference scores."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from titan_v4.data.primary9_registry import PRIMARY9_CLASSES
from titan_v4.reproducibility.inference import load_primary9_model,primary9_metrics
from titan_v4.reproducibility.refinement import read_manifest,sha256
from titan_v4.reproducibility.gold_gap import additional_correct

BUNDLE=ROOT/'outputs/reviewer_verification/loss_anchor_ablation_2026-10-12'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write',action='store_true')
    args=parser.parse_args()
    start=load_primary9_model(BUNDLE/'training/start.pth');selected=load_primary9_model(BUNDLE/'model/selected.pth')
    if sum(p.numel() for p in selected.parameters())!=1143751:raise ValueError('Parameter contract changed')
    a,b=start.state_dict(),selected.state_dict()
    if set(a)!=set(b) or any(a[n].shape!=b[n].shape for n in a):raise ValueError('Architecture changed')
    changed=[n for n in a if not torch.equal(a[n],b[n])]
    parameters=dict(selected.named_parameters())
    if any(n not in parameters or not n.startswith(('res_layers.3.','res_layers.4.','head_rhythm.')) for n in changed):
        raise ValueError('Frozen parameter or buffer changed')
    with torch.inference_mode():outputs=selected(torch.zeros(2,6,1250))
    if [tuple(v.shape) for v in outputs]!=[(2,9),(2,1),(2,3),(2,10)]:raise ValueError('Four-head contract changed')
    selection=json.loads((BUNDLE/'results/selection.json').read_text())
    if sha256(BUNDLE/'model/selected.pth')!=selection['selected_checkpoint_sha256']:raise ValueError('Selected checkpoint hash differs')
    protocol=json.loads((BUNDLE/'training/protocol.json').read_text())
    if sha256(BUNDLE/'training/start.pth')!=protocol['start_sha256']:raise ValueError('Warm start differs')
    refit=json.loads((BUNDLE/'results/refit.json').read_text())
    if not refit['all_state_tensors_bitwise_equal']:raise ValueError('Refit not verified')
    candidates=selection['candidates']
    eligible=[c for c in candidates if c['eligible'] and c['score']>selection['starting_score']+protocol['selection']['minimum_gain']]
    if not eligible or max(eligible,key=lambda c:c['score'])['name']!=selection['selected']:
        raise ValueError('Checkpoint does not match the predeclared validation selector')
    selected_validation=None
    parent=ROOT/'outputs/reviewer_verification/partial_backbone_refinement_2026-10-12'
    validation_reference=np.load(parent/'training/phase2_reference/validation_features.npz',allow_pickle=False)
    start_probabilities=np.load(parent/'validation_candidates/partial_lr1e-05_anchor2.0_focus1.25_e08_validation.npz',allow_pickle=False)['probabilities']
    for candidate in candidates:
        data=np.load(BUNDLE/'results'/(candidate['name']+'_validation.npz'),allow_pickle=False)
        if any(not np.array_equal(data[k],validation_reference[k]) for k in ('y','dataset','ecg_id')):
            raise ValueError('Validation targets or records changed')
        p=data['probabilities']
        if p.shape!=(1476,9) or not np.isfinite(p).all() or np.any(p<0) or not np.allclose(p.sum(1),1,atol=2e-6,rtol=0):
            raise ValueError('Validation probabilities invalid')
        labels=[PRIMARY9_CLASSES[int(i)] for i in data['y']]
        predicted=[PRIMARY9_CLASSES[int(i)] for i in data['probabilities'].argmax(1)]
        metrics=primary9_metrics(labels,predicted)
        if metrics['correct_predictions']!=candidate['metrics']['correct'] or abs(metrics['macro_f1']-candidate['metrics']['macro_f1'])>1e-12:
            raise ValueError('Validation metrics differ: '+candidate['name'])
        scores=[];gold_preserved=True;start_preserved=True
        for source in np.unique(data['dataset']):
            mask=data['dataset']==source
            truth=[PRIMARY9_CLASSES[int(i)] for i in data['y'][mask]]
            score=primary9_metrics(truth,[PRIMARY9_CLASSES[int(i)] for i in p[mask].argmax(1)])
            old=primary9_metrics(truth,[PRIMARY9_CLASSES[int(i)] for i in start_probabilities[mask].argmax(1)])
            gold=primary9_metrics(truth,[PRIMARY9_CLASSES[int(i)] for i in validation_reference['gold'][mask].argmax(1)])
            scores.append(score['macro_f1'])
            for label in ('SB','LBBB'):
                r=score['classification_report'][label]
                if r['support']>0:gold_preserved &= r['recall']>=gold['classification_report'][label]['recall']-.05-1e-12
            for label in protocol['selection']['protected_classes']:
                r=score['classification_report'][label]
                if r['support']>=protocol['selection']['minimum_class_support']:
                    start_preserved &= r['recall']>=old['classification_report'][label]['recall']-protocol['selection']['maximum_recall_drop']-1e-12
        eligible=bool(gold_preserved and start_preserved and max(candidate['auxiliary_nrmse'].values())<=protocol['selection']['maximum_auxiliary_nrmse'])
        if abs(float(np.mean(scores))-candidate['score'])>1e-12 or eligible!=candidate['eligible']:
            raise ValueError('Validation score or preservation criteria differ')
        if candidate['name']==selection['selected']:selected_validation=metrics
    rows=read_manifest(BUNDLE/'results/followup_predictions.csv')
    previous=read_manifest(ROOT/'outputs/reviewer_verification/partial_backbone_refinement_2026-10-12/evaluation/followup_predictions.csv')
    key=lambda r:(r['cohort'],r['dataset'],r['ecg_id'])
    prior={key(r):r for r in previous}
    if len(rows)!=2171 or len({key(r) for r in rows})!=2171 or set(prior)!=set(key(r) for r in rows):
        raise ValueError('Followup record set differs')
    current_correct=previous_correct=0;cohorts={}
    saved=json.loads((BUNDLE/'results/followup_metrics.json').read_text())
    for r in rows:
        old=prior[key(r)]
        if r['reference']!=old['reference'] or r['tensor_sha256']!=old['tensor_sha256']:
            raise ValueError('Followup target or signal tensor changed')
        p=np.array([float(r['prob_'+c]) for c in PRIMARY9_CLASSES])
        if not np.isfinite(p).all() or np.any(p<0) or abs(p.sum()-1)>2e-6 or PRIMARY9_CLASSES[p.argmax()]!=r['prediction']:
            raise ValueError('Probability/label mismatch')
        current_correct+=r['prediction']==r['reference'];previous_correct+=old['phase3_prediction']==r['reference']
    for cohort in sorted({r['cohort'] for r in rows}):
        subset=[r for r in rows if r['cohort']==cohort]
        metrics=primary9_metrics([r['reference'] for r in subset],[r['prediction'] for r in subset])
        if metrics['confusion_matrix']!=saved['cohorts'][cohort]['confusion_matrix']:raise ValueError('Followup matrix differs')
        cohorts[cohort]=metrics
    history=read_manifest(BUNDLE/'historical672/paired_predictions.csv')
    frozen={(r['source'].lower(),r['record_id'].lower()):r for r in read_manifest(
        ROOT/'outputs/reviewer_verification/arrhythmia_primary9/cedia_gold_2026-05-26/primary9_label_history_672.csv')}
    if len(history)!=672 or len({(r['source'],r['record_id']) for r in history})!=672:raise ValueError('Historical record set differs')
    for row in history:
        original=frozen[(row['source'].lower(),row['record_id'].lower())]
        if any(row[n]!=original[n] for n in ('original_rhythm_label','final_rhythm_label')):raise ValueError('Historical reference changed')
        p=np.array([float(row['prob_'+c]) for c in PRIMARY9_CLASSES])
        if not np.isfinite(p).all() or np.any(p<0) or abs(p.sum()-1)>2e-6 or PRIMARY9_CLASSES[p.argmax()]!=row['prediction']:
            raise ValueError('Historical probability/label mismatch')
    paired={}
    for name in ('original_rhythm_label','final_rhythm_label'):
        labels=[r[name] for r in history]
        new=primary9_metrics(labels,[r['prediction'] for r in history]);old=primary9_metrics(labels,[r['previous_prediction'] for r in history])
        paired[name]={'current':new,'previous':old,'net_correct_change':new['correct_predictions']-old['correct_predictions'],
                      'additional_net_correct_for_90pct':additional_correct(672,new['correct_predictions'])}
    report={'parameters':1143751,'changed_parameters':sum(parameters[n].numel() for n in changed),'changed_state_tensors':changed,
            'validation':selected_validation,'followup':{'records':len(rows),'correct':current_correct,'previous_correct':previous_correct,
            'net_correct_change':current_correct-previous_correct,'accuracy':current_correct/len(rows),'cohorts':cohorts},'historical672':paired,
            'labels_changed':False,'fallback_model_preserved':True}
    out=BUNDLE/'results/verification_summary.json'
    if args.write:out.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    elif json.loads(out.read_text(encoding='utf-8'))!=report:raise ValueError('Verification summary differs')
    checksum=BUNDLE/'SHA256SUMS.json'
    inventory={p.relative_to(BUNDLE).as_posix():sha256(p) for p in sorted(BUNDLE.rglob('*')) if p.is_file() and p!=checksum
               and '__pycache__' not in p.parts and p.suffix!='.pyc'}
    if args.write:checksum.write_text(json.dumps(inventory,indent=2)+'\n',encoding='utf-8')
    elif json.loads(checksum.read_text(encoding='utf-8'))!=inventory:raise ValueError('Bundle hash inventory differs')
    print(json.dumps({'selected':selection['selected'],'validation_correct':selected_validation['correct_predictions'],
                      'followup_correct':current_correct,'followup_net_change':current_correct-previous_correct,
                      'historical_original_correct':paired['original_rhythm_label']['current']['correct_predictions'],
                      'historical_final_correct':paired['final_rhythm_label']['current']['correct_predictions'],
                      'verified_artifacts':len(inventory)},indent=2))


if __name__=='__main__':main()
