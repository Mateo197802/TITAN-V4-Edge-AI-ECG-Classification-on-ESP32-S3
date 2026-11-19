"""Recalculate historical-cohort gaps and training-objective conflicts."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from titan_v4.data.primary9_registry import PRIMARY9_CLASSES
from titan_v4.reproducibility.gold_gap import additional_correct, effective_rhythm_target
from titan_v4.reproducibility.inference import primary9_metrics
from titan_v4.reproducibility.refinement import read_manifest, sha256

BUNDLE = ROOT / "outputs/reviewer_verification/partial_backbone_refinement_2026-10-12"
EVIDENCE = ROOT / "outputs/reviewer_verification/gold_target_gap_2026-10-12"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    report = {"purpose":"diagnostic gap audit, not a new independent test or historical run reproduction",
              "labels_changed":False,"gradient_training_performed":False,"training_objective":{},
              "historical_cohort":{},"inputs_sha256":{}}
    feature_path = BUNDLE / "training/phase2_reference/train_features.npz"
    train = dict(np.load(feature_path,allow_pickle=False))
    report["inputs_sha256"][feature_path.relative_to(ROOT).as_posix()] = sha256(feature_path)
    report["training_records"] = len(train['y'])
    report["training_teachers"] = {
        name:{"correct":int((train[name].argmax(1)==train['y']).sum()),
              "by_class":{label:{"support":int((train['y']==i).sum()),
                                 "correct":int(((train[name].argmax(1)==i)&(train['y']==i)).sum())}
                          for i,label in enumerate(PRIMARY9_CLASSES)}} for name in ('gold','teacher')}
    for weight in (0.,.1,.5,1.,2.,5.):
        target = effective_rhythm_target(train['y'],train['gold'],train['teacher'],weight,.15)
        disagreement = target.argmax(1)!=train['y']
        report['training_objective'][str(weight)] = {
            "gold_kl_weight":weight,"teacher_kl_weight":.15,
            "per_record_optimum_disagrees_with_reference":int(disagreement.sum()),
            "by_class":{label:int((disagreement&(train['y']==i)).sum()) for i,label in enumerate(PRIMARY9_CLASSES)}}
    report['objective_interpretation'] = (
        "Exact optimum of the isolated per-record rhythm CE/KL terms, assuming independent unrestricted probabilities. "
        "The CNN shares parameters and adds auxiliary losses; counts do not predict its realized accuracy or prove a performance ceiling.")
    for role in ('train','validation'):
        rows = read_manifest(BUNDLE / f'manifests/{role}_records.csv')
        report[f'{role}_sources'] = dict(Counter(r['dataset'] for r in rows))
        report[f'{role}_ptb_rbbb_subtypes'] = dict(Counter(
            'IRBBB' if 'IRBBB' in r['source_scp_codes'] else 'CRBBB' if 'CRBBB' in r['source_scp_codes'] else 'other'
            for r in rows if r['dataset']=='ptb-xl' and r['reference_label']=='RBBB'))
    pairs = read_manifest(EVIDENCE / 'historical672_paired_predictions.csv')
    if len(pairs)!=672 or len({(r['source'],r['record_id']) for r in pairs})!=672:
        raise ValueError('Historical diagnostic must contain all 672 unique records')
    history_path = ROOT / 'outputs/reviewer_verification/arrhythmia_primary9/cedia_gold_2026-05-26/primary9_label_history_672.csv'
    history = {(r['source'].lower(),r['record_id'].lower()):r for r in read_manifest(history_path)}
    for row in pairs:
        official = history[(row['source'].lower(),row['record_id'].lower())]
        if any(row[key]!=official[key] for key in ('original_rhythm_label','final_rhythm_label')):
            raise ValueError('Diagnostic reference differs from frozen label history')
        for model in ('gold','phase3','validation_mixture'):
            p = np.array([float(row[f'{model}_prob_{c}']) for c in PRIMARY9_CLASSES])
            if (not np.isfinite(p).all() or np.any(p<0) or abs(p.sum()-1)>2e-6
                or PRIMARY9_CLASSES[p.argmax()]!=row[f'{model}_prediction']):
                raise ValueError('Recorded probabilities and predictions differ')
    for model in ('gold','phase3','validation_mixture'):
        report['historical_cohort'][model]={}
        for ref in ('original_rhythm_label','final_rhythm_label'):
            truth=[r[ref] for r in pairs];predicted=[r[f'{model}_prediction'] for r in pairs]
            metrics=primary9_metrics(truth,predicted)
            errors=Counter(t for t,p in zip(truth,predicted) if t!=p)
            report['historical_cohort'][model][ref] = {**metrics,
                "additional_net_correct_for_90pct":additional_correct(len(pairs),metrics['correct_predictions']),
                "errors_by_class":dict(errors)}
    original = read_manifest(ROOT / 'data/external_validation/record_source_manifest.csv')
    overlap={}
    for role in ('train','validation'):
        records={(r['dataset'],r['ecg_id']) for r in read_manifest(BUNDLE / f'manifests/{role}_records.csv')}
        count=sum((r['dataset'],str(int(r['record_id'][2:])) if r['dataset']=='ptb-xl' and r['record_id'].startswith('HR')
                   else r['record_id']) in records for r in original)
        overlap[role]=count
    report['exact_record_or_numeric_ptb_alias_matches_with_added_splits']=overlap
    report['overlap_scope']='Exact/alias record IDs only; not patient linkage, signal equivalence across releases or initial Gold training provenance.'
    for path in (history_path,EVIDENCE/'historical672_paired_predictions.csv'):
        report['inputs_sha256'][path.relative_to(ROOT).as_posix()]=sha256(path)
    output=EVIDENCE/'gap_audit.json'
    if args.write:
        output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    elif json.loads(output.read_text(encoding='utf-8'))!=report:
        raise ValueError('Recorded audit differs from the current evidence')
    checksum_path=EVIDENCE/'SHA256SUMS.json'
    hashes={p.relative_to(EVIDENCE).as_posix():sha256(p) for p in sorted(EVIDENCE.rglob('*'))
            if p.is_file() and p!=checksum_path and '__pycache__' not in p.parts and p.suffix!='.pyc'}
    if args.write:
        checksum_path.write_text(json.dumps(hashes,indent=2)+'\n',encoding='utf-8')
    elif json.loads(checksum_path.read_text(encoding='utf-8'))!=hashes:
        raise ValueError('Audit artifact hashes or inventory differ')
    print(json.dumps({"gold_anchor_2_conflicts":report['training_objective']['2.0']['per_record_optimum_disagrees_with_reference'],
                      "phase3_final_correct":report['historical_cohort']['phase3']['final_rhythm_label']['correct_predictions'],
                      "net_gap_to_90pct":report['historical_cohort']['phase3']['final_rhythm_label']['additional_net_correct_for_90pct']},indent=2))


if __name__ == '__main__':
    main()
