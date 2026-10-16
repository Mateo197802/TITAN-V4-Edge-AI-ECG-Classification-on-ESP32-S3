"""Infer on all 672 historical signals, then score both unchanged reference columns."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from titan_v4.data.primary9_registry import PRIMARY9_CLASSES
from titan_v4.reproducibility.inference import primary9_metrics
from titan_v4.reproducibility.refinement import read_manifest,sha256

BUNDLE=ROOT/'outputs/reviewer_verification/joint_refinement_2026-10-12'
HISTORY=ROOT/'outputs/reviewer_verification/arrhythmia_primary9/cedia_gold_2026-05-26/primary9_label_history_672.csv'
PREVIOUS=ROOT/'outputs/reviewer_verification/loss_anchor_ablation_2026-10-12/historical672/paired_predictions.csv'


def pair_rows(predictions,history,previous):
    def key(r,source='source'):return r[source].casefold(),r['record_id'].casefold()
    frozen={key(r):r for r in history};old={key(r):r for r in previous}
    if not predictions or len(frozen)!=len(history) or len(old)!=len(previous) or len(predictions)!=len(history):raise ValueError('Missing or duplicate records')
    if len({key(r,'source_label') for r in predictions})!=len(predictions) or set(key(r,'source_label') for r in predictions)!=set(frozen) or set(old)!=set(frozen):
        raise ValueError('Record sets differ')
    result=[]
    for r in predictions:
        k=key(r,'source_label');ref=frozen[k]
        p=np.array([float(r['prob_'+label]) for label in PRIMARY9_CLASSES])
        if not np.isfinite(p).all() or np.any(p<0) or abs(p.sum()-1)>2e-6 or PRIMARY9_CLASSES[p.argmax()]!=r['predicted_label']:
            raise ValueError('Invalid inferred probabilities')
        row={name:ref[name] for name in ['source','record_id','original_rhythm_label','final_rhythm_label']}
        row.update({'prediction':r['predicted_label'],'previous_prediction':old[k]['prediction']})
        row.update({'prob_'+label:float(p[j]) for j,label in enumerate(PRIMARY9_CLASSES)})
        result.append(row)
    return result


def main():
    global BUNDLE
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',type=Path,default=BUNDLE)
    parser.add_argument('--model',choices=['joint_selected','rhythm_selected'],default='rhythm_selected')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--offline',action='store_true')
    args=parser.parse_args();BUNDLE=args.bundle
    if args.output.exists():raise FileExistsError('Use a fresh output directory')
    checkpoint=BUNDLE/'model'/(args.model+'.pth')
    selection=json.loads((BUNDLE/'results/selection.json').read_text())
    if sha256(checkpoint)!=selection['checkpoints'][args.model]:raise ValueError('Selected checkpoint hash differs')
    command=[sys.executable,str(ROOT/'scripts/run_primary9_inference.py'),'--checkpoint',str(checkpoint),'--out-dir',str(args.output/'signal_run')]
    if args.offline:command.append('--offline')
    subprocess.run(command,check=True,cwd=ROOT)
    rows=pair_rows(read_manifest(args.output/'signal_run/primary9_record_predictions.csv'),read_manifest(HISTORY),read_manifest(PREVIOUS))
    if len(rows)!=672:raise ValueError('Historical cohort must contain all 672 records')
    with (args.output/'paired_predictions.csv').open('x',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    scores={name:primary9_metrics([r[name] for r in rows],[r['prediction'] for r in rows]) for name in ['original_rhythm_label','final_rhythm_label']}
    run=json.loads((args.output/'signal_run/run_manifest.json').read_text())
    report={'model':args.model,'checkpoint_sha256':sha256(checkpoint),'records':len(rows),
            'reference_file_sha256':sha256(HISTORY),'reference_labels_changed':False,'results':scores,
            'original_signal_run_manifest_sha256':sha256(args.output/'signal_run/run_manifest.json'),
            'source_file_hashes_sha256':sha256(args.output/'signal_run/source_files_sha256.csv'),
            'inference_code_sha256':{name:sha256(ROOT/name) for name in ['scripts/run_joint_historical.py',*run['code_files']]},
            'environment':run['runtime'],'preprocessing':run['preprocessing'],'source_dataset':run['source_dataset'],
            'interpretation':'New research-model inference, evaluated after validation selection; not a reproduction of the original Gold execution'}
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({name:{k:v for k,v in metric.items() if k in ['correct_predictions','accuracy','macro_f1']} for name,metric in scores.items()},indent=2))


if __name__=='__main__':main()
