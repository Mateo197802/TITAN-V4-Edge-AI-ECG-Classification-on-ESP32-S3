"""Independently replay all candidate outputs and verify auxiliary preservation on CPU."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import platform
import sys

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from titan_v4.reproducibility.refinement import read_manifest,load_tensor,sha256
from titan_v4.reproducibility.inference import load_primary9_model

BUNDLE=ROOT/'outputs/reviewer_verification/joint_refinement_2026-10-12'


def predict(model,x):
    buckets=[[],[],[],[]]
    with torch.inference_mode():
        for i in range(0,len(x),128):
            for bucket,value in zip(buckets,model(torch.from_numpy(x[i:i+128]))):bucket.append(value.numpy())
    return tuple(np.concatenate(parts) for parts in buckets)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoints-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--data-root',type=Path,default=ROOT/'data/cache/physionet')
    parser.add_argument('--offline',action='store_true')
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Use a new output directory')
    selection=json.loads((BUNDLE/'results/selection.json').read_text());inventory=json.loads((BUNDLE/'results/SHA256SUMS.json').read_text())
    manifest=ROOT/'outputs/reviewer_verification/partial_backbone_refinement_2026-10-12/manifests/validation_records.csv'
    rows=read_manifest(manifest)
    with ThreadPoolExecutor(max_workers=4) as pool:x=np.stack(list(pool.map(lambda r:load_tensor(r,args.data_root,download=not args.offline),rows)))
    torch.set_num_threads(2)
    baseline=predict(load_primary9_model(BUNDLE/'training/start.pth'),x)
    qualities=[];bios=[];names=[];checks=[]
    for candidate in selection['candidates']:
        name=candidate['name'];checkpoint=args.checkpoints_dir/(name+'.pth')
        if sha256(checkpoint)!=inventory[name+'.pth']:raise ValueError('Original checkpoint hash differs')
        model=load_primary9_model(checkpoint)
        if sum(p.numel() for p in model.parameters())!=1143751:raise ValueError('Parameter count differs')
        values=predict(model,x)
        saved=np.load(BUNDLE/'results'/(name+'_validation.npz'),allow_pickle=False)
        rp=torch.softmax(torch.from_numpy(values[0]),1).numpy();pp=torch.sigmoid(torch.from_numpy(values[3])).numpy()
        maximum=float(max(np.max(np.abs(rp-saved['rhythm_probabilities'])),np.max(np.abs(pp-saved['pathology_probabilities']))))
        if maximum>2e-5 or not np.array_equal(rp.argmax(1),saved['rhythm_probabilities'].argmax(1)):raise ValueError('Candidate output replay differs')
        errors={label:float(np.sqrt(np.mean(((v-t)/np.maximum(t.std(0),.05))**2)))
                for label,v,t in zip(('quality','biometrics'),values[1:3],baseline[1:3])}
        error_difference=max(abs(errors[k]-candidate['quality_biometrics_nrmse'][k]) for k in errors)
        if error_difference>2e-5:raise ValueError('Auxiliary preservation replay differs')
        qualities.append(values[1]);bios.append(values[2]);names.append(name)
        checks.append({'candidate':name,'checkpoint_sha256':sha256(checkpoint),'max_probability_difference':maximum,
                       'cpu_nrmse':errors,'max_nrmse_difference':error_difference,'passed':True})
        print(name+' replay passed',flush=True)
    args.output.mkdir(parents=True)
    np.savez_compressed(args.output/'validation_auxiliary.npz',candidate_names=np.asarray(names),
                        candidate_quality=np.stack(qualities),candidate_biometrics=np.stack(bios),
                        baseline_quality=baseline[1],baseline_biometrics=baseline[2],
                        dataset=np.asarray([r['dataset'] for r in rows]),ecg_id=np.asarray([r['ecg_id'] for r in rows]))
    report={'records':len(rows),'candidates':len(names),'verified_original_source_tensors':len(rows),'checks':checks,
            'all_passed':all(c['passed'] for c in checks),'tolerance':2e-5,
            'environment':{'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,'device':'cpu'}}
    (args.output/'validation_auxiliary_replay.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'records':len(rows),'candidates':len(names),'all_passed':report['all_passed']}))


if __name__=='__main__':main()
