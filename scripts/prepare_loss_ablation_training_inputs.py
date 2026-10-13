"""Reconstruct frozen signals and prepare the CEDIA agreement-loss replay."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from titan_v4.reproducibility.refinement import sha256

BUNDLE=ROOT/'outputs/reviewer_verification/loss_anchor_ablation_2026-10-12'
START=ROOT/'outputs/reviewer_verification/partial_backbone_refinement_2026-10-12/model/partial_backbone_candidate.pth'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--data-root',type=Path,default=ROOT/'data/cache/physionet')
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--offline',action='store_true')
    args=parser.parse_args()
    if args.output_dir.exists():raise FileExistsError('Use a fresh reproduction directory')
    command=[sys.executable,str(ROOT/'scripts/prepare_refinement_training_inputs.py'),
             '--output-dir',str(args.output_dir),'--data-root',str(args.data_root),'--workers',str(args.workers)]
    if args.offline:command.append('--offline')
    subprocess.run(command,check=True)
    for name in ('contract.py','model_core.py','titan_v4_net.py','train_phase5.py','test_phase5.py','protocol.json','job.sbatch'):
        shutil.copy2(BUNDLE/'training'/name,args.output_dir/name)
    shutil.copy2(START,args.output_dir/'start.pth')
    hashes={path.name:sha256(path) for path in sorted(args.output_dir.iterdir()) if path.is_file() and path.name!='INPUT_SHA256.json'}
    (args.output_dir/'INPUT_SHA256.json').write_text(json.dumps(hashes,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'prepared_files':len(hashes),'warm_start_sha256':sha256(args.output_dir/'start.pth')},indent=2))


if __name__=='__main__':main()
