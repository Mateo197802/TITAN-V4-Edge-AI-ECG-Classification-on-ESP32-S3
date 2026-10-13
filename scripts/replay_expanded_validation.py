"""Replay expanded calibration and original rhythm validation from original signals."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import numpy as np
import torch

from replay_joint_validation_auxiliary import ROOT, predict, load_primary9_model, load_tensor, read_manifest, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--offline', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a fresh replay directory')
    bundle = args.bundle
    sets = [read_manifest(ROOT / 'outputs/reviewer_verification/partial_backbone_refinement_2026-10-12/manifests/validation_records.csv'),
            read_manifest(bundle / 'data/path_validation_records.csv')]
    tensors = []
    for rows in sets:
        with ThreadPoolExecutor(max_workers=4) as pool:
            tensors.append(np.stack(list(pool.map(lambda row: load_tensor(row, ROOT / 'data/cache/physionet', download=not args.offline), rows))))
    inventory = json.loads((bundle / 'results/SHA256SUMS.json').read_text())
    selection = json.loads((bundle / 'results/selection.json').read_text())
    thresholds = {r['name']: r['thresholds'] for r in selection['candidates']}
    thresholds['baseline'] = selection['baseline']['thresholds']
    torch.set_num_threads(2)
    checks = []
    for name in ['baseline'] + [r['name'] for r in selection['candidates']]:
        checkpoint = bundle / 'training/start.pth' if name == 'baseline' else bundle / 'results' / (name + '.pth')
        if name != 'baseline' and sha256(checkpoint) != inventory[name + '.pth']:
            raise ValueError('Checkpoint hash differs')
        model = load_primary9_model(checkpoint)
        rv, pv = [predict(model, x) for x in tensors]
        rp = torch.softmax(torch.from_numpy(rv[0]), 1).numpy()
        pp = torch.sigmoid(torch.from_numpy(pv[3])).numpy()
        saved = np.load(bundle / 'results' / (name + '_validation.npz'), allow_pickle=False)
        maximum = float(max(np.max(np.abs(rp - saved['rhythm_probabilities'])), np.max(np.abs(pp - saved['pathology_probabilities']))))
        t = np.asarray(thresholds[name])
        if maximum > 2e-5 or not np.array_equal(rp.argmax(1), saved['rhythm_probabilities'].argmax(1)) or not np.array_equal(pp >= t, saved['pathology_probabilities'] >= t):
            raise ValueError('Source inference replay differs: ' + name)
        aux = float(max(np.max(np.abs(rv[1] - saved['quality'])), np.max(np.abs(rv[2] - saved['biometrics']))))
        if aux > 2e-5:
            raise ValueError('Auxiliary outputs differ')
        checks.append({'name': name, 'checkpoint_sha256': sha256(checkpoint), 'maximum_probability_difference': maximum,
                       'maximum_auxiliary_difference': aux, 'rhythm_decisions_identical': True, 'pathology_decisions_identical': True})
        print(name + ': source replay passed', flush=True)
    args.output.mkdir(parents=True)
    report = {'rhythm_records': len(sets[0]), 'pathology_records': len(sets[1]), 'checkpoints': len(checks),
              'checks': checks, 'all_passed': True, 'device': 'cpu', 'torch': torch.__version__, 'numpy': np.__version__}
    (args.output / 'validation_source_replay.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
