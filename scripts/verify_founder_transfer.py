"""Verify the sealed study and, by default, rerun its student source inference."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from titan_v4.reproducibility.pretrained_transfer.evidence import verify_founder_bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, default=ROOT / 'outputs/reviewer_verification/founder_transfer_2026-10-14')
    parser.add_argument('--data-root', type=Path, default=ROOT / 'data/cache/physionet')
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--metrics-only', action='store_true', help='Verify bytes/references/selection without independent source inference')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = verify_founder_bundle(args.bundle, data_root=args.data_root, offline=args.offline,
                                   source_inference=not args.metrics_only)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n')
    print(json.dumps({'verified': result['verified'], 'source_inference_verified': result['source_inference_verified'],
                      'selected_arm': result['selection']['selected_arm'], 'output': str(args.output)}, indent=2))


if __name__ == '__main__':
    main()
