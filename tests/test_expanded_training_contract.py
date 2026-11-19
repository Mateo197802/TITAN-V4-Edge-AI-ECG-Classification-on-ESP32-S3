import importlib
from pathlib import Path

import numpy as np
import torch


def trainer(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / 'outputs/reviewer_verification/expanded_population_2026-10-13/training'))
    return importlib.import_module('train_expanded')


def test_pathology_calibration_does_not_use_rhythm_selected_population(monkeypatch):
    module = trainer(monkeypatch)
    y = np.arange(9)
    logits = np.eye(9, dtype=np.float32) * 8
    values = (logits, np.zeros((9, 1), np.float32), np.zeros((9, 3), np.float32), np.zeros((9, 10), np.float32))
    val = {'y': y, 'dataset': np.full(9, 'ptb-xl')}
    targets = np.tile(np.eye(10, dtype=np.float32), (2, 1))
    pval = {'targets': targets}
    pvalues = (None, None, None, (targets * 2 - 1) * 8)
    sources = module.c.source_metrics(val, module.b.rhythm_prob(values))
    base = {'metrics': module.c.metrics(y, module.b.rhythm_prob(values)), 'sources': sources, 'rhythm_score': 1.0}
    actual = module.report(val, values, values, pval, pvalues, base)
    assert actual['metrics']['n'] == 9
    assert actual['pathology']['records'] == 20
    assert actual['pathology']['primary5_macro_f1'] == 1.0
    assert actual['eligible']


def test_original_model_count_and_path_only_freeze_contract(monkeypatch):
    module = trainer(monkeypatch)
    checkpoint = Path(__file__).resolve().parents[1] / 'outputs/reviewer_verification/expanded_population_2026-10-13/training/start.pth'
    model = module.c.load_model(checkpoint)
    initial = {k: v.clone() for k, v in model.state_dict().items()}
    params = module.b.configure(model, 0, True)
    assert sum(p.numel() for p in model.parameters()) == 1143751
    assert sum(p.numel() for p in params) == 19498
    assert all(n.startswith('head_pathology.') for n, p in model.named_parameters() if p.requires_grad)
    with torch.no_grad():
        next(p for p in params).add_(0.01)
    changed = module.b.assert_frozen(model, initial, 0, True)
    assert changed and all(k.startswith('head_pathology.') for k in changed)
