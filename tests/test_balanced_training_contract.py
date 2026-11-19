import importlib.util
from pathlib import Path
import sys

import pytest
import torch
from torch import nn


ROOT=Path(__file__).resolve().parents[1]
TRAINING=ROOT/'outputs/reviewer_verification/balanced_joint_2026-10-13/training'


@pytest.fixture
def trainer(monkeypatch):
    monkeypatch.syspath_prepend(str(TRAINING))
    for name in ['contract','joint_metrics','baseline_trainer','balanced_metrics']:
        spec=importlib.util.spec_from_file_location(name,TRAINING/(name+'.py'))
        value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value)
        monkeypatch.setitem(sys.modules,name,value)
    spec=importlib.util.spec_from_file_location('balanced_trainer_test',TRAINING/'train_balanced.py')
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value)
    return value


def test_warm_start_architecture_and_frozen_heads(trainer):
    model=trainer.c.load_model(TRAINING/'start.pth')
    for blocks,count in [(0,19498),(2,778943),(5,1118787)]:
        parameters=trainer.b.configure(model,blocks,True)
        assert sum(p.numel() for p in parameters)==count
        assert sum(p.numel() for p in model.parameters())==1143751
        assert all(not m.training for m in model.modules() if isinstance(m,nn.BatchNorm1d))
        assert not any(p.requires_grad for p in model.head_quality.parameters())
        assert not any(p.requires_grad for p in model.head_biometrics.parameters())
    with pytest.raises(RuntimeError,match='CEDIA'):trainer.c.require_cedia_gpu()


def test_predeclared_trials_keep_targets_and_published_seed(trainer):
    protocol=trainer.b.PROTOCOL
    assert protocol['start_sha256']=='fd8a5d6ca95b7875e53305ef0fdc3962f2aa4d493fa16630befc7a75e1cb2249'
    assert protocol['data']['references_changed'] is False
    assert protocol['selection']['test_predictions_not_used'] is True
    assert len(protocol['optimization']['trials'])==7
    assert {t['mode'] for t in protocol['optimization']['trials']}=={'legacy','agreement','balanced'}
    assert protocol['optimization']['epochs']==40


def test_new_loss_does_not_reset_or_add_network_layers(trainer):
    model=trainer.c.load_model(TRAINING/'start.pth');initial={k:v.clone() for k,v in model.state_dict().items()}
    trainer.b.configure(model,2,True)
    assert trainer.b.assert_frozen(model,initial,2,True)==[]
    with torch.no_grad():model.head_biometrics[0].weight.add_(1)
    with pytest.raises(ValueError,match='Frozen'):trainer.b.assert_frozen(model,initial,2,True)


def test_duplicate_streams_do_not_double_start_anchor_coefficient(trainer):
    value=torch.tensor([[1.,2.]])
    reference=torch.tensor([[.4,.6]])
    y=torch.tensor([1]);stream=(value,reference,y)
    expected=.1*trainer.agreement_anchor_loss(*stream)
    assert torch.allclose(trainer.starting_anchor([stream]),expected)
    assert torch.allclose(trainer.starting_anchor([stream,stream]),expected)
