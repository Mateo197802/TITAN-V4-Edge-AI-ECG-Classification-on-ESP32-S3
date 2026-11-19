import importlib.util
from pathlib import Path
import sys

import pytest
import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[1]
TRAINING=ROOT/'outputs/reviewer_verification/loss_anchor_ablation_2026-10-12/training'


@pytest.fixture
def trainer(monkeypatch):
    monkeypatch.syspath_prepend(str(TRAINING))
    spec=importlib.util.spec_from_file_location('phase5_contract_test',TRAINING/'contract.py')
    contract=importlib.util.module_from_spec(spec);spec.loader.exec_module(contract)
    monkeypatch.setitem(sys.modules,'contract',contract)
    spec=importlib.util.spec_from_file_location('phase5_training_test',TRAINING/'train_phase5.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_training_agreement_mask_does_not_replace_reference_labels(trainer):
    y=torch.tensor([0,1]);gold=torch.tensor([[.1,.9],[.1,.9]]);teacher=torch.tensor([[.9,.1],[.9,.1]])
    original=y.clone();g,t=trainer.agreement_weights(y,gold,teacher,2,.15,True,True)
    assert torch.equal(y,original)
    assert torch.allclose(g,torch.tensor([0.,2.]))
    assert torch.allclose(t,torch.tensor([.15,0.]))


def test_control_pair_is_a_single_anchor_ablation(trainer):
    control,ablation=trainer.PROTOCOL['optimization']['trials'][:2]
    assert {k:v for k,v in control.items() if k not in ('name','gold_weight')}=={
        k:v for k,v in ablation.items() if k not in ('name','gold_weight')}
    assert (control['gold_weight'],ablation['gold_weight'])==(2.,0.)


def test_no_new_parameters_heads_or_trainable_buffers(trainer):
    model=trainer.c.load_model(TRAINING/'start.pth')
    initial={k:v.clone() for k,v in model.state_dict().items()}
    parameters=trainer.configure(model,2)
    assert sum(p.numel() for p in model.parameters())==1143751
    assert sum(p.numel() for p in parameters)==759445
    assert all(not m.training for m in model.modules() if isinstance(m,nn.BatchNorm1d))
    assert trainer.assert_frozen(model,initial,2)==[]
    with torch.no_grad():model.stem[0].weight.add_(1)
    with pytest.raises(ValueError,match='Frozen'):
        trainer.assert_frozen(model,initial,2)
