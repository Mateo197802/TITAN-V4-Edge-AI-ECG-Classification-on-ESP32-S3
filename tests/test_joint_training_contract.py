import importlib.util
from pathlib import Path
import sys

import pytest
import torch
from torch import nn

TRAINING=Path(__file__).resolve().parents[1]/'outputs/reviewer_verification/joint_refinement_2026-10-12/training'


@pytest.fixture
def trainer(monkeypatch):
    monkeypatch.syspath_prepend(str(TRAINING))
    for key,filename in [('contract','contract.py'),('joint_metrics','joint_metrics.py')]:
        spec=importlib.util.spec_from_file_location(key,TRAINING/filename)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        monkeypatch.setitem(sys.modules,key,module)
    spec=importlib.util.spec_from_file_location('joint_training_test',TRAINING/'train_joint.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_original_network_no_layer_reset_and_frozen_batchnorm(trainer):
    model=trainer.c.load_model(TRAINING/'start.pth')
    initial={k:v.clone() for k,v in model.state_dict().items()}
    params=trainer.configure(model,2,True)
    assert sum(p.numel() for p in model.parameters())==1143751
    assert sum(p.numel() for p in params)==778943
    assert all(not m.training for m in model.modules() if isinstance(m,nn.BatchNorm1d))
    assert trainer.assert_frozen(model,initial,2,True)==[]
    with torch.no_grad():model.head_quality[0].weight.add_(1)
    with pytest.raises(ValueError,match='Frozen'):trainer.assert_frozen(model,initial,2,True)


def test_head_only_control_does_not_enable_rhythm_or_backbone(trainer):
    model=trainer.c.load_model(TRAINING/'start.pth')
    params=trainer.configure(model,0,True)
    assert sum(p.numel() for p in params)==19498
    assert {n.split('.')[0] for n,p in model.named_parameters() if p.requires_grad}=={'head_pathology'}


def test_full_backbone_trial_keeps_count_buffers_and_auxiliary_heads(trainer):
    model=trainer.c.load_model(TRAINING/'start.pth')
    initial={k:v.clone() for k,v in model.state_dict().items()}
    params=trainer.configure(model,5,True)
    assert sum(p.numel() for p in params)==1118787
    assert all(not m.training for m in model.modules() if isinstance(m,nn.BatchNorm1d))
    assert not any(p.requires_grad for p in model.head_quality.parameters())
    assert not any(p.requires_grad for p in model.head_biometrics.parameters())
    assert trainer.assert_frozen(model,initial,5,True)==[]


def test_validation_eligibility_preserves_supported_class_recall(trainer):
    base={'ptb-xl':{'support':{'SB':25},'recall':{'SB':.9}}}
    assert trainer.protected_recalls({'ptb-xl':{'support':{'SB':25},'recall':{'SB':.85}}},base)
    assert not trainer.protected_recalls({'ptb-xl':{'support':{'SB':25},'recall':{'SB':.8}}},base)


def test_protocol_freezes_targets_and_selection_roles(trainer):
    assert trainer.PROTOCOL['data']['references_changed'] is False
    assert trainer.PROTOCOL['selection']['test_predictions_not_used'] is True
    assert trainer.PROTOCOL['architecture']['parameters']==1143751
