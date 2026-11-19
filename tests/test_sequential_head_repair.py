import numpy as np
import pytest
import torch
from torch import nn

from titan_v4.reproducibility.head_repair import configure_heads, assert_heads_frozen
from titan_v4.models.titan_v4_net import TitanV4Lite


def test_existing_two_heads_only_no_new_parameters_or_bn_updates():
    model=TitanV4Lite(in_channels=6,num_rhythm=9,num_pathology=10,morphology_dim=0)
    initial={k:v.clone() for k,v in model.state_dict().items()}
    params=configure_heads(model,rhythm=True)
    assert sum(p.numel() for p in params)==38899
    assert sum(p.numel() for p in model.parameters())==1143751
    assert all(not m.training for m in model.modules() if isinstance(m,nn.BatchNorm1d))
    assert assert_heads_frozen(model,initial,rhythm=True)==[]
    with torch.no_grad():model.stem[0].weight.add_(1)
    with pytest.raises(ValueError,match='Frozen'):assert_heads_frozen(model,initial,rhythm=True)


def test_pathology_only_control_cannot_change_rhythm_quality_biometrics():
    model=TitanV4Lite(in_channels=6,num_rhythm=9,num_pathology=10,morphology_dim=0)
    initial={k:v.clone() for k,v in model.state_dict().items()}
    params=configure_heads(model,rhythm=False)
    assert sum(p.numel() for p in params)==19498
    with torch.no_grad():model.head_rhythm[0].weight.add_(1)
    with pytest.raises(ValueError,match='Frozen'):assert_heads_frozen(model,initial,rhythm=False)


def test_native_feature_path_matches_full_existing_network():
    torch.set_num_threads(2)
    model=TitanV4Lite(in_channels=6,num_rhythm=9,num_pathology=10,morphology_dim=0).eval()
    with torch.inference_mode():
        x=torch.zeros(2,6,1250);features=model.forward_features(x);full=model(x)
        cached=tuple(getattr(model,name)(features) for name in ['head_rhythm','head_quality','head_biometrics','head_pathology'])
    assert features.shape==(2,192)
    assert all(torch.equal(a,b) for a,b in zip(full,cached))
