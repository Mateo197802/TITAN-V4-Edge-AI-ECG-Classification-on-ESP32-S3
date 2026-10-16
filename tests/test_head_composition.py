import pytest
import torch

from titan_v4.reproducibility.head_composition import compose_head_states
from titan_v4.models.titan_v4_net import TitanV4Lite


def test_composition_requires_identical_shared_body_and_auxiliary_state():
    model=TitanV4Lite(in_channels=6,num_rhythm=9,num_pathology=10,morphology_dim=0)
    rhythm={k:v.clone() for k,v in model.state_dict().items()};path={k:v.clone() for k,v in rhythm.items()}
    path['stem.0.weight'].add_(1)
    with pytest.raises(ValueError,match='Shared'):compose_head_states(rhythm,path)
    path={k:v.clone() for k,v in rhythm.items()};path['head_quality.0.weight'].add_(1)
    with pytest.raises(ValueError,match='Shared'):compose_head_states(rhythm,path)


def test_actual_single_model_preserves_chosen_heads_and_parameter_count():
    torch.set_num_threads(2)
    model=TitanV4Lite(in_channels=6,num_rhythm=9,num_pathology=10,morphology_dim=0).eval()
    rhythm={k:v.clone() for k,v in model.state_dict().items()};path={k:v.clone() for k,v in rhythm.items()}
    path['head_pathology.0.bias'].add_(1)
    combined=compose_head_states(rhythm,path);model.load_state_dict(combined,strict=True)
    assert sum(p.numel() for p in model.parameters())==1143751
    assert all(torch.equal(combined[k],path[k] if k.startswith('head_pathology.') else rhythm[k]) for k in combined)
    with torch.inference_mode():
        x=torch.zeros(2,6,1250);actual=model(x)
        model.load_state_dict(rhythm);r=model(x)
        model.load_state_dict(path);p=model(x)
    assert torch.equal(actual[0],r[0])
    assert torch.equal(actual[3],p[3])


def test_shared_signed_zero_is_not_accepted_as_byte_identical():
    r={'stem.weight':torch.tensor([0.]),'head_pathology.bias':torch.tensor([1.])}
    p={'stem.weight':torch.tensor([-0.]),'head_pathology.bias':torch.tensor([1.])}
    assert torch.equal(r['stem.weight'],p['stem.weight'])
    with pytest.raises(ValueError,match='Shared'):compose_head_states(r,p)
