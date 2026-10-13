import numpy as np
import pytest
import torch

from titan_v4.reproducibility.balanced_refinement import (
    pathology_sampling_probabilities, rhythm_training_weights,
    agreement_anchor_loss, asymmetric_pathology_loss, select_candidates,
    sample_pathology_indices,
)


def test_sampler_uses_known_training_targets_and_half_negative_mass():
    y=np.zeros((6,10),np.float32);mask=np.ones_like(y)
    y[0,0]=1;y[1,0]=1;y[2,8]=1;mask[5]=0
    p=pathology_sampling_probabilities(y,mask,role='train')
    assert p[5]==0
    assert p[:3].sum()==pytest.approx(.5)
    assert p[3:5].sum()==pytest.approx(.5)
    assert p[2]>p[0]
    with pytest.raises(ValueError,match='train'):
        pathology_sampling_probabilities(y,mask,role='followup')


def test_sampler_rejects_partial_masks_and_missing_negative_group():
    y=np.ones((4,10),np.float32);m=np.ones_like(y)
    with pytest.raises(ValueError,match='positive and negative'):
        pathology_sampling_probabilities(y,m,role='train')
    m[0,1]=0
    with pytest.raises(ValueError,match='mask'):
        pathology_sampling_probabilities(y,m,role='train')


def test_rhythm_weights_exclude_unknowns_and_equalize_source_class_mass():
    y=np.array([3,3,7,3,-1]);source=np.array(['a','a','a','b','b'])
    w=rhythm_training_weights(y,source,role='train')
    assert w[-1]==0
    assert w[:2].sum()==pytest.approx(w[2])
    assert w[2]==pytest.approx(w[3])
    assert w[y>=0].mean()==pytest.approx(1)
    with pytest.raises(ValueError):rhythm_training_weights(y,source,role='validation')


def test_anchor_does_not_penalize_correcting_a_wrong_initial_prediction():
    logits=torch.tensor([[0.,2.],[2.,0.]])
    teacher=torch.tensor([[.9,.1],[.8,.2]])
    y=torch.tensor([1,0])
    value=agreement_anchor_loss(logits,teacher,y)
    expected=torch.nn.functional.kl_div(logits[1:].log_softmax(1),teacher[1:],reduction='batchmean')
    assert torch.allclose(value,expected)
    assert agreement_anchor_loss(logits[:1],teacher[:1],y[:1]).item()==0


def test_asymmetric_loss_masks_unknowns_and_matches_bce_when_disabled():
    x=torch.zeros(2,10);y=torch.zeros_like(x);y[0,0]=1;m=torch.ones_like(x);m[1]=0
    value=asymmetric_pathology_loss(x,y,m,gamma_negative=0,clip=0)
    expected=torch.nn.functional.binary_cross_entropy_with_logits(x[:1],y[:1])
    assert torch.allclose(value,expected)
    x[1]=100
    assert torch.allclose(value,asymmetric_pathology_loss(x,y,m,gamma_negative=0,clip=0))
    assert asymmetric_pathology_loss(x,y,torch.zeros_like(m)).item()==0
    with pytest.raises(ValueError):asymmetric_pathology_loss(x,y,m+.1)


def test_asymmetric_loss_reduces_easy_negatives_without_lowering_positive_term():
    x=torch.full((1,10),-4.);y=torch.zeros_like(x);m=torch.ones_like(x)
    plain=asymmetric_pathology_loss(x,y,m,gamma_negative=0,clip=0)
    assert asymmetric_pathology_loss(x,y,m)<plain
    y[:]=1
    assert torch.allclose(asymmetric_pathology_loss(x,y,m),asymmetric_pathology_loss(x,y,m,gamma_negative=0,clip=0))


@pytest.mark.parametrize('positive,gamma,clip',[(True,0,0),(False,0,0),(True,4,.05)])
def test_extreme_wrong_logits_keep_corrective_loss_gradients(positive,gamma,clip):
    x=torch.full((1,10),-20. if positive else 20.,requires_grad=True)
    y=torch.full_like(x,1. if positive else 0.)
    loss=asymmetric_pathology_loss(x,y,torch.ones_like(x),gamma_negative=gamma,clip=clip)
    loss.backward()
    assert loss.item()==pytest.approx(20.,rel=1e-6)
    assert torch.allclose(x.grad,torch.full_like(x,-.1 if positive else .1),atol=1e-6)


def test_selection_uses_validation_and_does_not_trade_away_pathology():
    base={'rhythm_score':.7,'pathology':{'primary5_macro_f1':.5}}
    candidates=[{'name':'harmful','eligible':True,'rhythm_score':.9,'joint_score':1.1,'pathology':{'primary5_macro_f1':.4}},
                {'name':'joint','eligible':True,'rhythm_score':.71,'joint_score':.985,'pathology':{'primary5_macro_f1':.55}}]
    selected=select_candidates(candidates,base,role='validation')
    assert selected['rhythm_selected']['name']=='joint'
    assert selected['joint_selected']['name']=='joint'
    assert select_candidates([],base,role='validation')=={'joint_selected':None,'rhythm_selected':None}
    with pytest.raises(ValueError):select_candidates(candidates,base,role='followup')


def test_sampling_uses_a_dedicated_cpu_generator_without_consuming_global_rng():
    probabilities=torch.tensor([.2,.3,0.,.5],dtype=torch.float64)
    before=torch.random.get_rng_state().clone()
    first=sample_pathology_indices(probabilities,256,generator=torch.Generator(device='cpu').manual_seed(21))
    second=sample_pathology_indices(probabilities,256,generator=torch.Generator(device='cpu').manual_seed(21))
    assert torch.equal(first,second)
    assert torch.equal(before,torch.random.get_rng_state())
    assert first.device.type=='cpu' and first.shape==(256,)
    assert not (first==2).any()
