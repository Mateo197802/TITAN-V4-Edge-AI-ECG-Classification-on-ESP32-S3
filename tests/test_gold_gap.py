import numpy as np
import pytest
import torch
from torch.nn import functional as F

from titan_v4.reproducibility.gold_gap import effective_rhythm_target, additional_correct


def test_target_is_exact_optimum_of_recorded_rhythm_loss():
    gold = np.array([[.1,.9]])
    teacher = np.array([[.8,.2]])
    target = effective_rhythm_target(np.array([0]), gold, teacher, 2., .15)
    logits = torch.tensor(np.log(target), dtype=torch.float64, requires_grad=True)
    loss = F.cross_entropy(logits, torch.tensor([0]))
    loss += 2 * F.kl_div(F.log_softmax(logits,1), torch.tensor(gold), reduction='batchmean')
    loss += .15 * F.kl_div(F.log_softmax(logits,1), torch.tensor(teacher), reduction='batchmean')
    loss.backward()
    assert torch.max(torch.abs(logits.grad)) < 1e-12
    assert target.argmax(1).tolist() == [1]


def test_reducing_wrong_gold_anchor_preserves_reference_argmax():
    p = effective_rhythm_target(np.array([0]), np.array([[.1,.9]]), np.array([[.8,.2]]), .5, .15)
    assert p.argmax(1).tolist() == [0]


def test_targets_and_teachers_are_not_modified():
    y=np.array([0]); g=np.array([[.1,.9]]); t=np.array([[.8,.2]])
    before=(y.copy(),g.copy(),t.copy())
    effective_rhythm_target(y,g,t,2,.15)
    assert all(np.array_equal(a,b) for a,b in zip((y,g,t),before))


@pytest.mark.parametrize('bad', [np.array([[np.nan,.9]]),np.array([[.1,.8]]),np.array([[-.1,1.1]])])
def test_invalid_teacher_probabilities_are_rejected(bad):
    with pytest.raises(ValueError):
        effective_rhythm_target(np.array([0]),bad,np.array([[.8,.2]]),2,.15)


def test_target_gap_is_rounded_up_and_does_not_imply_macro_f1():
    assert additional_correct(672,576)==29
    assert additional_correct(1476,1185)==144
    assert additional_correct(672,605)==0


def test_frozen_training_cache_recalculates_recorded_objective_conflict():
    import json
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]
    report=json.loads((root/'outputs/reviewer_verification/gold_target_gap_2026-10-12/gap_audit.json').read_text())
    with np.load(root/'outputs/reviewer_verification/partial_backbone_refinement_2026-10-12/training/phase2_reference/train_features.npz') as data:
        target=effective_rhythm_target(data['y'],data['gold'],data['teacher'],2,.15)
        assert int((target.argmax(1)!=data['y']).sum())==report['training_objective']['2.0']['per_record_optimum_disagrees_with_reference']==1007
