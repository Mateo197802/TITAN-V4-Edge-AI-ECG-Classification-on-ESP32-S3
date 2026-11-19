import numpy as np
import pytest
import torch

from titan_v4.reproducibility.joint_refinement import (
    PATHOLOGY_CLASSES, PRIMARY5_INDEX, pathology_targets, masked_pathology_loss,
    calibrate_thresholds, pathology_metrics, select_extra_training, resolve_historical_ptb_ids, unambiguous_rhythm_target,
)


def test_pathology_order_and_alias_are_explicit():
    y, mask = pathology_targets('ptb-xl', {'IMI': 100, 'LAO/LAE': 0})
    assert PATHOLOGY_CLASSES == ('IMI','ASMI','LVH','ISC_','ISCAL','NST_','ILMI','AMI','ALMI','LAE')
    assert PRIMARY5_INDEX == (0, 8, 6, 9, 3)
    assert y.tolist() == [1,0,0,0,0,0,0,0,0,1]
    assert mask.all()


def test_unknown_sources_are_not_negative_pathology_labels():
    y, mask = pathology_targets('georgia', {'IMI': 100})
    assert not mask.any()
    logits = torch.zeros((1,10), requires_grad=True)
    loss = masked_pathology_loss(logits, torch.tensor(y[None]), torch.tensor(mask[None]), torch.ones(10))
    loss.backward()
    assert loss.item() == 0
    assert torch.equal(logits.grad, torch.zeros_like(logits))


def test_masked_loss_ignores_unknown_rows_and_stays_finite():
    logits = torch.zeros((2,10), requires_grad=True)
    y = torch.ones((2,10)); mask = torch.zeros((2,10)); mask[0] = 1
    loss = masked_pathology_loss(logits, y, mask, torch.ones(10))
    loss.backward()
    assert np.isclose(loss.item(), np.log(2))
    assert logits.grad[0].abs().sum() > 0
    assert logits.grad[1].abs().sum() == 0


def test_thresholds_require_validation_and_supported_targets():
    y = np.tile([0,1], (10,1)).T
    p = .1 + .8*y
    with pytest.raises(ValueError, match='validation'):
        calibrate_thresholds(y,p,role='test')
    t = calibrate_thresholds(y,p,role='validation')
    assert pathology_metrics(y,p,t)['primary5_macro_f1'] == 1
    assert pathology_metrics(y,np.zeros_like(p),np.full(10,.5))['primary5_macro_f1'] == 0
    with pytest.raises(ValueError, match='positive and negative'):
        calibrate_thresholds(np.zeros((2,10)),np.zeros((2,10)),role='validation')


def test_expansion_excludes_validation_and_followup_patients():
    rows = [dict(ecg_id='1', patient_id='p1', strat_fold='1', scp_codes="{'ALMI': 100}"),
            dict(ecg_id='2', patient_id='p2', strat_fold='9', scp_codes="{'ALMI': 100}"),
            dict(ecg_id='3', patient_id='p3', strat_fold='1', scp_codes="{'ALMI': 100}"),
            dict(ecg_id='4', patient_id='p4', strat_fold='1', scp_codes="{'SR': 0}")]
    assert [r['ecg_id'] for r in select_extra_training(rows, {'p3'}, {'1'})] == []


def test_invalid_pathology_score_is_not_silently_accepted():
    with pytest.raises(ValueError): pathology_targets('ptb-xl', {'IMI': float('nan')})


def test_historical_duplicate_alias_is_excluded_through_retained_record():
    assert resolve_historical_ptb_ids(['141','123'], {'2509','123'}) == ['2509','123']
    with pytest.raises(ValueError,match='Unresolved'):resolve_historical_ptb_ids(['99999'], {'2509'})


def test_added_rhythm_targets_use_only_unambiguous_official_statements():
    mapping={'SR':3,'RBBB':5,'AFIB':0,'PVC':4,'PACE':13}
    assert unambiguous_rhythm_target({'SR':0},mapping)==3
    assert unambiguous_rhythm_target({'SR':0,'RBBB':100},mapping)==5
    assert unambiguous_rhythm_target({'AFIB':0,'PVC':100},mapping)==-1
    assert unambiguous_rhythm_target({'PACE':100,'SR':0},mapping)==-1
    assert unambiguous_rhythm_target({'IMI':100},mapping)==-1


@pytest.mark.parametrize('outside',['SARRH','SVARR','SVTAC','PSVT','BIGU','TRIGU'])
def test_official_rhythms_absent_from_project_mapping_are_still_unknown(outside):
    assert unambiguous_rhythm_target({'SR':0,outside:100},{'SR':3,'AFIB':0}) == -1
