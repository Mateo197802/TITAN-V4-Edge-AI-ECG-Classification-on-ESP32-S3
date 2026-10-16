import numpy as np
import pytest

from titan_v4.reproducibility.supervision_control import source_masses, fixed_source_weights, confidence_weights, crossfit_pathology, patient_folds, configure_role, assert_role_frozen, select_components, verify_fold_alignment


def test_source_mass_is_preserved_when_source_gains_classes():
    y = np.array([0, 0, 1, 0, 0, -1]); source = np.array(['a','a','a','b','b','b'])
    mass = source_masses(y, source, role='train')
    assert mass == {'a': 2/3, 'b': 1/3}
    new_y = np.r_[y, 1, 2]; new_source = np.concatenate((source, ['b','b']))
    w = fixed_source_weights(new_y, new_source, mass, role='train')
    assert w[5] == 0
    assert w[new_source == 'a'].sum() / w.sum() == pytest.approx(2/3)
    assert w[new_source == 'b'].sum() / w.sum() == pytest.approx(1/3)
    with pytest.raises(ValueError):
        fixed_source_weights(new_y, new_source, mass, role='followup')


def test_unknown_likelihood_is_not_a_negative_or_zero_weight():
    targets = np.zeros((2,10),np.float32); targets[:,0] = 1
    likelihood = np.ones((2,10),np.float32)*100; likelihood[0,0] = 0; likelihood[1,0] = 15
    mask = np.ones_like(targets)
    human, certainty = confidence_weights(targets, mask, np.array([True,False]), likelihood, role='train')
    assert human[0,0] == 1 and human[1,0] == .5
    assert certainty[0,0] == 1
    assert certainty[1,0] == pytest.approx(.5*.575)
    assert np.array_equal(targets[:,0], [1,1])
    mask[:] = 0
    assert not confidence_weights(targets, mask, np.array([True,False]), likelihood, role='train')[0].any()


def test_patient_folds_keep_repeated_patient_in_one_group():
    assert np.array_equal(patient_folds(['x','x','y']), patient_folds(['x','x','y']))
    assert patient_folds(['x','x','y'])[0] == patient_folds(['x','x','y'])[1]


def test_crossfit_thresholds_do_not_consult_held_out_labels():
    y = np.tile(np.eye(10,dtype=np.float32),(10,1)); p = y*.8+.1
    folds = np.repeat(np.arange(5),20)
    first = crossfit_pathology(y,p,folds,role='validation')
    assert first['primary5_macro_f1'] == 1
    changed = y.copy(); changed[folds==0] = 1-changed[folds==0]
    second = crossfit_pathology(changed,p,folds,role='validation')
    assert first['folds'][0]['thresholds'] == second['folds'][0]['thresholds']
    assert first['folds'][0]['calibration_indices_sha256'] == second['folds'][0]['calibration_indices_sha256']
    with pytest.raises(ValueError):
        crossfit_pathology(y,p,folds,role='followup')


def test_separate_head_updates_leave_other_parameters_and_buffers_unchanged():
    import torch
    from torch import nn
    model=nn.Module(); model.stem=nn.BatchNorm1d(3)
    for role in ('rhythm','pathology','quality','biometrics'):
        setattr(model,'head_'+role,nn.Sequential(nn.Linear(3,2),nn.Dropout(.3)))
    initial={k:v.clone() for k,v in model.state_dict().items()}
    params=configure_role(model,'rhythm')
    assert len(params)==2 and not model.stem.training and not model.head_pathology.training
    with torch.no_grad(): params[0].add_(.01)
    assert_role_frozen(model,initial,'rhythm')
    with torch.no_grad(): model.head_pathology[0].weight.add_(.01)
    with pytest.raises(ValueError): assert_role_frozen(model,initial,'rhythm')


def test_selection_requires_oof_path_gain_not_only_resubstitution_gain():
    base={'metrics':{'macro_f1':.8},'rhythm_score':.7,'pathology_oof':{'primary5_macro_f1':.4},'pathology_ap':.45,'pathology':{'primary5_macro_f1':.5}}
    bad={**base,'name':'bad','trial':{'task':'pathology'},'eligible':True,'pathology':{'primary5_macro_f1':.7}}
    good={**bad,'name':'good','pathology_oof':{'primary5_macro_f1':.42}}
    result=select_components([bad,good],base,role='validation')
    assert result['pathology_component']['name']=='good' and result['rhythm_component'] is None


def test_fold_alignment_rejects_positional_shift_and_record_reordering():
    rows=[{'ecg_id':str(i),'patient_token':str(i//2)} for i in range(100)]
    ids=np.asarray([r['ecg_id'] for r in rows]); folds=patient_folds([r['patient_token'] for r in rows])
    verify_fold_alignment(folds,ids,rows)
    with pytest.raises(ValueError): verify_fold_alignment(np.roll(folds,1),ids,rows)
    with pytest.raises(ValueError): verify_fold_alignment(folds,np.roll(ids,1),rows)
