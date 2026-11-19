import numpy as np
import pytest

from titan_v4.reproducibility.plateau_audit import error_decomposition, shared_errors


def test_error_decomposition_reconciles_full_support_without_dropping_absent_classes():
    y=np.array([0,0,1,2]); pred=np.array([0,1,0,2])
    r=error_decomposition(y,pred)
    assert r['records']==4 and r['errors']==2
    assert sum(r['class_errors'])==r['errors']
    assert r['top_confusions']==[{'reference':0,'prediction':1,'records':1},{'reference':1,'prediction':0,'records':1}]
    assert len(r['class_errors'])==9
    with pytest.raises(ValueError): error_decomposition(np.array([-1]),np.array([0]))


def test_shared_error_oracle_is_diagnostic_and_not_a_reproducible_classifier():
    y=np.array([0,1,2,3]); predictions=np.array([[0,0,0,3],[1,1,0,3]])
    r=shared_errors(y,predictions)
    assert r['all_models_wrong']==1 and r['any_model_correct']==3
    assert r['oracle_accuracy']==.75 and r['deployable_classifier'] is False
    assert r['pairwise'][0]['first_only_correct']==1
    assert r['pairwise'][0]['second_only_correct']==1
    with pytest.raises(ValueError): shared_errors(y,np.ones((2,3),int))


def test_current_preprocessing_is_invariant_to_positive_lead_gain():
    from titan_v4.reproducibility.physionet import preprocess_primary9_signal
    t=np.arange(5000)/500
    raw=np.column_stack([np.sin((i+1)*2*np.pi*t)+.2*np.cos(3*np.pi*t) for i in range(6)])
    leads=['I','II','III','aVR','aVL','aVF']
    first=preprocess_primary9_signal(raw,leads,500)
    second=preprocess_primary9_signal(raw*np.arange(2,8)[None,:],leads,500)
    assert np.max(np.abs(first-second))<1e-4
