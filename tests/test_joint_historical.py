import importlib.util
from pathlib import Path
import pytest


def module():
    spec=importlib.util.spec_from_file_location('joint_history_test',Path(__file__).resolve().parents[1]/'scripts/run_joint_historical.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def test_pairing_preserves_both_references_and_scores_one_prediction():
    m=module();labels=m.PRIMARY9_CLASSES
    row={'source_label':'DATA_TEST','record_id':'a001','predicted_label':'NSR',**{'prob_'+c:float(c=='NSR') for c in labels}}
    history={'source':'data_test','record_id':'A001','original_rhythm_label':'SB','final_rhythm_label':'NSR'}
    previous={'source':'data_test','record_id':'A001','prediction':'SB'}
    paired=m.pair_rows([row],[history],[previous])
    assert paired[0]['original_rhythm_label']=='SB'
    assert paired[0]['final_rhythm_label']=='NSR'
    assert paired[0]['prediction']=='NSR'
    assert paired[0]['previous_prediction']=='SB'


def test_missing_or_duplicate_records_are_rejected():
    m=module()
    with pytest.raises(ValueError):m.pair_rows([], [{'source':'x','record_id':'a'}], [])
