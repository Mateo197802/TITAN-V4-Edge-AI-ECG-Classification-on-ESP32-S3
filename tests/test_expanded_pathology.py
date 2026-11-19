import pytest

from titan_v4.reproducibility.expanded_pathology import select_expanded_population


def row(rid, patient, fold):
    return {'ecg_id': str(rid), 'patient_id': patient, 'strat_fold': str(fold)}


def test_selection_keeps_negatives_and_positive_cases_without_prediction_access():
    rows = [row(5, 'b', 9), row(3, 'c', 8), row(1, 'a', 1), row(9, 'z', 10)]
    train, validation = select_expanded_population(rows, set(), set(), {'1'})
    assert [r['ecg_id'] for r in train] == ['3']
    assert [r['ecg_id'] for r in validation] == ['5']


def test_excludes_every_record_of_blocked_patients():
    rows = [row(1, 'historic', 2), row(2, 'historic', 2), row(3, 'followup', 9),
            row(4, 'training', 9), row(5, 'clean', 9)]
    train, validation = select_expanded_population(rows, {'historic', 'followup'}, {'training'}, set())
    assert train == []
    assert [r['ecg_id'] for r in validation] == ['5']


def test_rejects_duplicate_ids_and_invalid_folds():
    with pytest.raises(ValueError, match='Duplicate'):
        select_expanded_population([row(1, 'a', 1), row(1, 'b', 9)], set(), set(), set())
    with pytest.raises(ValueError, match='fold'):
        select_expanded_population([row(1, 'a', 0)], set(), set(), set())


def test_patient_must_not_cross_official_folds():
    with pytest.raises(ValueError, match='patient'):
        select_expanded_population([row(1, 'a', 1), row(2, 'a', 9)], set(), set(), set())
