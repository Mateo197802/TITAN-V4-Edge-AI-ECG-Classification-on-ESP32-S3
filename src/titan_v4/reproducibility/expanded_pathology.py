"""Outcome-blind native PTB-XL population expansion, with patient exclusions."""


def select_expanded_population(metadata, excluded_patients, training_patients, existing_train_ids):
    train, validation = [], []
    ids, patient_folds = set(), {}
    for row in metadata:
        rid, patient, fold = str(int(row['ecg_id'])), row['patient_id'], int(row['strat_fold'])
        if rid in ids:
            raise ValueError('Duplicate ECG identifier')
        ids.add(rid)
        if not 1 <= fold <= 10:
            raise ValueError('Invalid official fold')
        if patient in patient_folds and patient_folds[patient] != fold:
            raise ValueError('One patient crosses official folds')
        patient_folds[patient] = fold
        if patient in excluded_patients:
            continue
        if fold <= 8 and rid not in existing_train_ids:
            train.append(row)
        elif fold == 9 and patient not in training_patients:
            validation.append(row)
    key = lambda row: int(row['ecg_id'])
    return sorted(train, key=key), sorted(validation, key=key)
