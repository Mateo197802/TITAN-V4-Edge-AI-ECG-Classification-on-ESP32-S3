import numpy as np
import pytest
import torch

from titan_v4.reproducibility.directed_refinement import rhythm_reference, select_records, project_shared_gradients, validate_header_reference


def test_reference_accepts_one_primary_diagnosis_with_sinus_background():
    assert rhythm_reference({'426783006', '59118001'}) == 5
    assert rhythm_reference({'284470004', '164865005'}) == 7
    assert rhythm_reference({'426783006'}) == 3


@pytest.mark.parametrize('codes', [
    {'427393009', '426783006'}, {'426761007', '59118001'},
    {'284470004', '17338001'}, {'10370003', '426783006'}, set(),
    {'426995002', '59118001'}, {'251164006', '284470004'},
    {'195060002', '270492004'},
    {'426648003', '59118001'}, {'999999999', '284470004'},
])
def test_reference_masks_unsupported_or_multiple_rhythms(codes):
    assert rhythm_reference(codes) == -1


def test_selection_uses_fixed_hash_order_and_excludes_identity_and_signal():
    rows = [dict(ecg_id=f'JS{i:05}', signal_sha256=str(i), rhythm_target=7) for i in range(8)]
    selected = select_records(rows, {'JS00000'}, {'1'}, cap=3)
    assert len(selected) == 3
    assert selected == select_records(list(reversed(rows)), {'JS00000'}, {'1'}, cap=3)
    assert all(r['ecg_id'] not in {'JS00000', 'JS00001'} for r in selected)


def test_pcgrad_projects_only_conflicting_shared_coordinates():
    a = [torch.tensor([1., 0.]), torch.tensor([2.])]
    b = [torch.tensor([-1., 1.]), torch.tensor([3.])]
    out, stats = project_shared_gradients(a, b, [True, False])
    torch.testing.assert_close(out[0], torch.tensor([.5, 1.5]))
    torch.testing.assert_close(out[1], torch.tensor([5.]))
    assert stats['conflict'] and stats['cosine'] < 0


def test_pcgrad_zero_and_nonconflicting_gradients_are_safe():
    out, stats = project_shared_gradients([torch.tensor([1.]), None], [torch.tensor([2.]), torch.tensor([3.])], [True, False])
    torch.testing.assert_close(out[0], torch.tensor([3.]))
    torch.testing.assert_close(out[1], torch.tensor([3.]))
    assert not stats['conflict']
    assert np.isfinite(stats['cosine'])
    with pytest.raises(ValueError):
        project_shared_gradients([torch.tensor([float('nan')])], [torch.ones(1)], [True])


def test_header_verification_rejects_omitted_unsupported_code():
    import hashlib
    import json
    raw = b'JS00001 12 500 5000\n# Dx: 59118001,426648003\n'
    row = {'ecg_id': 'JS00001', 'header_sha256': hashlib.sha256(raw).hexdigest(),
           'source_snomed_codes': json.dumps(['59118001']), 'rhythm_target': 5}
    with pytest.raises(ValueError, match='Source diagnosis'):
        validate_header_reference(row, raw)
    row.update(source_snomed_codes=json.dumps(['59118001', '426648003']), rhythm_target=-1)
    validate_header_reference(row, raw)
    with pytest.raises(ValueError, match='header hash'):
        validate_header_reference(row, raw + b' ')


def test_header_identity_mismatch_is_permitted_only_for_excluded_census_row():
    import hashlib
    raw = b'S00001 12 500 5000\n#Dx: 10370003\n'
    row = {'ecg_id': 'JS00001', 'header_sha256': hashlib.sha256(raw).hexdigest(),
           'source_snomed_codes': '["10370003"]', 'rhythm_target': -1}
    assert validate_header_reference(row, raw) is False
    row['rhythm_target'] = 5
    with pytest.raises(ValueError, match='Source identity'):
        validate_header_reference(row, raw)
