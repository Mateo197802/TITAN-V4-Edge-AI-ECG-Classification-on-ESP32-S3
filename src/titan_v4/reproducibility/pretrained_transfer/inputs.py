from __future__ import annotations

import ast
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import hashlib
from pathlib import Path, PurePosixPath

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, medfilt
import wfdb

from titan_v4.reproducibility.joint_refinement import pathology_targets
from titan_v4.reproducibility.physionet import preprocess_primary9_signal, FILTER_CONTEXT_SECONDS, TARGET_FS, TARGET_SAMPLES
from titan_v4.reproducibility.refinement import read_manifest, sha256
from .contracts import CONTRACTS, LEADS, PRIMARY9_CLASSES, PairedRecord, TensorPair


def _lead_name(value: str) -> str:
    return ''.join(c for c in value.upper() if c.isalnum())


def physical_mv(signals: np.ndarray, units: list[str], *, allow_nonfinite: bool = False) -> np.ndarray:
    data = np.asarray(signals, dtype=np.float64)
    if (data.ndim != 2 or data.shape[1] != len(units) or not len(data)
            or (not allow_nonfinite and not np.isfinite(data).all())):
        raise ValueError('Invalid or nonfinite physical signals')
    scales = {'mv': 1.0, 'uv': .001, 'v': 1000.0}
    factors = []
    for unit in units:
        name = str(unit).strip().lower().replace('\u00b5', 'u').replace('\u03bc', 'u')
        if name not in scales:
            raise ValueError('Unsupported physical signal units')
        factors.append(scales[name])
    return data * np.asarray(factors)[None, :]


def preprocess_founder(signals, names, sample_rate, units, contract):
    if contract not in CONTRACTS or sample_rate != 500:
        raise ValueError('Unregistered teacher contract or sampling frequency')
    names = [_lead_name(str(name)) for name in names]
    if len(names) != 12 or len(set(names)) != 12 or set(names) != {_lead_name(v) for v in LEADS}:
        raise ValueError('Missing or duplicate named teacher leads')
    data = physical_mv(signals, list(units))
    if data.shape[1] != len(names):
        raise ValueError('Physical signals and named leads differ')
    ordered = list(LEADS)
    if contract.startswith('heed_'):
        ordered[4], ordered[5] = ordered[5], ordered[4]
    data = data[:5000, [names.index(_lead_name(v)) for v in ordered]].T
    flags = []
    if data.shape[1] < 5000:
        flags.append('right_padding')
        data = np.pad(data, ((0, 0), (0, 5000 - data.shape[1])))
    if np.std(data) < 1e-8:
        flags.append('constant_signal')
    if contract.endswith('_filtered'):
        b, a = iirnotch(50, 30, fs=500)
        data = filtfilt(b, a, data, axis=1)
        b, a = butter(4, [0.67, 40], btype='bandpass', fs=500)
        data = filtfilt(b, a, data, axis=1)
        data -= np.stack([medfilt(channel, kernel_size=201) for channel in data])
    data = (data - data.mean()) / (data.std() + 1e-8)
    if not np.isfinite(data).all():
        raise ValueError('Nonfinite preprocessed teacher signal')
    return data.astype(np.float32), tuple(flags)


def _paths(record: PairedRecord, cache: Path) -> tuple[Path, Path]:
    parts = record.key.split('/')
    path = PurePosixPath(record.record_path)
    signal = PurePosixPath(record.signal_filename)
    if (len(parts) != 3 or any(not v or v in ('.', '..') for v in parts)
            or path.is_absolute() or '..' in path.parts or not path.parts or path.suffix
            or '\\' in record.record_path or signal.name != str(signal)
            or signal.stem != path.name or signal.suffix not in ('.dat', '.mat')):
        raise ValueError('Unsafe source record path')
    release = record.source_release or ('challenge-2021' if record.dataset in ('cpsc_2018', 'georgia') else record.dataset)
    if release not in ('ptb-xl', 'challenge-2021', 'ecg-arrhythmia'):
        raise ValueError('Unsupported source cache path')
    root = cache.resolve()
    base = root / release / record.version / path
    header, signal_path = base.with_suffix('.hea'), base.parent / signal
    if not all(p.resolve().is_relative_to(root) for p in (header, signal_path)):
        raise ValueError('Source path escapes cache')
    return header, signal_path


def read_source(record: PairedRecord, cache: Path):
    header, signal = _paths(record, cache)
    for file, expected in ((header, record.header_sha256), (signal, record.signal_sha256)):
        if sha256(file) != expected.lower():
            raise ValueError('Source byte hash mismatch: ' + file.name)
    meta = wfdb.rdheader(str(header.with_suffix('')))
    if set(meta.file_name) != {signal.name}:
        raise ValueError('Header references unexpected source files')
    required = int((TARGET_SAMPLES / TARGET_FS + FILTER_CONTEXT_SECONDS) * meta.fs)
    return wfdb.rdrecord(str(header.with_suffix('')), sampto=min(meta.sig_len, required))


def validated_wfdb16_zero_fill(physical, digital, formats):
    data = np.asarray(physical, dtype=np.float64)
    digital = np.asarray(digital)
    if (data.ndim != 2 or data.shape != digital.shape or len(formats) != data.shape[1]
            or any(str(fmt) != '16' for fmt in formats) or np.isinf(data).any()):
        raise ValueError('Unregistered or nonfinite WFDB missing-sample input')
    if not np.array_equal(np.isnan(data), digital == -32768):
        raise ValueError('NaNs do not match verified WFDB16 missing-sample sentinels')
    return np.where(np.isnan(data), 0.0, data)


def _teacher_source(record, raw, cache, missing_policy):
    if missing_policy not in ('reject', 'wfdb16_zero_fill'):
        raise ValueError('Unregistered missing-sample policy')
    missing = int(np.isnan(raw.p_signal).sum())
    data = raw.p_signal
    flags = ()
    if missing and missing_policy == 'wfdb16_zero_fill':
        header, _ = _paths(record, cache)
        digital = wfdb.rdrecord(str(header.with_suffix('')), physical=False, sampto=len(data))
        data = validated_wfdb16_zero_fill(data, digital.d_signal, digital.fmt)
        flags = ('wfdb16_missing_zero_fill',)
    return data, flags, missing


def load_pair(record: PairedRecord, cache: Path, contract: str, *, missing_policy: str = 'reject') -> TensorPair:
    if record.window_start != 0 or record.window_samples != 5000:
        raise ValueError('First-window pairing contract differs')
    raw = read_source(record, cache)
    data, repair_flags, missing = _teacher_source(record, raw, cache, missing_policy)
    teacher, flags = preprocess_founder(data, raw.sig_name, raw.fs, raw.units, contract)
    student = preprocess_primary9_signal(data, raw.sig_name, raw.fs)
    if hashlib.sha256(student.tobytes()).hexdigest() != record.student_tensor_sha256.lower():
        raise ValueError('Original student tensor hash mismatch')
    return TensorPair(teacher, student, record.key, contract, flags + repair_flags, missing)


def load_student_tensor(record: PairedRecord, cache: Path) -> tuple[np.ndarray, tuple[str, ...]]:
    if record.window_start != 0 or record.window_samples != 5000:
        raise ValueError('Original student first-window contract differs')
    raw = read_source(record, cache)
    legacy = record.role in ('historical', 'tracking', 'train_excluded')
    physical_mv(raw.p_signal, list(raw.units), allow_nonfinite=legacy)
    data = raw.p_signal
    flags = ('legacy_nonfinite_zero_fill',) if not np.isfinite(data).all() else ()
    student = preprocess_primary9_signal(data, raw.sig_name, raw.fs)
    if hashlib.sha256(student.tobytes()).hexdigest() != record.student_tensor_sha256.lower():
        raise ValueError('Original student tensor hash mismatch')
    return student, flags


def assert_disjoint(train: list[PairedRecord], excluded: list[PairedRecord]) -> None:
    if any(r.role != 'train' or r.window_start != 0 or r.window_samples != 5000 for r in train):
        raise ValueError('Training role/window must be frozen before window derivation')
    for field in ('key', 'signal_sha256', 'student_tensor_sha256'):
        values = [getattr(r, field).lower() for r in train if getattr(r, field)]
        if len(values) != len(set(values)):
            raise ValueError('Training duplicate ' + field)
    for field in ('key', 'patient_token', 'signal_sha256', 'student_tensor_sha256'):
        a = {str(getattr(r, field)).lower() for r in train if getattr(r, field)}
        b = {str(getattr(r, field)).lower() for r in excluded if getattr(r, field)}
        if a & b:
            raise ValueError('Training/exclusion overlap: ' + field)


def apply_fitting_exclusions(train, excluded):
    fields = ('key', 'patient_token', 'signal_sha256', 'student_tensor_sha256')
    identities = {field: {str(getattr(r, field)).lower() for r in excluded if getattr(r, field)} for field in fields}
    kept, removed = [], []
    for r in train:
        reasons = tuple(field for field in fields if getattr(r, field) and str(getattr(r, field)).lower() in identities[field])
        if reasons:
            removed.append(replace(r, role='train_excluded', exclusion_reasons=reasons))
        else:
            kept.append(r)
    assert_disjoint(kept, excluded)
    return kept, removed


def _patient(rid: str, metadata: dict[str, dict]) -> str | None:
    if rid == '141' and rid not in metadata:
        rid = '2509'
    if rid not in metadata:
        raise ValueError('Unresolved native patient metadata')
    return hashlib.sha256(('ptb-xl:' + metadata[rid]['patient_id']).encode()).hexdigest()


def _convert_row(row, role, metadata):
    dataset, version, rid = row['dataset'], row['version'], row['ecg_id']
    native = metadata.get(str(int(rid))) if dataset == 'ptb-xl' else None
    if dataset == 'ptb-xl' and native is None:
        raise ValueError('Native record absent from pinned metadata')
    scores = ast.literal_eval(native['scp_codes']) if native else {}
    targets, mask = pathology_targets(dataset, scores)
    y = int(row['rhythm_target']) if 'rhythm_target' in row else PRIMARY9_CLASSES.index(row['reference_label'])
    if y not in range(-1, 9):
        raise ValueError('Rhythm reference outside frozen class order')
    token = _patient(rid, metadata) if native else (row.get('patient_token') or None)
    if row.get('patient_token') and row['patient_token'] != token:
        raise ValueError('Patient token differs from pinned metadata')
    if native and role == 'train' and int(native['strat_fold']) > 8:
        raise ValueError('Validation patient fold included in fitting')
    for j, label in enumerate(('IMI','ASMI','LVH','ISC_','ISCAL','NST_','ILMI','AMI','ALMI','LAE')):
        if 'target_' + label in row and float(row['target_' + label]) != float(targets[j]):
            raise ValueError('Archived native pathology reference differs')
    return PairedRecord(f'{dataset}/{version}/{rid}', token, role, row['record_path'], row['signal_filename'],
                        row['header_sha256'].lower(), row['signal_sha256'].lower(), 0, 5000, y,
                        tuple(float(v) for v in targets), tuple(bool(v) for v in mask),
                        row['tensor_sha256'].lower(), row['source_release'])


def build_pair_manifest(root: Path, config: dict) -> list[PairedRecord]:
    paths = [p for values in config['manifests'].values() for p in values]
    paths += [config[k] for k in ('historical_sources', 'historical_hashes', 'historical_labels')]
    for relative in paths:
        if sha256(root / relative) != config.get('manifest_sha256', {}).get(relative):
            raise ValueError('Frozen reference manifest hash differs')
    metadata_path = root / config['metadata']
    if sha256(metadata_path) != config['metadata_sha256']:
        raise ValueError('Pinned native metadata hash differs')
    metadata = {str(int(r['ecg_id'])): r for r in read_manifest(metadata_path)}
    result = []
    for role, paths in config['manifests'].items():
        for rel in paths:
            result.extend(_convert_row(r, role, metadata) for r in read_manifest(root / rel))
    historical = read_manifest(root / config['historical_sources'])
    sources = {r['relative_path']: r['sha256'].lower() for r in read_manifest(root / config['historical_hashes'])}
    labels = {(r['source'].casefold(), r['record_id'].casefold()): r for r in read_manifest(root / config['historical_labels'])}
    for r in historical:
        history = labels[(r['source_label'].casefold(), r['record_id'].casefold())]
        path = r['record_path']
        signal_name = PurePosixPath(path).name + '.mat'
        token = _patient(str(int(r['record_id'][2:])), metadata) if r['dataset'] == 'ptb-xl' else None
        item = PairedRecord(f"{r['dataset']}/1.0.3/{r['record_id']}", token, 'historical', path,
                            signal_name, sources[path + '.hea'], sources[str(PurePosixPath(path).parent / signal_name)],
                            0, 5000, PRIMARY9_CLASSES.index(history['final_rhythm_label']),
                            (0.,) * 10, (False,) * 10, '', 'challenge-2021',
                            PRIMARY9_CLASSES.index(history['original_rhythm_label']))
        raw = read_source(item, root / config['data_root'])
        physical_mv(raw.p_signal, list(raw.units), allow_nonfinite=True)
        tensor = preprocess_primary9_signal(raw.p_signal, raw.sig_name, raw.fs)
        result.append(replace(item, student_tensor_sha256=hashlib.sha256(tensor.tobytes()).hexdigest()))
    candidates = [r for r in result if r.role == 'train']
    if len(candidates) != config['expected_training_union']:
        raise ValueError('Original training union differs')
    held = [r for r in result if r.role != 'train']
    kept, removed = apply_fitting_exclusions(candidates, held)
    result = held + kept + removed
    counts = Counter(r.role for r in result)
    if dict(counts) != config['expected_role_counts']:
        raise ValueError('Frozen manifest role counts differ')
    train = [r for r in result if r.role == 'train']
    if sum(any(r.path_mask) for r in train) != config['expected_pathology_training']:
        raise ValueError('Native pathology support differs')
    assert_disjoint(train, [r for r in result if r.role != 'train'])
    return sorted(result, key=lambda r: (r.role, r.key))


def verify_archived_inputs(root: Path, config: dict) -> dict[str, str]:
    verified = {}
    for relative, expected in config['archived_input_sha256'].items():
        actual = sha256(root / relative)
        if actual != expected:
            raise ValueError('Original archived input hash differs: ' + Path(relative).name)
        verified[relative] = actual
    return verified


def verify_source_pairs(rows, cache, missing_policy, *, workers=4, progress=None):
    if not 1 <= workers <= 8:
        raise ValueError('Source verification worker count outside bounded range')

    def verify(row):
        if row.role in ('historical', 'tracking', 'train_excluded'):
            _, flags = load_student_tensor(row, cache)
            missing = int(np.isnan(read_source(row, cache).p_signal).sum()) if flags else 0
        else:
            pair = load_pair(row, cache, 'canonical_unfiltered', missing_policy=missing_policy)
            flags, missing = pair.quality_flags, pair.missing_samples
        return {'key': row.key, 'role': row.role, 'flags': flags, 'missing_samples': missing, 'verified': True}

    result = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for index, receipt in enumerate(pool.map(verify, rows), 1):
            result.append(receipt)
            if progress is not None:
                progress(index)
    return result
