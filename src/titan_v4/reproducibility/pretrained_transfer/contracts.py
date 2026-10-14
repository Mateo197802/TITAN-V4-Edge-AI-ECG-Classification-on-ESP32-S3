from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from titan_v4.data.primary9_registry import PRIMARY9_CLASSES
from titan_v4.reproducibility.joint_refinement import PATHOLOGY_CLASSES, PRIMARY5_INDEX

LEADS = ('I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6')
CONTRACTS = ('canonical_unfiltered', 'canonical_filtered', 'heed_unfiltered', 'heed_filtered')
SEEDS = (20261014, 20261015, 20261016)
STUDENT_SHA256 = '72d3ece9b00e844d6a94f207f4ced71af0af69c59cef66b625031ff92b2f965b'
FOUNDER_SHA256 = 'ee199f3781f4ae1f732973267f003da0a759ea12bddb0dd28a77faa60aca7997'


@dataclass(frozen=True)
class PairedRecord:
    key: str
    patient_token: str | None
    role: str
    record_path: str
    signal_filename: str
    header_sha256: str
    signal_sha256: str
    window_start: int
    window_samples: int
    rhythm_label: int
    path_targets: tuple[float, ...]
    path_mask: tuple[bool, ...]
    student_tensor_sha256: str
    source_release: str = ''
    original_rhythm_label: int | None = None
    exclusion_reasons: tuple[str, ...] = ()

    @property
    def dataset(self) -> str:
        return self.key.split('/')[0]

    @property
    def version(self) -> str:
        return self.key.split('/')[1]


@dataclass(frozen=True)
class TensorPair:
    teacher: np.ndarray
    student: np.ndarray
    key: str
    contract: str
    quality_flags: tuple[str, ...] = ()
    missing_samples: int = 0
