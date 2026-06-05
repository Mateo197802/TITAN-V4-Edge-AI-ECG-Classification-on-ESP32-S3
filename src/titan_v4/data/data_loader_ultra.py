"""
Clinical ECG Dataset Loader (TITAN V4/V5) — Versión Producción
=============================================================
data_loader_ultra.py — TITAN V4 Lite — Pipeline de Datos Clínicos

Pipeline DSP completo para entrenamiento con datos ECG reales de PhysioNet:
  1. Lectura WFDB (.hea/.mat) con CACHÉ LRU inteligente
  2. Extracción de derivadas triásicas (I, II, V2)
  3. Filtro Butterworth pasa-banda 0.5-45 Hz (AHA/ESC 2023)
  4. Resampling polifásico a 125 Hz (scipy.signal.resample_poly)
  5. Sliding Window 10s con stride 5s (50% overlap)
  6. ECG Augmentation (6 tipos de ruido clínico) + Baseline Wander + Channel Dropout)
  6. Z-Score Normalization DESPUÉS de augmentación (BUG3 FIX)
  7. Extracción obligatoria de biometría (Edad, Sexo) desde cabeceras .hea
"""
import os
import glob
import json
import torch
from torch.utils.data import Dataset
import wfdb
import numpy as np
from scipy.signal import butter, lfilter, resample_poly
from math import gcd
from label_registry import (
    NUM_RHYTHM_CLASSES,
    SNOMED_TO_RHYTHM,
    normalize_record_path,
    parse_dx_codes_from_header,
    primary_label_from_codes,
)
from morphology_features import MORPHOLOGY_FEATURE_DIM, normalized_morphology_features

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTES CLÍNICAS
# ─────────────────────────────────────────────────────────────────────────────
TARGET_FS = 125          # Hz — Frecuencia destino universal (V5 estándar)
TARGET_DURATION = 10     # segundos (AHA rhythm strip estándar)
TARGET_LENGTH = TARGET_FS * TARGET_DURATION   # 1250 puntos exactos
NUM_LEADS = 6            # Derivadas Frontales: I, II, III, aVR, aVL, aVF
FILTER_CONTEXT_SEC = 2   # margen para filtrar/resamplear sin transitorios de borde

# Mapeo SNOMED-CT centralizado. No existe fallback silencioso a NSR.
SNOMED_MAPPING = dict(SNOMED_TO_RHYTHM)

# Mapeo Sexo
SEX_MAPPING = {'Male': 1.0, 'Female': 0.0, 'male': 1.0, 'female': 0.0, 'M': 1.0, 'F': 0.0, 'Unknown': 0.5}


def virtual_window_key(source_record: str, window_start: int) -> str:
    return f"virtual::{normalize_record_path(source_record)}|w:{int(window_start)}"


def _resolve_virtual_source(sidecar_path: str, source_record: str) -> str:
    if os.path.isabs(source_record):
        return source_record
    sidecar_dir = os.path.dirname(sidecar_path)
    data_root = os.path.dirname(sidecar_dir) if os.path.basename(sidecar_dir) == "virtual_windows" else sidecar_dir
    return os.path.join(data_root, source_record)


def load_virtual_windows(data_dirs):
    windows = []
    for data_dir in data_dirs:
        if not os.path.isdir(data_dir):
            continue
        pattern = os.path.join(data_dir, "**", "*_window_label_map.json")
        for sidecar_path in glob.glob(pattern, recursive=True):
            try:
                with open(sidecar_path, "r", encoding="utf-8") as handle:
                    payload = json.load(handle)
            except (OSError, json.JSONDecodeError):
                continue
            for _window_id, meta in (payload.get("windows", {}) or {}).items():
                if not isinstance(meta, dict):
                    continue
                source_record = meta.get("source_record")
                if not source_record or "label" not in meta:
                    continue
                source_path = _resolve_virtual_source(sidecar_path, str(source_record))
                start = int(meta.get("window_start", 0))
                label = int(meta["label"])
                if label < 0 or label >= NUM_RHYTHM_CLASSES:
                    continue
                key = virtual_window_key(source_path, start)
                windows.append((source_path, start, key, label))
    return windows


def _lead_token(name: str) -> str:
    return "".join(ch for ch in str(name).upper() if ch.isalnum())


LEAD_ALIASES = {
    "I": ("I", "ECG1"),
    "II": ("II", "MLII", "ML2", "ECG2", "ECG"),
    "III": ("III", "MLIII", "ML3"),
    "AVR": ("AVR",),
    "AVL": ("AVL",),
    "AVF": ("AVF",),
}

NON_ECG_TOKENS = ("RESP", "PLETH", "ABP", "BP", "PRESS", "PAP", "CVP", "ART")


def _is_ecg_like_lead(name: str) -> bool:
    token = _lead_token(name)
    if not token or any(bad in token for bad in NON_ECG_TOKENS):
        return False
    if token in {"I", "II", "III", "AVR", "AVL", "AVF", "MLI", "MLII", "MLIII", "ECG"}:
        return True
    if token.startswith("ECG") or token.startswith("ML"):
        return True
    if token.startswith("V") and token[1:].isdigit():
        return True
    return False


def extract_target_leads(signals, sig_names, target_leads):
    """Extract target ECG leads with case-insensitive aliases and ECG fallbacks."""
    if signals is None or len(np.shape(signals)) != 2 or np.shape(signals)[1] == 0:
        return np.zeros((len(target_leads), 0), dtype=np.float32)

    token_to_index = {}
    for idx, name in enumerate(sig_names or []):
        token = _lead_token(name)
        if token and token not in token_to_index:
            token_to_index[token] = idx

    ecg_indices = [
        idx
        for idx, name in enumerate(sig_names or [])
        if idx < signals.shape[1] and _is_ecg_like_lead(name)
    ]
    if not ecg_indices:
        ecg_indices = list(range(signals.shape[1]))

    extracted = []
    fallback_pos = 0
    for lead in target_leads:
        target_token = _lead_token(lead)
        candidates = (target_token, *LEAD_ALIASES.get(target_token, ()))
        selected_idx = None
        for candidate in candidates:
            idx = token_to_index.get(_lead_token(candidate))
            if idx is not None and idx < signals.shape[1] and _is_ecg_like_lead(sig_names[idx]):
                selected_idx = idx
                break
        if selected_idx is None and ecg_indices:
            selected_idx = ecg_indices[fallback_pos % len(ecg_indices)]
            fallback_pos += 1

        if selected_idx is None:
            channel = np.zeros(signals.shape[0], dtype=np.float32)
        else:
            channel = signals[:, selected_idx].astype(np.float32)
        extracted.append(channel)

    return np.stack(extracted, axis=0).astype(np.float32)

# ─────────────────────────────────────────────────────────────────────────────
# FILTRO BUTTERWORTH (AHA/ESC 2023)
# ─────────────────────────────────────────────────────────────────────────────
def butterworth_bandpass(signal_1d, fs, lowcut=0.5, highcut=45.0, order=3):
    """Filtro pasa-banda clínico 0.5–45 Hz según guías AHA/ESC."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = min(highcut / nyq, 0.99)  # Protección si fs es muy bajo
    b, a = butter(order, [low, high], btype='band')
    return lfilter(b, a, signal_1d)

# ─────────────────────────────────────────────────────────────────────────────
# RESAMPLING POLIFÁSICO
# ─────────────────────────────────────────────────────────────────────────────
def resample_to_target(signal_1d, original_fs, target_fs=TARGET_FS):
    """
    Resampling polifásico exacto usando scipy.signal.resample_poly.
    Convierte cualquier frecuencia de muestreo (100, 250, 257, 300, 360, 500, 1000 Hz)
    a la frecuencia destino estándar de 125 Hz.
    """
    if original_fs == target_fs:
        return signal_1d
    
    # Calculamos la razón de decimación/interpolación
    g = gcd(int(target_fs), int(original_fs))
    up = int(target_fs) // g
    down = int(original_fs) // g
    
    return resample_poly(signal_1d, up, down)

# ─────────────────────────────────────────────────────────────────────────────
# ECG AUGMENTOR (8 tipos de ruido clínico)
# ─────────────────────────────────────────────────────────────────────────────
class ECGAugmentor:
    """
    Data Augmentation específica para señales ECG.
    Se aplica ANTES de la normalización Z-Score (BUG3 FIX).
    """
    
    @staticmethod
    def gaussian_noise(signal, snr_db=25):
        """Ruido gaussiano blanco con SNR controlado."""
        power = np.mean(signal ** 2)
        noise_power = power / (10 ** (snr_db / 10))
        noise = np.random.normal(0, np.sqrt(noise_power), signal.shape)
        return signal + noise
    
    @staticmethod
    def baseline_wander(signal, fs=TARGET_FS, amplitude=0.15):
        """Simula artefacto respiratorio (Baseline Wander, ~0.3 Hz)."""
        t = np.arange(signal.shape[-1]) / fs
        freq = np.random.uniform(0.1, 0.5)
        phase = np.random.uniform(0, 2 * np.pi)
        wander = amplitude * np.sin(2 * np.pi * freq * t + phase)
        return signal + wander
    
    @staticmethod
    def amplitude_scaling(signal, scale_range=(0.8, 1.2)):
        """Escala la amplitud simulando variaciones en el contacto del electrodo."""
        scale = np.random.uniform(*scale_range)
        return signal * scale
    
    @staticmethod
    def time_shift(signal, max_shift_samples=15):
        """Desplazamiento temporal aleatorio con relleno circular."""
        shift = np.random.randint(-max_shift_samples, max_shift_samples + 1)
        return np.roll(signal, shift, axis=-1)
    
    @staticmethod
    def powerline_noise(signal, fs=TARGET_FS, freq=60.0, amplitude=0.05):
        """Interferencia de línea eléctrica (50/60 Hz)."""
        t = np.arange(signal.shape[-1]) / fs
        noise = amplitude * np.sin(2 * np.pi * freq * t)
        return signal + noise
    
    @staticmethod
    def channel_dropout(signal, p=0.1):
        """Simula pérdida de contacto de un electrodo (pone un canal a 0)."""
        if np.random.random() < p:
            channel = np.random.randint(0, signal.shape[0])
            signal[channel, :] = 0.0
        return signal

    @staticmethod
    def time_mask(signal, max_width=80, p=0.2):
        """Enmascara una banda temporal corta (cutout) en todos los canales."""
        if np.random.random() >= p:
            return signal
        width = int(np.random.randint(10, max_width + 1))
        start = int(np.random.randint(0, max(1, signal.shape[1] - width)))
        signal[:, start:start + width] = 0.0
        return signal

    @staticmethod
    def per_lead_gain(signal, gain_range=(0.9, 1.1), p=0.25):
        """Aplica ganancia distinta por derivada (simula impedancias distintas)."""
        if np.random.random() >= p:
            return signal
        gains = np.random.uniform(gain_range[0], gain_range[1], size=(signal.shape[0], 1)).astype(np.float32)
        return signal * gains

    @staticmethod
    def rr_warp(signal, fs=TARGET_FS, max_jitter_ms=40.0, p=0.15):
        """
        Semi-sintético (B): warping temporal guiado por R-peaks (usa gqrs_detect en lead II).
        Mantiene longitud total fija.
        """
        if np.random.random() >= p:
            return signal
        try:
            import wfdb.processing as wfproc
        except Exception:
            return signal

        if signal.shape[1] < 200:
            return signal
        lead_idx = 1 if signal.shape[0] > 1 else 0
        x = signal[lead_idx].astype(np.float64)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        try:
            rpeaks = wfproc.gqrs_detect(sig=x, fs=fs)
        except Exception:
            return signal
        if rpeaks is None:
            return signal
        rpeaks = np.asarray(rpeaks, dtype=np.int64)
        rpeaks = rpeaks[(rpeaks > 10) & (rpeaks < signal.shape[1] - 10)]
        if rpeaks.size < 3:
            return signal

        L = int(signal.shape[1])
        knots = np.concatenate(([0], rpeaks, [L - 1])).astype(np.int64)
        # Random jitter for internal knots.
        max_jitter = int(round((max_jitter_ms / 1000.0) * fs))
        if max_jitter < 1:
            return signal
        jitter = np.random.randint(-max_jitter, max_jitter + 1, size=(knots.size,), dtype=np.int64)
        jitter[0] = 0
        jitter[-1] = 0
        warped = knots + jitter

        # Enforce strict monotonicity to build a valid time-warp mapping.
        min_sep = 5
        warped[0] = 0
        warped[-1] = L - 1
        for i in range(1, warped.size - 1):
            if warped[i] <= warped[i - 1] + min_sep:
                warped[i] = warped[i - 1] + min_sep
        for i in range(warped.size - 2, 0, -1):
            if warped[i] >= warped[i + 1] - min_sep:
                warped[i] = warped[i + 1] - min_sep
        if not np.all(np.diff(warped) > 0):
            return signal

        # Build mapping: new_t -> original_t via piecewise linear inverse.
        # original knots (knots) map to warped time (warped). We want to sample original at times corresponding to each new t.
        new_t = np.arange(L, dtype=np.float64)
        orig_t = np.interp(new_t, warped.astype(np.float64), knots.astype(np.float64))

        out = np.empty_like(signal, dtype=np.float32)
        src_t = np.arange(L, dtype=np.float64)
        for ch in range(signal.shape[0]):
            y = signal[ch].astype(np.float64)
            y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
            out[ch] = np.interp(orig_t, src_t, y).astype(np.float32)
        return out
    
    @staticmethod
    def apply(signal, p=0.65):
        """
        Aplica aumentación con probabilidad p.
        signal shape: (num_leads, seq_len) — numpy array
        """
        if np.random.random() > p:
            return signal  # Sin augmentación
        
        # Seleccionar aleatoriamente 1-3 tipos de augmentación
        augs = np.random.choice(
            ['gauss', 'wander', 'scale', 'shift', 'powerline', 'channel_drop', 'time_mask', 'per_lead_gain', 'rr_warp'],
            size=np.random.randint(1, 4),
            replace=False
        )
        
        for aug in augs:
            if aug == 'gauss':
                signal = ECGAugmentor.gaussian_noise(signal)
            elif aug == 'wander':
                signal = ECGAugmentor.baseline_wander(signal)
            elif aug == 'scale':
                signal = ECGAugmentor.amplitude_scaling(signal)
            elif aug == 'shift':
                signal = ECGAugmentor.time_shift(signal)
            elif aug == 'powerline':
                signal = ECGAugmentor.powerline_noise(signal)
            elif aug == 'channel_drop':
                signal = ECGAugmentor.channel_dropout(signal)
            elif aug == 'time_mask':
                signal = ECGAugmentor.time_mask(signal)
            elif aug == 'per_lead_gain':
                signal = ECGAugmentor.per_lead_gain(signal)
            elif aug == 'rr_warp':
                signal = ECGAugmentor.rr_warp(signal)
         
        return signal


# ─────────────────────────────────────────────────────────────────────────────
# DATASET PRINCIPAL CON SLIDING WINDOW
# ─────────────────────────────────────────────────────────────────────────────
class ClinicalECGDataset(Dataset):
    """
    Dataset PyTorch para datos clínicos reales ECG con SLIDING WINDOW.
    
    Registros largos (MIT-BIH 30min, LTDB 24h) se segmentan en múltiples
    ventanas de 10 segundos, multiplicando los datos de entrenamiento.
    
    Ejemplo: 1 registro MIT-BIH de 30 min → 180 ventanas de 10s
    
    Pipeline por ventana:
      .hea/.mat → wfdb.rdrecord → Extracción Triásica (I,II,V2)
      → Butterworth 0.5-45Hz → Resample a 125Hz
      → Sliding Window 10s (1250pts, stride=5s para 50% overlap)
      → ECG Augmentation → Z-Score Normalization
      → Tensor [3, 1250]
    """
    
    # Stride de la ventana deslizante: 5 segundos (50% overlap)
    WINDOW_STRIDE_SEC = 5
    
    def __init__(
        self,
        data_dirs,
        target_length=TARGET_LENGTH,
        target_leads=None,
        augment=True,
        record_label_map=None,
        skip_unlabeled=False,
        return_morphology=False,
        return_window_key=False,
    ):
        if target_leads is None:
            target_leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF']
        
        self.target_length = target_length
        self.target_leads = target_leads
        self.augment = augment
        self.stride = self.WINDOW_STRIDE_SEC * TARGET_FS  # 625 puntos
        self.record_label_map = {
            (str(path) if str(path).startswith("virtual::") else normalize_record_path(path)): int(label)
            for path, label in (record_label_map or {}).items()
        }
        self.skip_unlabeled = bool(skip_unlabeled)
        self.return_morphology = bool(return_morphology)
        self.return_window_key = bool(return_window_key)
        
        
        # ─── Indexación con Sliding Window ──────────────────────────────
        # Cada entrada en self.windows es (record_path, window_start_sample)
        self.windows = []
        self.window_split_keys = []
        self.window_label_overrides = {}
        record_count = 0
        virtual_count = 0

        def append_window(record_base, window_start, split_key=None, label_override=None):
            self.windows.append((record_base, int(window_start)))
            key = split_key or normalize_record_path(record_base)
            self.window_split_keys.append(key)
            if label_override is not None:
                self.window_label_overrides[key] = int(label_override)
        
        for d in data_dirs:
            if not os.path.exists(d):
                print(f"  [WARN] Directorio no encontrado: {d}")
                continue
            
            hea_files = glob.glob(os.path.join(d, "**", "*.hea"), recursive=True)
            db_name = os.path.basename(d)
            db_windows = 0
            
            for hea in hea_files:
                record_base = hea[:-4]
                if self.skip_unlabeled and normalize_record_path(record_base) not in self.record_label_map:
                    continue
                
                # Leer solo la cabecera para saber la duración (sin cargar señal)
                try:
                    header = wfdb.rdheader(record_base)
                    original_fs = header.fs
                    sig_len = header.sig_len
                    
                    # Calcular longitud resampleada a 125Hz
                    resampled_len = int(sig_len * TARGET_FS / original_fs)
                    
                    if resampled_len <= 0:
                        continue
                    
                    # Generar ventanas con sliding window
                    if resampled_len <= target_length:
                        # Registro corto: una sola ventana con padding
                        append_window(record_base, 0)
                        db_windows += 1
                    else:
                        # Registro largo: múltiples ventanas con stride
                        start = 0
                        while start + target_length <= resampled_len:
                            append_window(record_base, start)
                            db_windows += 1
                            start += self.stride
                        # Agregar ventana final si queda un segmento significativo
                        if start < resampled_len and (resampled_len - start) > target_length // 2:
                            append_window(record_base, resampled_len - target_length)
                            db_windows += 1
                except Exception:
                    # Si no podemos leer la cabecera, una ventana genérica
                    append_window(record_base, 0)
                    db_windows += 1
                
                record_count += 1
            
            print(f"  [OK] {db_name}: {len(hea_files)} registros -> {db_windows} ventanas de {TARGET_DURATION}s")

        for record_base, start, key, label in load_virtual_windows(data_dirs):
            if self.skip_unlabeled and key not in self.record_label_map:
                continue
            append_window(record_base, start, split_key=key, label_override=label)
            virtual_count += 1
        if virtual_count:
            print(f"  [OK] virtual_windows: {virtual_count} ventanas anotadas de {TARGET_DURATION}s")
        
        print(f"{'='*60}")
        print(f"  ARCHIVOS ORIGINALES : {record_count}")
        print(f"  VENTANAS TOTALES    : {len(self.windows)} (Sliding Window, stride={self.WINDOW_STRIDE_SEC}s)")
        print(f"  MULTIPLICADOR       : {len(self.windows)/max(record_count,1):.1f}x")
        print(f"  TARGET FS: {TARGET_FS} Hz | VENTANA: {TARGET_DURATION}s | PUNTOS: {target_length}")
        print(f"  DERIVADAS: {self.target_leads}")
        print(f"  AUGMENTATION: {'ACTIVO' if augment else 'INACTIVO'}")
        print(f"{'='*60}")
    
    def __len__(self):
        return len(self.windows)

    def split_key_for_index(self, idx):
        return self.window_split_keys[idx]

    def window_key_for_index(self, idx):
        record_base, window_start = self.windows[idx]
        return virtual_window_key(record_base, int(window_start))

    def label_for_index(self, idx):
        key = self.window_split_keys[idx]
        if key in self.window_label_overrides:
            return int(self.window_label_overrides[key])
        record_base, _window_start = self.windows[idx]
        return self._label_for_record(record_base)
    
    def _label_for_record(self, record_path, window_start=None):
        if window_start is not None:
            key = virtual_window_key(record_path, int(window_start))
            if key in self.window_label_overrides:
                return int(self.window_label_overrides[key])
        key = normalize_record_path(record_path)
        if key in self.record_label_map:
            return int(self.record_label_map[key])

        label = primary_label_from_codes(parse_dx_codes_from_header(record_path), SNOMED_MAPPING)
        if label is not None:
            return int(label)
        raise ValueError(f"Registro sin etiqueta ritmica valida: {record_path}")

    def _make_fallback(self, record_path, window_start=0):
        """Tensor de emergencia cuando un registro está corrupto."""
        base = (
            torch.zeros(NUM_LEADS, self.target_length, dtype=torch.float32),
            torch.tensor(self._label_for_record(record_path), dtype=torch.long),
            torch.tensor([0.5], dtype=torch.float32),
            torch.tensor([0.5, 0.5, 0.25], dtype=torch.float32),
            record_path,
        )
        if self.return_morphology:
            base = (*base, torch.zeros(MORPHOLOGY_FEATURE_DIM, dtype=torch.float32))
        if self.return_window_key:
            base = (*base, virtual_window_key(record_path, int(window_start)))
        return base
    
    def __getitem__(self, idx):
        record_path, window_start = self.windows[idx]
        rhythm_class = self.label_for_index(idx)
        
        try:
            # ─── PASO 1: Lectura Rápida de Cabecera ────────────────────────
            header = wfdb.rdheader(record_path)
            original_fs = header.fs
            sig_len_orig = header.sig_len
            comments = header.comments if header.comments else []
            
            # ─── PASO 2: Cálculo On-Demand de Índices (sampfrom / sampto) ──
            # window_start esta en base a TARGET_FS (125 Hz). Lo pasamos a original_fs.
            window_sampfrom = int(window_start * (original_fs / TARGET_FS))
            window_sampto = window_sampfrom + int((self.target_length / TARGET_FS) * original_fs)
            context_samples = int(FILTER_CONTEXT_SEC * original_fs)
            requested_from = window_sampfrom - context_samples
            requested_to = window_sampto + context_samples
            
            # Ajustar limites si exceden el tamano del archivo. El padding conserva
            # la posicion de la ventana central despues de filtrar y resamplear.
            sampfrom = max(0, requested_from)
            sampto = min(sig_len_orig, requested_to)
            pad_left = max(0, sampfrom - requested_from)
            pad_right = max(0, requested_to - sampto)
            
            if sampfrom >= sig_len_orig or sampto <= 0:
                return self._make_fallback(record_path, window_start)
            
            # ─── PASO 3: Lectura Parcial WFDB con margen de contexto ───────
            record = wfdb.rdrecord(record_path, sampfrom=sampfrom, sampto=sampto)
            signals = record.p_signal
            sig_names = record.sig_name
            
            if signals is None or signals.size == 0:
                return self._make_fallback(record_path, window_start)
                
            # Aplicar padding si la señal era más corta que la ventana
            if pad_left > 0 or pad_right > 0:
                signals = np.pad(signals, ((pad_left, pad_right), (0, 0)), mode='constant')
            
            # ─── PASO 4: Extracción robusta de derivaciones ECG ───────────
            extracted = extract_target_leads(signals, sig_names, self.target_leads)
            extracted = np.nan_to_num(extracted, nan=0.0, posinf=0.0, neginf=0.0)
            
            # ─── PASO 5: Filtro Butterworth 0.5-45 Hz con margen ───────────
            for ch in range(extracted.shape[0]):
                if np.std(extracted[ch]) > 1e-6:
                    extracted[ch] = butterworth_bandpass(extracted[ch], fs=original_fs)
            
            # ─── PASO 6: Resampling a 125 Hz y recorte exacto a 10s ─────────
            resampled_channels = []
            for ch in range(extracted.shape[0]):
                resampled_channels.append(resample_to_target(extracted[ch], original_fs, TARGET_FS))
            
            resampled = np.stack(resampled_channels, axis=0).astype(np.float32)
            crop_start = int(round((window_sampfrom - requested_from) * (TARGET_FS / original_fs)))
            crop_end = crop_start + self.target_length
            final_signal = resampled[:, crop_start:crop_end]
            
            # Asegurar tamaño exacto (1250 pts) por ligeras variaciones matemáticas
            seq_len = final_signal.shape[1]
            if seq_len > self.target_length:
                final_signal = final_signal[:, :self.target_length].copy()
            elif seq_len < self.target_length:
                final_signal = np.pad(final_signal, ((0, 0), (0, self.target_length - seq_len)), mode='constant')
                
        except Exception as e:
            return self._make_fallback(record_path, window_start)
        
        # ─── PASO 7: ECG Augmentation (ANTES de normalización) ─────────
        if self.augment:
            final_signal = ECGAugmentor.apply(final_signal, p=0.65)
        
        # ─── PASO 8: Z-Score Normalization ──────────────────────────────
        mean = np.mean(final_signal, axis=1, keepdims=True)
        std = np.std(final_signal, axis=1, keepdims=True)
        std[std < 1e-6] = 1.0
        final_signal = (final_signal - mean) / std
        
        # ─── PASO 9: Extracción de Bio-Metadata ─────────────────────────
        age = None
        sex = None
        
        for c in comments:
            c_stripped = c.strip()
            if c_stripped.startswith('Age:') or c_stripped.startswith('#Age:'):
                try:
                    age_str = c_stripped.split(':')[1].strip()
                    parsed_age = float(age_str)
                    if np.isfinite(parsed_age) and 0 <= parsed_age <= 120:
                        age = parsed_age
                except (ValueError, IndexError):
                    pass
            elif c_stripped.startswith('Sex:') or c_stripped.startswith('#Sex:'):
                try:
                    s = c_stripped.split(':')[1].strip()
                    mapped_sex = SEX_MAPPING.get(s, None)
                    if mapped_sex in (0.0, 1.0):
                        sex = mapped_sex
                except IndexError:
                    pass
        
        age_norm = np.clip(age / 100.0, 0.0, 1.2) if age is not None else -1.0
        sex_value = float(sex) if sex is not None else -1.0
        
        # ─── PASO 10: Conversión a Tensores PyTorch ─────────────────────
        tensor_x  = torch.tensor(final_signal, dtype=torch.float32)
        tensor_x  = torch.nan_to_num(tensor_x, nan=0.0, posinf=0.0, neginf=0.0)
        tensor_ry = torch.tensor(rhythm_class, dtype=torch.long)
        
        # Quality score heurístico
        quality = 1.0 if not np.all(final_signal == 0) else 0.5
        tensor_q  = torch.tensor([quality], dtype=torch.float32)
        # Channel 0 is legacy BMI kept only for checkpoint compatibility.
        # Channels 1/2 are sex and age; -1 marks missing metadata and is masked in training.
        tensor_bio = torch.tensor([0.25, sex_value, age_norm], dtype=torch.float32)
        tensor_bio = torch.nan_to_num(tensor_bio, nan=-1.0, posinf=-1.0, neginf=-1.0)

        if self.return_morphology:
            morph = normalized_morphology_features(final_signal, fs=TARGET_FS)
            tensor_morph = torch.tensor(morph, dtype=torch.float32)
            tensor_morph = torch.nan_to_num(tensor_morph, nan=0.0, posinf=0.0, neginf=0.0)
            if self.return_window_key:
                return tensor_x, tensor_ry, tensor_q, tensor_bio, record_path, tensor_morph, self.window_key_for_index(idx)
            return tensor_x, tensor_ry, tensor_q, tensor_bio, record_path, tensor_morph

        if self.return_window_key:
            return tensor_x, tensor_ry, tensor_q, tensor_bio, record_path, self.window_key_for_index(idx)
        return tensor_x, tensor_ry, tensor_q, tensor_bio, record_path
