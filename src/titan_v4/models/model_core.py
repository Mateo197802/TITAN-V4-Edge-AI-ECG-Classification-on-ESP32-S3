"""
TITAN V4 LOWCOST - Core Configuration and Signal Processing
Incorpora el procesamiento de señales de V5.2.3 en una arquitectura liviana.
"""
import torch
import torch.nn as nn
import numpy as np
from scipy.signal import butter, lfilter

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN CLÍNICA GLOBAL (Heredado de V5)
# ─────────────────────────────────────────────────────────────────────────────
class Config:
    # Hyperparameters
    SR = 125                 # Hz (Standard V5 — Resampling Polifásico Universal)
    DURATION = 10            # seconds (Standard AHA rhythm strip)
    SAMPLES = SR * DURATION  # 1250 puntos exactos
    BATCH_SIZE = 32          # Smaller for V4 LowCost
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Clinical Thresholds (Safety Net)
    HR_BRADYCARDIA = 50      # bpm
    HR_TACHYCARDIA = 100     # bpm
    HR_SVT_MIN     = 150     # bpm
    QRS_WIDE_MS    = 120     # ms
    QRS_NARROW_MS  = 120     # ms (<=)

    # Simplified AAMI extended map (35 classes)
    # 0 = Normal, 1-34 = Arritmias
    AAMI_EXTENDED_MAP = {
        'N': 0, 'L': 1, 'R': 2, 'V': 3, 'A': 4, 'F': 5, 'f': 6, '!': 7,
        'E': 8, 'j': 9, 'a': 10, 'J': 11, 'S': 12, 'e': 13, 'n': 14,
        'x': 15, 'r': 16, 'AFIB': 17, 'AFL': 18, 'SVT': 19, 'VT': 20,
        'VF': 21, 'VFL': 22, 'NODAL': 23, 'STACH': 24, 'SBRAD': 25,
        'PAC': 26, 'PVC': 27, 'LBBB': 28, 'RBBB': 29, '1AVB': 30,
        '2AVB': 31, '3AVB': 32, 'WPW': 33, 'PACE': 34
    }
    NUM_CLASSES_RHYTHM = len(AAMI_EXTENDED_MAP)

# ─────────────────────────────────────────────────────────────────────────────
# PROCESAMIENTO DE SEÑALES (Heredado de V5)
# ─────────────────────────────────────────────────────────────────────────────
class SignalProcessor:
    @staticmethod
    def butter_bandpass(lowcut, highcut, fs, order=3):
        # Butterworth order 3 for smoother clinical bounds
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a

    @classmethod
    def apply_clinical_filter(cls, data, fs=Config.SR):
        """0.5 - 45 Hz clinical bandpass (AHA Guidelines)"""
        b, a = cls.butter_bandpass(0.5, 45.0, fs, order=3)
        return lfilter(b, a, data)

    @staticmethod
    def normalize_signal(signal):
        """Z-score normalization to ensure consistent scale."""
        if len(np.unique(signal)) <= 1:
            return signal
        mean_val = np.mean(signal)
        std_val = np.std(signal)
        return (signal - mean_val) / (std_val + 1e-8)

# ─────────────────────────────────────────────────────────────────────────────
# BLOQUES NEURONALES (Heredado de V5 con SiLU)
# ─────────────────────────────────────────────────────────────────────────────
class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for 1D channel attention"""
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.silu = nn.SiLU()  # BUG4 FIX
        self.fc2 = nn.Linear(channels // reduction, channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _ = x.size()
        y = x.mean(dim=2)  # Global Average Pool 1D
        y = self.fc1(y)
        y = self.silu(y)
        y = self.fc2(y)
        y = self.sigmoid(y).view(b, c, 1)
        return x * y

class ResBlock(nn.Module):
    """Residual Block for 1D convolutions con SiLU"""
    def __init__(self, in_channels, out_channels, stride=1, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=7, stride=stride, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.silu = nn.SiLU()  # BUG4 FIX
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.se = SEBlock(out_channels)
        
        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        identity = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.silu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)
        out += identity
        out = self.silu(out)
        return out
