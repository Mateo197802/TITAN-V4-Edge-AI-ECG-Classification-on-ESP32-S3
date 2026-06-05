"""
TITAN V4 — Arquitectura Neuronal Escalada
==========================================
ResNet-1D con Squeeze-and-Excitation (SE) Attention.

Escalada de 3 → 5 ResBlocks para alcanzar exactamente ~1 Millón de parámetros.
Ideal para memoria PSRAM extendida en dispositivos de borde avanzados.

Arquitectura:
  Stem:      Conv1D(6→64, k=15, s=2) + BN + SiLU
  ResBlock1: 64→64   (stride=1, dropout=0.1)
  ResBlock2: 64→96   (stride=2, dropout=0.15)
  ResBlock3: 96→128  (stride=2, dropout=0.2)
  ResBlock4: 128→160 (stride=2, dropout=0.25)
  ResBlock5: 160→192 (stride=2, dropout=0.3)
  Global Average Pool → [192]
  Head Ritmo:     192→96→35 clases
  Head Calidad:   192→64→1 (Sigmoid)
  Head Biometría: 192→64→3 (Age, Sex, BMI)

Parámetros totales: ~1.04 Millones
Ratio datos/params: 570,706 / 1,040,000 ≈ 0.5:1
"""
import torch
import torch.nn as nn
import sys
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
from model_core import ResBlock, Config


class TitanV4Lite(nn.Module):
    """
    Arquitectura TITAN V4 — ResNet-1D Escalada con SE-Attention.
    
    Diseñada para alcanzar el "Sweet Spot" de 1 Millón de parámetros.
    5 ResBlocks + 3 Heads multi-tarea (Ritmo, Calidad, Biometría).
    """
    
    def __init__(
        self,
        in_channels=6,
        num_rhythm=Config.NUM_CLASSES_RHYTHM,
        num_pathology: int = 0,
        morphology_dim: int = 0,
    ):
        super().__init__()
        self.morphology_dim = int(morphology_dim or 0)
        
        # ─── 1. Stem (Entrada de señal cruda → 64 canales) ──────────────
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(64),
            nn.SiLU()
        )
        
        # ─── 2. ResNet Core Escalado (6 bloques) ────────────────────────
        # Progresión de canales para alcanzar ~1M parámetros:
        # 64 → 64 → 96 → 128 → 160 → 192
        self.res_layers = nn.Sequential(
            ResBlock(64,  64,   stride=1, dropout=0.1),   
            ResBlock(64,  96,   stride=2, dropout=0.15),  
            ResBlock(96,  128,  stride=2, dropout=0.2),   
            ResBlock(128, 160,  stride=2, dropout=0.25),  
            ResBlock(160, 192,  stride=2, dropout=0.3),   
        )
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        
        # ─── 3. Heads de Salida (Multi-Tarea) ──────────────────────────
        
        # Head 1: Clasificación de Ritmo (AAMI Extended — 35 clases)
        rhythm_input_dim = 192 + self.morphology_dim
        self.head_rhythm = nn.Sequential(
            nn.Linear(rhythm_input_dim, 96),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(96, num_rhythm)
        )
        
        # Head 2: Estimación de Calidad de Señal [0, 1]
        self.head_quality = nn.Sequential(
            nn.Linear(192, 64),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Head 3: Predicción Biométrica (Age, Sex, BMI)
        self.head_biometrics = nn.Sequential(
            nn.Linear(192, 64),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 3) 
        )

        self.num_pathology = int(num_pathology or 0)
        self.head_pathology = None
        if self.num_pathology > 0:
            # Multi-label pathology logits (use BCEWithLogitsLoss in training).
            self.head_pathology = nn.Sequential(
                nn.Linear(192, 96),
                nn.SiLU(),
                nn.Dropout(0.3),
                nn.Linear(96, self.num_pathology),
            )

    def forward(self, x, morphology_features=None, return_features=False):
        """
        x shape: (batch_size, 6, 1250)  — 10s @ 125Hz, 6 Derivadas Frontales
        
        Returns:
            out_rhythm:     (batch_size, 35)  — logits de clasificación rítmica
            out_quality:    (batch_size, 1)   — probabilidad de calidad [0, 1]
            out_biometrics: (batch_size, 3)   — [BMI, Sexo, Edad_norm]
        """
        features = self.forward_features(x)
        rhythm_features = features
        if self.morphology_dim > 0:
            if morphology_features is None:
                raise ValueError("morphology_features requerido cuando morphology_dim > 0")
            morphology_features = morphology_features.to(device=features.device, dtype=features.dtype)
            if morphology_features.ndim != 2 or morphology_features.shape[1] != self.morphology_dim:
                raise ValueError(
                    f"morphology_features debe tener forma [B,{self.morphology_dim}], "
                    f"recibido {tuple(morphology_features.shape)}"
                )
            morphology_features = torch.nan_to_num(morphology_features, nan=0.0, posinf=0.0, neginf=0.0)
            rhythm_features = torch.cat([features, morphology_features], dim=1)
        
        # Multi-Task Heads
        out_rhythm = self.head_rhythm(rhythm_features)
        out_quality = self.head_quality(features)
        out_biometrics = self.head_biometrics(features)
        
        if self.head_pathology is None:
            outputs = (out_rhythm, out_quality, out_biometrics)
            return (*outputs, features) if return_features else outputs
        out_pathology = self.head_pathology(features)
        outputs = (out_rhythm, out_quality, out_biometrics, out_pathology)
        return (*outputs, features) if return_features else outputs

    def forward_features(self, x):
        """
        Returns the shared embedding before heads.
        Shape: [B, 192]
        """
        x = self.stem(x)           # [B, 64, 625]
        x = self.res_layers(x)     # [B, 192, ~...]
        x = self.global_pool(x)    # [B, 192, 1]
        features = self.flatten(x) # [B, 192]
        return features


if __name__ == '__main__':
    print("=" * 60)
    print("  TITAN V4 — Test de Arquitectura Escalada")
    print("=" * 60)
    
    model = TitanV4Lite(in_channels=6, num_rhythm=35)
    
    # Tensor simulado: Batch=4, Leads=6, Samples=1250 (10s @ 125Hz)
    dummy_input = torch.randn(4, 6, 1250)
    
    # Conteo de parámetros por componente
    stem_params = sum(p.numel() for p in model.stem.parameters())
    res_params = sum(p.numel() for p in model.res_layers.parameters())
    head_r_params = sum(p.numel() for p in model.head_rhythm.parameters())
    head_q_params = sum(p.numel() for p in model.head_quality.parameters())
    head_b_params = sum(p.numel() for p in model.head_biometrics.parameters())
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"  Stem:           {stem_params:>10,} params")
    print(f"  ResBlocks (×6): {res_params:>10,} params")
    print(f"  Head Ritmo:     {head_r_params:>10,} params")
    print(f"  Head Calidad:   {head_q_params:>10,} params")
    print(f"  Head Biometría: {head_b_params:>10,} params")
    print(f"  {'─'*35}")
    print(f"  TOTAL:          {total_params:>10,} params")
    print(f"  Tamaño FP32:    {total_params * 4 / 1024 / 1024:.1f} MB")
    print(f"  Tamaño INT8:    {total_params / 1024 / 1024:.1f} MB")
    print()
    
    # Forward pass
    out_r, out_q, out_b = model(dummy_input)
    print(f"  Input shape:      {list(dummy_input.shape)}")
    print(f"  Output Rhythm:    {list(out_r.shape)}  (35 clases)")
    print(f"  Output Quality:   {list(out_q.shape)}  (Sigmoid)")
    print(f"  Output Biometrics:{list(out_b.shape)}  (Age/Sex/BMI)")
    print()
    print(f"  Ratio datos/params: 570,706 / {total_params:,} = {570706/total_params:.1f}:1")
    print("  Test exitoso ✅")
