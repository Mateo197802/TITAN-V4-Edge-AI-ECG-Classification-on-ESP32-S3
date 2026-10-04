from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from titan_v4.data.primary9_registry import PRIMARY9_CLASSES
from titan_v4.models.titan_v4_net import TitanV4Lite


def load_primary9_model(checkpoint_path: str | Path, *, device: str = "cpu") -> TitanV4Lite:
    """Load the distributed state-dict checkpoint against the packaged architecture."""
    checkpoint = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=True)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    if not isinstance(checkpoint, dict) or not checkpoint:
        raise ValueError("Checkpoint must contain a non-empty model state dictionary")

    model = TitanV4Lite(
        in_channels=6,
        num_rhythm=len(PRIMARY9_CLASSES),
        num_pathology=10,
    )
    model.load_state_dict(checkpoint, strict=True)
    model.to(torch.device(device))
    model.eval()
    return model


def predict_primary9(model: TitanV4Lite, signals: np.ndarray, *, batch_size: int = 32) -> np.ndarray:
    """Return class probabilities for [records, six leads, 1250 samples]."""
    inputs = np.asarray(signals, dtype=np.float32)
    if inputs.ndim != 3 or tuple(inputs.shape[1:]) != (6, 1250):
        raise ValueError(f"Expected [records, 6, 1250] model inputs, got {inputs.shape}")
    if not np.isfinite(inputs).all():
        raise ValueError("Model inputs contain non-finite values")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    device = next(model.parameters()).device
    batches: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(inputs), batch_size):
            batch = torch.from_numpy(inputs[start : start + batch_size]).to(device)
            logits = model(batch)[0]
            if logits.ndim != 2 or logits.shape[1] != len(PRIMARY9_CLASSES):
                raise ValueError(f"Checkpoint returned unexpected rhythm logits shape {tuple(logits.shape)}")
            batches.append(torch.softmax(logits, dim=1).cpu().numpy())
    if not batches:
        raise ValueError("No ECG records were provided for inference")
    return np.concatenate(batches, axis=0)


def primary9_metrics(y_true: Iterable[str], y_pred: Iterable[str]) -> dict[str, object]:
    truth = [str(value) for value in y_true]
    predictions = [str(value) for value in y_pred]
    if len(truth) != len(predictions) or not truth:
        raise ValueError("Ground truth and predictions must have the same non-zero length")
    unknown = sorted((set(truth) | set(predictions)) - set(PRIMARY9_CLASSES))
    if unknown:
        raise ValueError(f"Labels outside the frozen Primary-9 class set: {unknown}")

    report = classification_report(
        truth,
        predictions,
        labels=list(PRIMARY9_CLASSES),
        target_names=list(PRIMARY9_CLASSES),
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(truth, predictions, labels=list(PRIMARY9_CLASSES))
    return {
        "classes": list(PRIMARY9_CLASSES),
        "total_records": len(truth),
        "correct_predictions": int(sum(actual == predicted for actual, predicted in zip(truth, predictions))),
        "accuracy": float(accuracy_score(truth, predictions)),
        "macro_f1": float(f1_score(truth, predictions, labels=list(PRIMARY9_CLASSES), average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(truth, predictions, labels=list(PRIMARY9_CLASSES), average="weighted", zero_division=0)),
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
    }
