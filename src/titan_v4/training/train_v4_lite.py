"""
TITAN V4 LOWCOST - Training Pipeline for CEDIA HPC
=================================================

This version fixes the previous evaluation leak:
  - record-level train/validation split before window slicing is consumed
  - validation uses augment=False
  - best checkpoint is selected by validation macro-F1
  - focal loss receives class weights from the training split
  - optional weighted sampler reduces NSR dominance
  - early stopping prevents memorization
"""
import argparse
import csv
import json
import math
import os
import random
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader, Subset

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from audit_label_coverage import audit_label_coverage, write_audit_reports
from data_loader_ultra import ClinicalECGDataset, NUM_RHYTHM_CLASSES, SNOMED_MAPPING, TARGET_LENGTH
from distillation_utils import project_full14_logits_to_primary9
from label_registry import RHYTHM_CLASS_NAMES
from morphology_features import MORPHOLOGY_FEATURE_DIM
from primary9_registry import PRIMARY9_CLASSES, PRIMARY9_TEACHER_GROUP_INDICES, PRIMARY9_TO_FULL14_INDEX
from titan_v4_net import TitanV4Lite
from training_utils import (
    build_record_splits,
    compute_balanced_sample_weights,
    compute_class_weights,
    compute_sample_weights,
    compute_tempered_sample_weights,
    labels_for_indices,
    load_record_split_from_manifests,
    save_split_metadata,
    split_window_indices_by_record,
)


RHYTHM_NAMES = list(RHYTHM_CLASS_NAMES)


def resolve_active_rhythm_config(args):
    if getattr(args, "primary9_mode", False):
        return len(PRIMARY9_CLASSES), list(PRIMARY9_CLASSES)
    return NUM_RHYTHM_CLASSES, list(RHYTHM_NAMES)


def maybe_project_teacher_logits(student_logits, teacher_logits, *, primary9_mode: bool):
    if teacher_logits.shape == student_logits.shape:
        return teacher_logits
    if primary9_mode and student_logits.ndim == 2 and teacher_logits.ndim == 2:
        if student_logits.shape[1] == len(PRIMARY9_CLASSES) and teacher_logits.shape[1] >= NUM_RHYTHM_CLASSES:
            return project_full14_logits_to_primary9(teacher_logits)
    raise ValueError(
        f"Teacher/student logits incompatibles: student={tuple(student_logits.shape)} "
        f"teacher={tuple(teacher_logits.shape)} primary9={primary9_mode}"
    )


def _unpack_batch(batch):
    # Backward/forward compatible with datasets returning:
    # (x, ry, qy, bio), (x, ry, qy, bio, record_path) or
    # (x, ry, qy, bio, record_path, morphology_features/window_key) or
    # (x, ry, qy, bio, record_path, morphology_features, window_key)
    if isinstance(batch, (list, tuple)) and len(batch) == 7:
        return batch[0], batch[1], batch[2], batch[3], batch[4], batch[5], batch[6]
    if isinstance(batch, (list, tuple)) and len(batch) == 6:
        if torch.is_tensor(batch[5]):
            return batch[0], batch[1], batch[2], batch[3], batch[4], batch[5], None
        return batch[0], batch[1], batch[2], batch[3], batch[4], None, batch[5]
    if isinstance(batch, (list, tuple)) and len(batch) == 5:
        return batch[0], batch[1], batch[2], batch[3], batch[4], None, None
    if isinstance(batch, (list, tuple)) and len(batch) == 4:
        return batch[0], batch[1], batch[2], batch[3], None, None, None
    raise ValueError(f"Batch inesperado (len={len(batch) if isinstance(batch,(list,tuple)) else 'n/a'}): {type(batch)}")


def masked_age_sex_loss(out_b, batch_by):
    """MSE for sex and age only; legacy BMI and missing metadata are ignored."""
    pred = out_b[:, 1:3].float()
    target = batch_by[:, 1:3].float()
    if not torch.isfinite(target).all():
        return target.sum()
    valid = target >= 0.0
    if not torch.any(valid):
        return pred.sum() * 0.0
    return torch.mean((pred[valid] - target[valid]) ** 2)


def compute_biometry_metrics(pred_b, target_b):
    """Compute sex classification and age regression metrics; legacy BMI is ignored."""
    if isinstance(pred_b, torch.Tensor):
        pred = pred_b.detach().cpu().float()
    else:
        pred = torch.as_tensor(pred_b, dtype=torch.float32)
    if isinstance(target_b, torch.Tensor):
        target = target_b.detach().cpu().float()
    else:
        target = torch.as_tensor(target_b, dtype=torch.float32)

    if pred.numel() == 0 or target.numel() == 0:
        return {
            "biometry_sex_valid": 0,
            "biometry_sex_acc": float("nan"),
            "biometry_sex_f1": float("nan"),
            "biometry_age_valid": 0,
            "biometry_age_mae_years": float("nan"),
            "biometry_age_within5_acc": float("nan"),
            "biometry_age_within10_acc": float("nan"),
        }

    sex_target = target[:, 1]
    sex_valid = torch.isfinite(sex_target) & (sex_target >= 0.0) & (sex_target <= 1.0)
    sex_acc = float("nan")
    sex_f1 = float("nan")
    if torch.any(sex_valid):
        y_true = (sex_target[sex_valid] >= 0.5).numpy().astype(int)
        y_pred = (pred[:, 1][sex_valid] >= 0.5).numpy().astype(int)
        sex_acc = float(accuracy_score(y_true, y_pred))
        sex_f1 = float(f1_score(y_true, y_pred, average="binary", zero_division=0))

    age_target = target[:, 2]
    age_valid = torch.isfinite(age_target) & (age_target >= 0.0)
    age_mae = float("nan")
    age_within5 = float("nan")
    age_within10 = float("nan")
    if torch.any(age_valid):
        age_err_years = torch.abs(pred[:, 2][age_valid] - age_target[age_valid]) * 100.0
        age_mae = float(torch.mean(age_err_years).item())
        age_within5 = float(torch.mean((age_err_years <= 5.0).float()).item())
        age_within10 = float(torch.mean((age_err_years <= 10.0).float()).item())

    return {
        "biometry_sex_valid": int(torch.sum(sex_valid).item()),
        "biometry_sex_acc": sex_acc,
        "biometry_sex_f1": sex_f1,
        "biometry_age_valid": int(torch.sum(age_valid).item()),
        "biometry_age_mae_years": age_mae,
        "biometry_age_within5_acc": age_within5,
        "biometry_age_within10_acc": age_within10,
    }


def multi_task_checkpoint_score(val_metrics, *, pathology_alpha=0.1, biometry_alpha=0.05):
    """Primary score remains rhythm F1; auxiliary heads can break ties/improve multitask candidates."""
    score = float(val_metrics.get("f1_macro", 0.0))
    pr_auc = val_metrics.get("pathology_pr_auc_macro", float("nan"))
    if pr_auc == pr_auc:
        score += float(pathology_alpha) * float(pr_auc)

    sex_f1 = val_metrics.get("biometry_sex_f1", float("nan"))
    age_within10 = val_metrics.get("biometry_age_within10_acc", float("nan"))
    aux = []
    if sex_f1 == sex_f1:
        aux.append(float(sex_f1))
    if age_within10 == age_within10:
        aux.append(float(age_within10))
    if aux:
        score += float(biometry_alpha) * (sum(aux) / len(aux))
    return score


def freeze_batchnorm_modules(model):
    """Keep BatchNorm running statistics fixed during fine-tuning."""
    frozen = 0
    batchnorm_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)
    for module in model.modules():
        if isinstance(module, batchnorm_types):
            module.eval()
            for param in module.parameters():
                param.requires_grad_(False)
            frozen += 1
    return frozen


def freeze_for_auxiliary_head_tuning(model, *, train_quality=False):
    """Freeze shared ECG encoder and rhythm head; train pathology + age/sex heads only."""
    for param in model.parameters():
        param.requires_grad_(False)

    trainable_modules = [model.head_biometrics]
    if getattr(model, "head_pathology", None) is not None:
        trainable_modules.append(model.head_pathology)
    if train_quality:
        trainable_modules.append(model.head_quality)

    for module in trainable_modules:
        for param in module.parameters():
            param.requires_grad_(True)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {"trainable_params": int(trainable), "total_params": int(total)}


def apply_early_backbone_freeze(model, *, epoch: int, freeze_epochs: int = 0, train_last_n_resblocks: int = 1):
    """Freeze stem and early ResBlocks for staged fine-tuning."""
    target = model.module if isinstance(model, nn.DataParallel) else model
    if int(freeze_epochs or 0) <= 0:
        for param in target.parameters():
            param.requires_grad_(True)
        return {"active": False, "trainable_params": int(sum(p.numel() for p in target.parameters() if p.requires_grad))}

    freeze_active = int(epoch) <= int(freeze_epochs)
    for param in target.parameters():
        param.requires_grad_(True)
    if freeze_active:
        for param in target.stem.parameters():
            param.requires_grad_(False)
        res_layers = list(target.res_layers)
        frozen_until = max(0, len(res_layers) - int(train_last_n_resblocks or 1))
        for layer in res_layers[:frozen_until]:
            for param in layer.parameters():
                param.requires_grad_(False)
    trainable = sum(p.numel() for p in target.parameters() if p.requires_grad)
    return {"active": bool(freeze_active), "trainable_params": int(trainable)}


def recall_for_class(y_true, y_pred, class_idx: int) -> float:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    mask = y_true == int(class_idx)
    if not np.any(mask):
        return float("nan")
    return float(np.mean(y_pred[mask] == int(class_idx)))


def _normalize_slash(path: str) -> str:
    return (path or "").replace("\\", "/")


def load_ptbxl_pathology_sidecar(pathology_sidecar_json: str, ptbxl_root: str):
    """
    Sidecar format:
      {
        "num_classes": 10,
        "class_names": [...],
        "records": {
          "records100/00000/00008_lr": {"labels":[0,3], ...},
          ...
        }
      }
    Returns:
      class_names: list[str]
      record_to_multihot: dict[abs_record_path_no_ext -> np.ndarray shape [C]]
    """
    if not pathology_sidecar_json:
        return [], {}
    if not os.path.exists(pathology_sidecar_json):
        raise FileNotFoundError(f"No existe pathology sidecar: {pathology_sidecar_json}")
    with open(pathology_sidecar_json, "r", encoding="utf-8") as handle:
        obj = json.load(handle)

    class_names = list(obj.get("class_names") or [])
    num_classes = int(obj.get("num_classes") or len(class_names) or 0)
    if not class_names and num_classes > 0:
        class_names = [f"P{i}" for i in range(num_classes)]

    records = obj.get("records") or {}
    ptbxl_root = os.path.abspath(ptbxl_root)
    record_to_multihot = {}

    for rel_key, rec in records.items():
        if not isinstance(rec, dict):
            continue
        labels = rec.get("labels")
        if not isinstance(labels, list):
            continue
        y = np.zeros((num_classes,), dtype=np.float32)
        for idx in labels:
            try:
                j = int(idx)
            except Exception:
                continue
            if 0 <= j < num_classes:
                y[j] = 1.0
        abs_record = os.path.abspath(os.path.join(ptbxl_root, rel_key))
        record_to_multihot[abs_record] = y

    return class_names, record_to_multihot


def compute_pathology_pos_weight(record_paths, pathology_record_to_multihot, num_classes, max_weight=20.0):
    labeled = []
    for path in record_paths:
        vec = pathology_record_to_multihot.get(os.path.abspath(path))
        if vec is not None:
            labeled.append(np.asarray(vec, dtype=np.float32))

    if not labeled or int(num_classes) <= 0:
        return [1.0] * int(num_classes), {
            "labeled_windows": 0,
            "positive_counts": [0] * int(num_classes),
            "negative_counts": [0] * int(num_classes),
        }

    y = np.stack(labeled, axis=0).astype(np.float32)
    pos = y.sum(axis=0)
    neg = y.shape[0] - pos
    weights = np.ones((int(num_classes),), dtype=np.float32)
    valid = pos > 0
    weights[valid] = neg[valid] / np.maximum(pos[valid], 1.0)
    weights = np.clip(weights, 1.0, float(max_weight))
    return [float(x) for x in weights.tolist()], {
        "labeled_windows": int(y.shape[0]),
        "positive_counts": [int(x) for x in pos.astype(np.int64).tolist()],
        "negative_counts": [int(x) for x in neg.astype(np.int64).tolist()],
    }


def build_training_summary_payload(
    *,
    best_epoch,
    best_val_f1,
    best_val_score,
    checkpoint_metric,
    training_log,
    train_windows,
    val_windows,
    full_train_windows,
    full_val_windows,
    sampler,
    train_samples_per_epoch,
    balanced_samples_per_class,
    class_weights,
    loss_class_weight_mode,
    loss_class_weights,
    num_workers,
    pin_memory,
    persistent_workers,
    prefetch_factor,
    initial_val_metrics=None,
    pathology_summary=None,
    biometry_summary=None,
    boundary_aux_summary=None,
    teacher_distillation_summary=None,
    offline_teacher_summary=None,
    morphology_fusion=False,
    morphology_dim=0,
):
    summary = {
        "best_epoch": int(best_epoch),
        "best_val_f1_macro": float(best_val_f1),
        "best_val_checkpoint_score": float(best_val_score),
        "checkpoint_metric": checkpoint_metric,
        "epochs_ran": len(training_log),
        "train_windows": int(train_windows),
        "val_windows": int(val_windows),
        "full_train_windows": int(full_train_windows),
        "full_val_windows": int(full_val_windows),
        "sampler": sampler,
        "train_samples_per_epoch": int(train_samples_per_epoch),
        "balanced_samples_per_class": int(balanced_samples_per_class),
        "class_weights": list(class_weights),
        "loss_class_weight_mode": loss_class_weight_mode,
        "loss_class_weights": list(loss_class_weights),
        "num_workers": int(num_workers),
        "pin_memory": bool(pin_memory),
        "persistent_workers": bool(persistent_workers),
        "prefetch_factor": prefetch_factor,
        "morphology_fusion": bool(morphology_fusion),
        "morphology_dim": int(morphology_dim),
    }
    if initial_val_metrics is not None:
        summary.update(
            {
                "initial_val_loss": initial_val_metrics["loss"],
                "initial_val_f1_macro": initial_val_metrics["f1_macro"],
                "initial_val_acc": initial_val_metrics["acc"],
                "initial_val_nonfinite_loss_batches": initial_val_metrics.get("nonfinite_loss_batches", 0),
            }
        )
    if pathology_summary:
        summary.update(pathology_summary)
    if biometry_summary:
        summary.update(biometry_summary)
    if boundary_aux_summary:
        summary.update(boundary_aux_summary)
    if teacher_distillation_summary:
        summary.update(teacher_distillation_summary)
    if offline_teacher_summary:
        summary.update(offline_teacher_summary)
    return summary


def write_training_summary(path: str, **kwargs) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(build_training_summary_payload(**kwargs), handle, indent=2)


def _best_f1_thresholds_per_class(y_true: np.ndarray, y_prob: np.ndarray, class_names):
    # y_true/y_prob shapes: [N, C]
    thresholds = {}
    macro_f1_classes = []
    if y_true.size == 0:
        return thresholds, float("nan"), 0
    c = y_true.shape[1]
    grid = np.linspace(0.05, 0.95, 19)
    f1s = []
    used = 0
    for j in range(c):
        yt = y_true[:, j].astype(np.int64)
        if yt.sum() == 0:
            thresholds[class_names[j] if j < len(class_names) else f"P{j}"] = 0.5
            continue
        used += 1
        best_t = 0.5
        best_f1 = -1.0
        for t in grid:
            yp = (y_prob[:, j] >= t).astype(np.int64)
            tp = int(((yp == 1) & (yt == 1)).sum())
            fp = int(((yp == 1) & (yt == 0)).sum())
            fn = int(((yp == 0) & (yt == 1)).sum())
            denom = (2 * tp + fp + fn)
            f1 = (2 * tp / denom) if denom > 0 else 0.0
            if f1 > best_f1:
                best_f1 = f1
                best_t = float(t)
        thresholds[class_names[j] if j < len(class_names) else f"P{j}"] = best_t
        f1s.append(best_f1 if best_f1 >= 0 else 0.0)
    macro_f1 = float(np.mean(f1s)) if f1s else float("nan")
    return thresholds, macro_f1, used


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.1):
        super().__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(
            weight=alpha,
            reduction="none",
            label_smoothing=label_smoothing,
        )

    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        pt = torch.exp(-ce_loss)
        return (((1 - pt) ** self.gamma) * ce_loss).mean()


def combine_boundary_aux_loss(primary_loss, boundary_loss, boundary_weight: float):
    """Add an auxiliary rhythm-boundary loss without touching other heads."""
    weight = float(boundary_weight or 0.0)
    if boundary_loss is None or weight <= 0.0:
        return primary_loss
    return primary_loss + weight * boundary_loss


def distillation_kl_loss(student_logits, teacher_logits, temperature: float = 2.0):
    """KL teacher-student loss for preserving the active rhythm boundary."""
    temp = max(float(temperature or 1.0), 1e-6)
    student_log_probs = F.log_softmax(student_logits / temp, dim=1)
    teacher_probs = F.softmax(teacher_logits.detach() / temp, dim=1)
    return F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (temp ** 2)


def parse_float_list(value: str | None, expected_len: int | None = None, default: float = 1.0):
    if value is None or str(value).strip() == "":
        if expected_len is None:
            return []
        return [float(default)] * int(expected_len)
    parts = [part.strip() for part in str(value).split(",") if part.strip()]
    values = [float(part) for part in parts]
    if expected_len is not None and len(values) != int(expected_len):
        raise ValueError(f"Se esperaban {expected_len} valores, recibidos {len(values)}")
    return values


def _batch_key_list(batch_window_keys):
    if batch_window_keys is None:
        return None
    if isinstance(batch_window_keys, (list, tuple)):
        return [str(item) for item in batch_window_keys]
    if isinstance(batch_window_keys, np.ndarray):
        return [str(item) for item in batch_window_keys.tolist()]
    return [str(item) for item in batch_window_keys]


def load_offline_teacher_tables(npz_paths, *, weights=None, device=None):
    """Load precomputed teacher logits keyed by exact 10 s window id."""
    paths = [str(path) for path in (npz_paths or []) if str(path).strip()]
    if not paths:
        return []
    weights = list(weights or [1.0] * len(paths))
    if len(weights) != len(paths):
        raise ValueError("teacher_logits_weight debe tener la misma longitud que teacher_logits_npz")

    tables = []
    for path, weight in zip(paths, weights):
        if not os.path.exists(path):
            raise FileNotFoundError(f"No existe teacher logits npz: {path}")
        payload = np.load(path, allow_pickle=False)
        if "window_keys" not in payload or "rhythm_logits" not in payload:
            raise ValueError(f"NPZ teacher invalido, faltan window_keys/rhythm_logits: {path}")
        keys = [str(item) for item in payload["window_keys"].astype(str).tolist()]
        logits = np.asarray(payload["rhythm_logits"], dtype=np.float32)
        if logits.ndim != 2 or logits.shape[0] != len(keys):
            raise ValueError(f"Shape invalida para rhythm_logits en {path}: {logits.shape} keys={len(keys)}")
        embeddings = None
        if "embeddings" in payload:
            embeddings = np.asarray(payload["embeddings"], dtype=np.float32)
            if embeddings.ndim != 2 or embeddings.shape[0] != len(keys):
                raise ValueError(f"Shape invalida para embeddings en {path}: {embeddings.shape} keys={len(keys)}")
        basename = os.path.basename(path).lower()
        group_name = None
        for candidate in PRIMARY9_TEACHER_GROUP_INDICES:
            if candidate in basename:
                group_name = candidate
                break
        tables.append(
            {
                "path": os.path.abspath(path),
                "weight": float(weight),
                "group_name": group_name,
                "group_indices": PRIMARY9_TEACHER_GROUP_INDICES.get(group_name),
                "logits_by_key": {key: logits[i] for i, key in enumerate(keys)},
                "embeddings_by_key": ({key: embeddings[i] for i, key in enumerate(keys)} if embeddings is not None else {}),
                "num_rows": int(len(keys)),
                "num_classes": int(logits.shape[1]),
                "embedding_dim": int(embeddings.shape[1]) if embeddings is not None else 0,
            }
        )
    return tables


def offline_teacher_distillation_loss(
    student_logits,
    batch_window_keys,
    teacher_tables,
    *,
    primary9_mode: bool,
    temperature: float,
):
    keys = _batch_key_list(batch_window_keys)
    if not keys or not teacher_tables:
        return None, {"matched": 0, "missing": len(keys or [])}
    total = student_logits.sum() * 0.0
    used_tables = 0
    matched_total = 0
    missing_total = 0
    for table in teacher_tables:
        rows = []
        positions = []
        for pos, key in enumerate(keys):
            row = table["logits_by_key"].get(key)
            if row is None:
                continue
            rows.append(row)
            positions.append(pos)
        missing_total += len(keys) - len(rows)
        if not rows:
            continue
        teacher = torch.as_tensor(np.stack(rows), dtype=student_logits.dtype, device=student_logits.device)
        student_subset = student_logits.index_select(
            dim=0,
            index=torch.as_tensor(positions, dtype=torch.long, device=student_logits.device),
        )
        teacher = maybe_project_teacher_logits(student_subset, teacher, primary9_mode=primary9_mode)
        group_indices = table.get("group_indices")
        if group_indices:
            idx = torch.as_tensor(group_indices, dtype=torch.long, device=student_logits.device)
            student_for_loss = student_subset.index_select(dim=1, index=idx)
            teacher_for_loss = teacher.index_select(dim=1, index=idx)
        else:
            student_for_loss = student_subset
            teacher_for_loss = teacher
        total = total + float(table["weight"]) * distillation_kl_loss(student_for_loss, teacher_for_loss, temperature)
        used_tables += 1
        matched_total += len(rows)
    if used_tables == 0:
        return None, {"matched": 0, "missing": missing_total}
    return total / float(used_tables), {"matched": int(matched_total), "missing": int(missing_total)}


def offline_teacher_embedding_loss(student_embedding, batch_window_keys, teacher_tables):
    keys = _batch_key_list(batch_window_keys)
    if not keys or not teacher_tables:
        return None, {"matched": 0, "missing": len(keys or [])}
    total = student_embedding.sum() * 0.0
    used_tables = 0
    matched_total = 0
    missing_total = 0
    for table in teacher_tables:
        if not table.get("embeddings_by_key"):
            continue
        rows = []
        positions = []
        for pos, key in enumerate(keys):
            row = table["embeddings_by_key"].get(key)
            if row is None:
                continue
            rows.append(row)
            positions.append(pos)
        missing_total += len(keys) - len(rows)
        if not rows:
            continue
        teacher = torch.as_tensor(np.stack(rows), dtype=student_embedding.dtype, device=student_embedding.device)
        student_subset = student_embedding.index_select(
            dim=0,
            index=torch.as_tensor(positions, dtype=torch.long, device=student_embedding.device),
        )
        if teacher.shape != student_subset.shape:
            raise ValueError(f"Embedding teacher/student incompatible: {teacher.shape} vs {student_subset.shape}")
        total = total + F.mse_loss(student_subset, teacher)
        used_tables += 1
        matched_total += len(rows)
    if used_tables == 0:
        return None, {"matched": 0, "missing": missing_total}
    return total / float(used_tables), {"matched": int(matched_total), "missing": int(missing_total)}


def next_cyclic_batch(loader, iterator):
    """Return the next batch and restart the loader when the iterator is exhausted."""
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def parse_loss_class_weights(value: str | None, num_classes: int):
    if value is None or str(value).strip() == "":
        return None
    parts = [part.strip() for part in str(value).split(",") if part.strip()]
    if len(parts) != int(num_classes):
        raise ValueError(f"loss_class_weights debe tener {num_classes} valores, recibidos {len(parts)}")
    weights = [float(part) for part in parts]
    if any(weight <= 0 for weight in weights):
        raise ValueError("loss_class_weights debe contener solo valores positivos")
    return weights


def resolve_rhythm_loss_weights(
    class_weights,
    sampler: str,
    mode: str,
    num_classes: int,
    custom_weights: str | None = None,
):
    parsed_custom = parse_loss_class_weights(custom_weights, num_classes)
    if parsed_custom is not None:
        return parsed_custom
    if mode == "none" or (mode == "auto" and sampler in {"balanced", "tempered"}):
        return [1.0] * int(num_classes)
    return [float(x) for x in class_weights]


def resolve_sampler_num_samples(
    train_size: int,
    num_classes: int,
    sampler: str,
    balanced_samples_per_class: int,
) -> int:
    if sampler != "balanced" or int(balanced_samples_per_class) <= 0:
        return int(train_size)
    target = int(num_classes) * int(balanced_samples_per_class)
    return min(int(train_size), max(int(num_classes), target))


def _epoch_seed(seed: int, epoch: int) -> int:
    return int(seed) + int(epoch) * 1_000_003


def build_epoch_sample_indices(
    *,
    train_size: int,
    labels,
    sampler: str,
    num_classes: int,
    epoch: int,
    seed: int,
    num_samples: int,
    max_class_weight: float = 20.0,
    min_class_weight: float = 0.05,
    sampler_alpha: float = 0.5,
):
    """Build a deterministic per-epoch list of positions inside train_indices."""
    train_size = int(train_size)
    num_samples = int(num_samples)
    if train_size <= 0 or num_samples <= 0:
        return []

    generator = torch.Generator()
    generator.manual_seed(_epoch_seed(seed, epoch))
    sampler = sampler or "shuffle"

    if sampler == "balanced":
        weights = compute_balanced_sample_weights(labels, num_classes)
        weight_tensor = torch.as_tensor(weights, dtype=torch.double)
        return torch.multinomial(
            weight_tensor,
            num_samples=num_samples,
            replacement=True,
            generator=generator,
        ).tolist()

    if sampler == "weighted":
        weights = compute_sample_weights(
            labels,
            num_classes,
            max_weight=max_class_weight,
            min_weight=min_class_weight,
        )
        weight_tensor = torch.as_tensor(weights, dtype=torch.double)
        return torch.multinomial(
            weight_tensor,
            num_samples=num_samples,
            replacement=True,
            generator=generator,
        ).tolist()

    if sampler == "tempered":
        weights = compute_tempered_sample_weights(labels, num_classes, alpha=sampler_alpha)
        weight_tensor = torch.as_tensor(weights, dtype=torch.double)
        return torch.multinomial(
            weight_tensor,
            num_samples=num_samples,
            replacement=True,
            generator=generator,
        ).tolist()

    order = torch.randperm(train_size, generator=generator).tolist()
    if num_samples <= train_size:
        return order[:num_samples]
    repeats = math.ceil(num_samples / max(train_size, 1))
    return (order * repeats)[:num_samples]


def strip_module_prefix(state_dict):
    return {k.replace("module.", ""): v for k, v in state_dict.items()}


def resolve_data_paths(data_root):
    if not os.path.isdir(data_root):
        raise FileNotFoundError(f"No se encontro DATA en: {data_root}")
    paths = [
        os.path.join(data_root, name)
        for name in sorted(os.listdir(data_root))
        if os.path.isdir(os.path.join(data_root, name))
    ]
    if not paths:
        raise FileNotFoundError(f"No se encontraron bases dentro de: {data_root}")
    return paths


def append_extra_data_paths(data_paths, extra_data_paths):
    paths = list(data_paths)
    for raw_path in extra_data_paths or []:
        path = os.path.abspath(str(raw_path))
        if not os.path.isdir(path):
            raise FileNotFoundError(f"No se encontro extra_data_path: {path}")
        paths.append(path)
    deduped = []
    seen = set()
    for path in paths:
        key = os.path.normcase(os.path.abspath(path))
        if key in seen:
            continue
        deduped.append(path)
        seen.add(key)
    return deduped


def capped_indices(indices, cap, seed):
    if cap <= 0 or len(indices) <= cap:
        return list(indices)
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(indices, size=cap, replace=False).tolist())


def data_loader_runtime_kwargs(args, device):
    kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda" and not args.no_pin_memory,
    }
    if args.num_workers > 0:
        kwargs["persistent_workers"] = args.persistent_workers
        if args.prefetch_factor > 0:
            kwargs["prefetch_factor"] = args.prefetch_factor
    return kwargs


def append_benchmark_row(path, row):
    if not path:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    exists = os.path.exists(path)
    fieldnames = [
        "timestamp",
        "num_workers",
        "prefetch_factor",
        "persistent_workers",
        "pin_memory",
        "batch_size",
        "max_train_windows",
        "max_val_windows",
        "train_windows_used",
        "val_windows_used",
        "full_train_windows",
        "full_val_windows",
        "train_batches",
        "val_batches",
        "train_batch_per_sec",
        "val_batch_per_sec",
        "epoch_seconds",
        "estimated_full_epoch_seconds",
        "val_f1_macro",
        "val_acc",
    ]
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({name: row.get(name, "") for name in fieldnames})


def _read_int_file(path):
    try:
        with open(path, "r", encoding="utf-8") as handle:
            raw = handle.read().strip()
        if raw == "max":
            return None
        return int(raw)
    except (OSError, ValueError):
        return None


def cgroup_memory_snapshot():
    """Return cgroup memory usage where available; values are in bytes."""
    if not sys.platform.startswith("linux"):
        return None

    try:
        with open("/proc/self/cgroup", "r", encoding="utf-8") as handle:
            lines = handle.readlines()
    except OSError:
        return None

    # cgroup v2: 0::/path
    for line in lines:
        parts = line.strip().split(":", 2)
        if len(parts) == 3 and parts[0] == "0":
            base = os.path.join("/sys/fs/cgroup", parts[2].lstrip("/"))
            current = _read_int_file(os.path.join(base, "memory.current"))
            maximum = _read_int_file(os.path.join(base, "memory.max"))
            if current is not None:
                return {"current": current, "max": maximum}

    # cgroup v1 memory controller.
    for line in lines:
        parts = line.strip().split(":", 2)
        if len(parts) == 3 and "memory" in parts[1].split(","):
            base = os.path.join("/sys/fs/cgroup/memory", parts[2].lstrip("/"))
            current = _read_int_file(os.path.join(base, "memory.usage_in_bytes"))
            maximum = _read_int_file(os.path.join(base, "memory.limit_in_bytes"))
            if current is not None:
                return {"current": current, "max": maximum}
    return None


def format_memory_snapshot(snapshot):
    if not snapshot:
        return "mem=n/a"
    used_gb = snapshot["current"] / (1024 ** 3)
    limit = snapshot.get("max")
    if not limit or limit > 10 ** 15:
        return f"mem={used_gb:.1f}GB"
    limit_gb = limit / (1024 ** 3)
    pct = 100.0 * snapshot["current"] / max(limit, 1)
    return f"mem={used_gb:.1f}/{limit_gb:.1f}GB ({pct:.1f}%)"


def evaluate_epoch(
    model,
    loader,
    device,
    criterion_rhythm,
    criterion_quality,
    criterion_bio,
    *,
    criterion_pathology=None,
    pathology_record_to_multihot=None,
    pathology_num_classes=0,
    pathology_class_names=None,
    pathology_weight=0.0,
    biometry_weight=0.2,
    morphology_fusion=False,
):
    model.eval()
    total_loss = 0.0
    finite_loss_batches = 0
    nonfinite_loss_batches = 0
    all_preds = []
    all_targs = []
    path_true = []
    path_prob = []
    bio_pred = []
    bio_true = []

    with torch.no_grad():
        for batch in loader:
            batch_x, batch_ry, batch_qy, batch_by, batch_paths, batch_morph, _batch_window_keys = _unpack_batch(batch)
            batch_x = batch_x.to(device)
            batch_ry = batch_ry.to(device)
            batch_qy = batch_qy.to(device)
            batch_by = batch_by.to(device)
            if morphology_fusion:
                if batch_morph is None:
                    raise ValueError("morphology_fusion activo pero el batch no contiene morphology_features")
                batch_morph = batch_morph.to(device)

            outputs = model(batch_x, morphology_features=batch_morph) if morphology_fusion else model(batch_x)
            if isinstance(outputs, (list, tuple)) and len(outputs) == 4:
                out_r, out_q, out_b, out_p = outputs
            else:
                out_r, out_q, out_b = outputs
                out_p = None
            out_q_safe = torch.clamp(out_q, min=1e-7, max=1.0 - 1e-7)

            loss_r = criterion_rhythm(out_r, batch_ry)
            loss_q = criterion_quality(out_q_safe, batch_qy)
            loss_b = masked_age_sex_loss(out_b, batch_by)
            loss = loss_r + 0.1 * loss_q + float(biometry_weight) * loss_b
            bio_pred.append(out_b.detach().cpu())
            bio_true.append(batch_by.detach().cpu())

            if (
                criterion_pathology is not None
                and pathology_weight > 0
                and out_p is not None
                and pathology_record_to_multihot
                and batch_paths is not None
                and pathology_num_classes > 0
            ):
                # Build multi-hot labels only for rows that have sidecar labels.
                abs_paths = [os.path.abspath(p) for p in batch_paths]
                has = [p in pathology_record_to_multihot for p in abs_paths]
                if any(has):
                    y = np.zeros((len(abs_paths), pathology_num_classes), dtype=np.float32)
                    for i, p in enumerate(abs_paths):
                        vec = pathology_record_to_multihot.get(p)
                        if vec is not None:
                            y[i, :] = vec
                    y_t = torch.from_numpy(y).to(device=device, dtype=torch.float32)
                    mask = torch.tensor(has, device=device, dtype=torch.bool)
                    loss_p = criterion_pathology(out_p[mask], y_t[mask])
                    loss = loss + float(pathology_weight) * loss_p

                    prob = torch.sigmoid(out_p[mask]).detach().cpu().numpy()
                    path_prob.append(prob)
                    path_true.append(y[mask.detach().cpu().numpy()])

            if torch.isfinite(loss):
                total_loss += float(loss.item())
                finite_loss_batches += 1
            else:
                nonfinite_loss_batches += 1
            all_preds.append(torch.argmax(out_r, dim=1).cpu())
            all_targs.append(batch_ry.cpu())

    if not all_preds:
        return {
            "loss": float("inf"),
            "acc": 0.0,
            "f1_macro": 0.0,
            "f1_weighted": 0.0,
            "y_true": np.asarray([], dtype=np.int64),
            "y_pred": np.asarray([], dtype=np.int64),
        }

    y_pred = torch.cat(all_preds).numpy()
    y_true = torch.cat(all_targs).numpy()
    out = {
        "loss": total_loss / max(finite_loss_batches, 1),
        "nonfinite_loss_batches": nonfinite_loss_batches,
        "acc": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "y_true": y_true,
        "y_pred": y_pred,
    }
    if path_true and path_prob:
        y_pt = np.concatenate(path_true, axis=0)
        y_pp = np.concatenate(path_prob, axis=0)
        # Macro PR-AUC (skip empty/no-positive classes by using average="macro" on filtered columns).
        pr_auc_macro = float("nan")
        used_cols = []
        for j in range(y_pt.shape[1]):
            if y_pt[:, j].sum() > 0:
                used_cols.append(j)
        if used_cols:
            pr_auc_macro = float(
                average_precision_score(y_pt[:, used_cols], y_pp[:, used_cols], average="macro")
            )
        class_names = list(pathology_class_names or [f"P{i}" for i in range(y_pt.shape[1])])
        thresholds, f1_macro, used_classes = _best_f1_thresholds_per_class(y_pt, y_pp, class_names)
        out.update(
            {
                "pathology_pr_auc_macro": pr_auc_macro,
                "pathology_f1_macro": float(f1_macro) if f1_macro == f1_macro else float("nan"),
                "pathology_thresholds": thresholds,
                "pathology_labeled_windows": int(y_pt.shape[0]),
                "pathology_used_classes": int(used_classes),
            }
        )
    if bio_pred and bio_true:
        out.update(compute_biometry_metrics(torch.cat(bio_pred, dim=0), torch.cat(bio_true, dim=0)))
    return out


def plot_training_log(training_log, out_dir):
    epochs = [row["epoch"] for row in training_log]
    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, [row["train_loss"] for row in training_log], "o-", label="Train loss")
    plt.plot(epochs, [row["val_loss"] for row in training_log], "s-", label="Val loss")
    plt.xlabel("Epoca")
    plt.ylabel("Loss")
    plt.title("Convergencia")
    plt.grid(alpha=0.3)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, [row["train_f1_macro"] for row in training_log], "o-", label="Train F1 macro")
    plt.plot(epochs, [row["val_f1_macro"] for row in training_log], "s-", label="Val F1 macro")
    plt.xlabel("Epoca")
    plt.ylabel("F1 macro")
    plt.ylim(0, 1.05)
    plt.title("Generalizacion")
    plt.grid(alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "training_curves.png"), dpi=250, bbox_inches="tight")
    plt.close()


def save_validation_artifacts(y_true, y_pred, out_dir, rhythm_names=None):
    present = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    rhythm_names = list(rhythm_names or RHYTHM_NAMES)
    names = [rhythm_names[i] if i < len(rhythm_names) else f"C{i}" for i in present]
    report = classification_report(
        y_true,
        y_pred,
        labels=present,
        target_names=names,
        zero_division=0,
        output_dict=True,
    )
    with open(os.path.join(out_dir, "validation_report.json"), "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    cm = confusion_matrix(y_true, y_pred, labels=present)
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=names, yticklabels=names, ax=axes[0])
    axes[0].set_title("Validacion - conteos")
    axes[0].set_xlabel("Prediccion")
    axes[0].set_ylabel("Real")
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", xticklabels=names, yticklabels=names, ax=axes[1], vmin=0, vmax=1)
    axes[1].set_title("Validacion - recall por clase")
    axes[1].set_xlabel("Prediccion")
    axes[1].set_ylabel("Real")
    for ax in axes:
        ax.tick_params(axis="x", rotation=45)
        ax.tick_params(axis="y", rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "confusion_matrix_val.png"), dpi=250, bbox_inches="tight")
    plt.close()


def _model_state_dict(model):
    return model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()


def _load_model_state_dict(model, state_dict):
    target = model.module if isinstance(model, nn.DataParallel) else model
    target.load_state_dict(state_dict)


def filter_compatible_state_dict(source_state, target_state):
    """Return source weights compatible with target, adapting rhythm fusion head when possible."""
    filtered = {}
    skipped = []
    for key, value in source_state.items():
        if key not in target_state:
            skipped.append(key)
            continue
        source_tensor = value
        target_tensor = target_state[key]
        if getattr(source_tensor, "shape", None) == getattr(target_tensor, "shape", None):
            filtered[key] = source_tensor
            continue
        if (
            key == "head_rhythm.3.weight"
            and source_tensor.ndim == 2
            and target_tensor.ndim == 2
            and source_tensor.shape[0] >= NUM_RHYTHM_CLASSES
            and target_tensor.shape[0] == len(PRIMARY9_CLASSES)
            and source_tensor.shape[1] == target_tensor.shape[1]
        ):
            idx = torch.as_tensor(PRIMARY9_TO_FULL14_INDEX, dtype=torch.long, device=source_tensor.device)
            filtered[key] = source_tensor.index_select(dim=0, index=idx)
            continue
        if (
            key == "head_rhythm.3.bias"
            and source_tensor.ndim == 1
            and target_tensor.ndim == 1
            and source_tensor.shape[0] >= NUM_RHYTHM_CLASSES
            and target_tensor.shape[0] == len(PRIMARY9_CLASSES)
        ):
            idx = torch.as_tensor(PRIMARY9_TO_FULL14_INDEX, dtype=torch.long, device=source_tensor.device)
            filtered[key] = source_tensor.index_select(dim=0, index=idx)
            continue
        if (
            key == "head_rhythm.0.weight"
            and source_tensor.ndim == 2
            and target_tensor.ndim == 2
            and source_tensor.shape[0] == target_tensor.shape[0]
            and source_tensor.shape[1] < target_tensor.shape[1]
        ):
            adapted = target_tensor.detach().clone()
            adapted[:, : source_tensor.shape[1]] = source_tensor
            filtered[key] = adapted
            continue
        skipped.append(key)
    return filtered, skipped


def capture_rng_state():
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _coerce_rng_byte_tensor(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().to(dtype=torch.uint8)
    return torch.as_tensor(value, dtype=torch.uint8, device="cpu")


def restore_rng_state(state):
    if not state:
        return
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch" in state:
        torch.set_rng_state(_coerce_rng_byte_tensor(state["torch"]))
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([_coerce_rng_byte_tensor(item) for item in state["cuda"]])


def save_training_checkpoint(
    path,
    *,
    model,
    optimizer,
    scheduler,
    scaler,
    epoch,
    best_val_f1,
    best_val_score,
    best_epoch,
    stale_epochs,
    training_log,
    args,
    batch_idx=0,
    global_step=0,
    epoch_complete=True,
    partial_epoch_state=None,
    rng_state=None,
):
    checkpoint = {
        "checkpoint_version": 2,
        "epoch": int(epoch),
        "batch_idx": int(batch_idx),
        "global_step": int(global_step),
        "epoch_complete": bool(epoch_complete),
        "model_state": _model_state_dict(model),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "rng_state": rng_state if rng_state is not None else capture_rng_state(),
        "partial_epoch_state": partial_epoch_state,
        "best_val_f1": float(best_val_f1),
        "best_val_score": float(best_val_score),
        "best_epoch": int(best_epoch),
        "stale_epochs": int(stale_epochs),
        "training_log": list(training_log),
        "args": vars(args) if hasattr(args, "__dict__") else {},
    }
    path = os.fspath(path)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp_path = f"{path}.tmp"
    torch.save(checkpoint, tmp_path)
    os.replace(tmp_path, path)


def load_training_checkpoint(path, *, model, optimizer, scheduler, scaler, device):
    path = os.fspath(path)
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)

    _load_model_state_dict(model, checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    scheduler.load_state_dict(checkpoint["scheduler_state"])
    if scaler is not None and checkpoint.get("scaler_state") is not None:
        scaler.load_state_dict(checkpoint["scaler_state"])
    restore_rng_state(checkpoint.get("rng_state"))

    epoch = int(checkpoint.get("epoch", 0))
    epoch_complete = bool(checkpoint.get("epoch_complete", True))
    resume_batch_idx = 0 if epoch_complete else int(checkpoint.get("batch_idx", 0))
    return {
        "start_epoch": epoch + 1 if epoch_complete else epoch,
        "resume_batch_idx": resume_batch_idx,
        "global_step": int(checkpoint.get("global_step", 0)),
        "epoch_complete": epoch_complete,
        "partial_epoch_state": checkpoint.get("partial_epoch_state"),
        "best_val_f1": float(checkpoint.get("best_val_f1", -1.0)),
        "best_val_score": float(checkpoint.get("best_val_score", -1.0)),
        "best_epoch": int(checkpoint.get("best_epoch", 0)),
        "stale_epochs": int(checkpoint.get("stale_epochs", 0)),
        "training_log": list(checkpoint.get("training_log", [])),
    }


def build_partial_epoch_state(total_loss, total_batches, nan_skips, train_preds, train_targs, train_epoch_seconds):
    return {
        "total_loss": float(total_loss),
        "total_batches": int(total_batches),
        "nan_skips": int(nan_skips),
        "train_y_pred": torch.cat(train_preds).cpu() if train_preds else torch.empty(0, dtype=torch.long),
        "train_y_true": torch.cat(train_targs).cpu() if train_targs else torch.empty(0, dtype=torch.long),
        "train_epoch_seconds": float(train_epoch_seconds),
    }


def restore_partial_epoch_state(partial_state):
    if not partial_state:
        return 0.0, 0, 0, [], [], 0.0
    train_pred = partial_state.get("train_y_pred")
    train_true = partial_state.get("train_y_true")
    train_preds = [train_pred.cpu()] if isinstance(train_pred, torch.Tensor) and train_pred.numel() else []
    train_targs = [train_true.cpu()] if isinstance(train_true, torch.Tensor) and train_true.numel() else []
    return (
        float(partial_state.get("total_loss", 0.0)),
        int(partial_state.get("total_batches", 0)),
        int(partial_state.get("nan_skips", 0)),
        train_preds,
        train_targs,
        float(partial_state.get("train_epoch_seconds", 0.0)),
    )


def load_initial_weights(path, *, model, device):
    path = os.fspath(path)
    try:
        payload = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=device)
    state_dict = payload.get("model_state", payload) if isinstance(payload, dict) else payload
    if not isinstance(state_dict, dict):
        raise ValueError(f"No se pudo leer state_dict desde {path}")
    target = model.module if isinstance(model, nn.DataParallel) else model
    clean_state = {str(key).replace("module.", ""): value for key, value in state_dict.items()}
    filtered, skipped = filter_compatible_state_dict(clean_state, target.state_dict())
    missing, unexpected = target.load_state_dict(filtered, strict=False)
    if skipped:
        print(f"[INIT] Pesos omitidos por forma incompatible: {len(skipped)} ({', '.join(skipped[:5])})", flush=True)
    if missing:
        print(f"[INIT] Pesos nuevos/no cargados: {len(missing)} ({', '.join(list(missing)[:5])})", flush=True)
    if unexpected:
        print(f"[INIT] Pesos inesperados ignorados: {len(unexpected)}", flush=True)


def train_v4_model(args):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_root = os.path.abspath(args.data_root or os.path.join(base_dir, "DATA"))
    out_dir = os.path.abspath(args.output_dir or os.path.join(base_dir, "03_OUTPUTS"))
    os.makedirs(out_dir, exist_ok=True)
    active_num_classes, active_rhythm_names = resolve_active_rhythm_config(args)

    data_paths = append_extra_data_paths(resolve_data_paths(data_root), args.extra_data_path)
    print("=" * 72)
    print("TITAN V4 - entrenamiento con validacion real por registro")
    print(f"DATA: {data_root}")
    print(f"OUTPUTS: {out_dir}")
    print(f"Ritmo activo: clases={active_num_classes} primary9={bool(args.primary9_mode)}")
    print(f"Epocas={args.epochs} Batch={args.batch_size} Val={args.val_fraction:.0%} Seed={args.seed}")
    print("=" * 72)
    for path in data_paths:
        print(f"  DB: {os.path.basename(path)}")

    pathology_class_names = []
    pathology_record_to_multihot = {}
    pathology_num_classes = 0
    pathology_sidecar_used = None
    if args.pathology_head:
        ptbxl_root = os.path.join(data_root, "ptb-xl")
        pathology_sidecar_used = os.path.abspath(
            args.pathology_sidecar or os.path.join(ptbxl_root, "ptbxl_pathology_label_map.json")
        )
        pathology_class_names, pathology_record_to_multihot = load_ptbxl_pathology_sidecar(
            pathology_sidecar_json=pathology_sidecar_used,
            ptbxl_root=ptbxl_root,
        )
        pathology_num_classes = len(pathology_class_names)
        print(
            f"[PathologyHead] ACTIVO | clases={pathology_num_classes} "
            f"| sidecar={pathology_sidecar_used} | records_labelled={len(pathology_record_to_multihot)}"
        )

    audit = audit_label_coverage(
        data_paths,
        val_fraction=args.val_fraction,
        seed=args.seed,
        min_train_records_per_class=args.min_train_records_per_class,
        min_val_records_per_class=args.min_val_records_per_class,
        min_train_windows_per_class=args.min_train_windows_per_class,
        min_val_windows_per_class=args.min_val_windows_per_class,
    )
    audit_json, audit_csv = write_audit_reports(audit, out_dir)
    print(f"Auditoria etiquetas: {audit_json}")
    print(f"Auditoria etiquetas CSV: {audit_csv}")
    if not audit["passed"] and not args.allow_incomplete_classes and not args.primary9_mode:
        raise RuntimeError(
            "Auditoria de cobertura fallo; no se permite entrenamiento 15 clases. "
            + " | ".join(audit["errors"])
        )
    if args.primary9_mode and not audit["passed"]:
        print("[Primary9] Auditoria full-rhythm queda como diagnostico; el gate activo viene de manifiestos Primary-9.")

    if bool(args.train_manifest_csv) != bool(args.val_manifest_csv):
        raise ValueError("--train_manifest_csv y --val_manifest_csv deben usarse juntos")
    if args.train_manifest_csv and args.val_manifest_csv:
        split = load_record_split_from_manifests(
            args.train_manifest_csv,
            args.val_manifest_csv,
            data_root=data_root,
            seed=args.seed,
        )
        print(f"[ManifestSplit] train={args.train_manifest_csv} val={args.val_manifest_csv}")
    else:
        split = build_record_splits(
            data_paths,
            SNOMED_MAPPING,
            val_fraction=args.val_fraction,
            seed=args.seed,
            unknown_policy="skip",
        )
    save_split_metadata(split, os.path.join(out_dir, "split_metadata.json"), active_rhythm_names)
    print(f"Registros train={len(split.train_records)} val={len(split.val_records)}")
    print(f"Train label counts: {split.train_label_counts}")
    print(f"Val label counts:   {split.val_label_counts}")

    train_dataset_all = ClinicalECGDataset(
        data_paths,
        target_length=TARGET_LENGTH,
        augment=True,
        record_label_map=split.record_labels,
        skip_unlabeled=True,
        return_morphology=args.morphology_fusion,
        return_window_key=bool(args.teacher_logits_npz),
    )
    val_dataset_all = ClinicalECGDataset(
        data_paths,
        target_length=TARGET_LENGTH,
        augment=False,
        record_label_map=split.record_labels,
        skip_unlabeled=True,
        return_morphology=args.morphology_fusion,
        return_window_key=False,
    )
    train_indices, val_indices, skipped = split_window_indices_by_record(train_dataset_all, split)
    if skipped:
        print(f"[WARN] Ventanas omitidas por no estar en split: {skipped}")
    if not train_indices or not val_indices:
        raise RuntimeError("Split invalido: train o val quedaron vacios")

    full_train_windows = len(train_indices)
    full_val_windows = len(val_indices)
    train_indices = capped_indices(train_indices, args.max_train_windows, args.seed)
    val_indices = capped_indices(val_indices, args.max_val_windows, args.seed + 1000)
    if len(train_indices) != full_train_windows or len(val_indices) != full_val_windows:
        print(
            f"[BENCH/CAP] Ventanas usadas train={len(train_indices)}/{full_train_windows} "
            f"val={len(val_indices)}/{full_val_windows}"
        )

    train_labels = labels_for_indices(train_dataset_all, train_indices, split.record_labels)
    val_labels = labels_for_indices(val_dataset_all, val_indices, split.record_labels)
    class_weights = compute_class_weights(
        train_labels,
        active_num_classes,
        max_weight=args.max_class_weight,
        min_weight=args.min_class_weight,
    )
    print(f"Ventanas train={len(train_indices)} val={len(val_indices)}")
    print(f"Pesos de clase: {[round(x, 3) for x in class_weights]}")

    val_subset = Subset(val_dataset_all, val_indices)

    sampler_num_samples = len(train_indices)
    if args.sampler == "balanced":
        sampler_num_samples = resolve_sampler_num_samples(
            train_size=len(train_indices),
            num_classes=active_num_classes,
            sampler=args.sampler,
            balanced_samples_per_class=args.balanced_samples_per_class,
        )
    elif args.sampler in {"weighted", "tempered"}:
        sampler_num_samples = len(train_indices)
    train_batches_per_epoch = sampler_num_samples // args.batch_size
    if train_batches_per_epoch <= 0:
        raise RuntimeError(
            f"No hay batches train completos: muestras={sampler_num_samples} batch_size={args.batch_size}"
        )
    sampler_num_samples = train_batches_per_epoch * args.batch_size

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo activo: {device}")
    loader_kwargs = data_loader_runtime_kwargs(args, device)
    print(
        "DataLoader: "
        f"workers={args.num_workers} pin_memory={loader_kwargs['pin_memory']} "
        f"persistent_workers={loader_kwargs.get('persistent_workers', False)} "
        f"prefetch_factor={loader_kwargs.get('prefetch_factor', 'default')}"
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        **loader_kwargs,
    )
    boundary_aux_loader = None
    boundary_aux_summary = {
        "boundary_aux_enabled": False,
        "boundary_aux_weight": float(args.boundary_aux_weight),
        "boundary_aux_train_windows": 0,
        "boundary_aux_full_train_windows": 0,
        "boundary_aux_batches": 0,
    }
    if args.boundary_aux_weight > 0.0 or args.boundary_aux_train_manifest_csv or args.boundary_aux_val_manifest_csv:
        if args.boundary_aux_weight <= 0.0:
            raise ValueError("--boundary_aux_weight debe ser > 0 si se definen manifests auxiliares")
        if bool(args.boundary_aux_train_manifest_csv) != bool(args.boundary_aux_val_manifest_csv):
            raise ValueError("--boundary_aux_train_manifest_csv y --boundary_aux_val_manifest_csv deben usarse juntos")
        if not args.boundary_aux_train_manifest_csv:
            raise ValueError("--boundary_aux_train_manifest_csv requerido cuando boundary_aux_weight > 0")

        boundary_split = load_record_split_from_manifests(
            args.boundary_aux_train_manifest_csv,
            args.boundary_aux_val_manifest_csv,
            data_root=data_root,
            seed=args.seed,
        )
        boundary_dataset_all = ClinicalECGDataset(
            data_paths,
            target_length=TARGET_LENGTH,
            augment=True,
            record_label_map=boundary_split.record_labels,
            skip_unlabeled=True,
            return_morphology=args.morphology_fusion,
            return_window_key=False,
        )
        boundary_train_indices, _boundary_val_indices, boundary_skipped = split_window_indices_by_record(
            boundary_dataset_all,
            boundary_split,
        )
        if boundary_skipped:
            print(f"[BoundaryAux][WARN] Ventanas omitidas por no estar en split auxiliar: {boundary_skipped}")
        full_boundary_train_windows = len(boundary_train_indices)
        boundary_train_indices = capped_indices(
            boundary_train_indices,
            args.boundary_aux_max_windows,
            args.seed + 2000,
        )
        if not boundary_train_indices:
            raise RuntimeError("Boundary auxiliary invalido: train quedo vacio")
        boundary_batch_size = int(args.boundary_aux_batch_size or args.batch_size)
        boundary_subset = Subset(boundary_dataset_all, boundary_train_indices)
        boundary_aux_loader = DataLoader(
            boundary_subset,
            batch_size=boundary_batch_size,
            shuffle=True,
            drop_last=True,
            **loader_kwargs,
        )
        if len(boundary_aux_loader) <= 0:
            raise RuntimeError(
                f"Boundary auxiliary sin batches completos: windows={len(boundary_train_indices)} "
                f"batch={boundary_batch_size}"
            )
        boundary_aux_summary = {
            "boundary_aux_enabled": True,
            "boundary_aux_weight": float(args.boundary_aux_weight),
            "boundary_aux_train_manifest_csv": args.boundary_aux_train_manifest_csv,
            "boundary_aux_val_manifest_csv": args.boundary_aux_val_manifest_csv,
            "boundary_aux_batch_size": boundary_batch_size,
            "boundary_aux_train_windows": int(len(boundary_train_indices)),
            "boundary_aux_full_train_windows": int(full_boundary_train_windows),
            "boundary_aux_batches": int(len(boundary_aux_loader)),
        }
        print(
            "[BoundaryAux] ACTIVO | "
            f"lambda={args.boundary_aux_weight} "
            f"windows={len(boundary_train_indices)}/{full_boundary_train_windows} "
            f"batch={boundary_batch_size} batches={len(boundary_aux_loader)}",
            flush=True,
        )

    model = TitanV4Lite(
        in_channels=6,
        num_rhythm=active_num_classes,
        num_pathology=pathology_num_classes if args.pathology_head else 0,
        morphology_dim=MORPHOLOGY_FEATURE_DIM if args.morphology_fusion else 0,
    ).to(device)
    teacher_model = None
    if args.teacher_kl_weight > 0.0 or args.teacher_weights:
        if args.teacher_kl_weight <= 0.0:
            raise ValueError("--teacher_kl_weight debe ser > 0 si se define --teacher_weights")
        if not args.teacher_weights:
            raise ValueError("--teacher_weights requerido cuando teacher_kl_weight > 0")
        teacher_path = os.path.abspath(args.teacher_weights)
        if not os.path.exists(teacher_path):
            raise FileNotFoundError(f"Pesos teacher no existen: {teacher_path}")
        teacher_model = TitanV4Lite(
            in_channels=6,
            num_rhythm=NUM_RHYTHM_CLASSES if args.primary9_mode else active_num_classes,
            num_pathology=pathology_num_classes if args.pathology_head else 0,
            morphology_dim=MORPHOLOGY_FEATURE_DIM if args.morphology_fusion else 0,
        ).to(device)
        load_initial_weights(teacher_path, model=teacher_model, device=device)
        teacher_model.eval()
        for param in teacher_model.parameters():
            param.requires_grad_(False)
        teacher_distillation_summary = {
            "teacher_distillation_enabled": True,
            "teacher_weights": teacher_path,
            "teacher_kl_weight": float(args.teacher_kl_weight),
            "teacher_temperature": float(args.teacher_temperature),
        }
        print(
            "[TeacherDistill] ACTIVO | "
            f"weights={teacher_path} alpha={args.teacher_kl_weight} T={args.teacher_temperature}",
            flush=True,
        )
    else:
        teacher_distillation_summary = {
            "teacher_distillation_enabled": False,
            "teacher_kl_weight": float(args.teacher_kl_weight),
            "teacher_temperature": float(args.teacher_temperature),
        }
    offline_teacher_tables = load_offline_teacher_tables(
        args.teacher_logits_npz,
        weights=parse_float_list(args.teacher_logits_weight, expected_len=len(args.teacher_logits_npz), default=1.0),
        device=device,
    )
    offline_teacher_summary = {
        "offline_teacher_logits_enabled": bool(offline_teacher_tables),
        "offline_teacher_logits_weight": float(args.teacher_logits_kl_weight),
        "offline_teacher_embedding_weight": float(args.teacher_embedding_weight),
        "offline_teacher_temperature": float(args.teacher_logits_temperature),
        "offline_teacher_tables": [
            {
                "path": table["path"],
                "weight": float(table["weight"]),
                "rows": int(table["num_rows"]),
                "num_classes": int(table["num_classes"]),
                "embedding_dim": int(table["embedding_dim"]),
                "group_name": table.get("group_name"),
                "group_indices": table.get("group_indices"),
            }
            for table in offline_teacher_tables
        ],
    }
    if offline_teacher_tables:
        print(
            "[OfflineTeacherLogits] ACTIVO | "
            f"tables={len(offline_teacher_tables)} "
            f"kl_weight={args.teacher_logits_kl_weight} "
            f"embed_weight={args.teacher_embedding_weight}",
            flush=True,
        )
    if args.morphology_fusion:
        print(f"[MorphologyFusion] ACTIVO | dim={MORPHOLOGY_FEATURE_DIM}", flush=True)
    if torch.cuda.device_count() > 1:
        print(f"[HPC] Usando {torch.cuda.device_count()} GPUs con DataParallel")
        model = nn.DataParallel(model)

    scale_factor = args.batch_size / 32.0
    lr = args.base_lr * scale_factor
    max_lr = args.max_lr * scale_factor
    loss_class_weights = resolve_rhythm_loss_weights(
        class_weights=class_weights,
        sampler=args.sampler,
        mode=args.loss_class_weight_mode,
        num_classes=active_num_classes,
        custom_weights=args.loss_class_weights,
    )
    print(f"Sampler: {args.sampler}")
    print(f"Muestras train por epoca: {sampler_num_samples}/{len(train_indices)}")
    print(f"Modo pesos loss ritmo: {args.loss_class_weight_mode}")
    print(f"Pesos loss ritmo: {[round(x, 3) for x in loss_class_weights]}")
    class_weight_tensor = torch.tensor(loss_class_weights, dtype=torch.float32, device=device)
    criterion_rhythm = FocalLoss(alpha=class_weight_tensor, gamma=args.gamma, label_smoothing=args.label_smoothing)
    criterion_quality = nn.BCELoss()
    criterion_bio = nn.MSELoss()
    pathology_pos_weights = []
    pathology_pos_weight_stats = {}
    criterion_pathology = None
    if args.pathology_head:
        train_pathology_paths = [train_dataset_all.windows[int(idx)][0] for idx in train_indices]
        pathology_pos_weights, pathology_pos_weight_stats = compute_pathology_pos_weight(
            train_pathology_paths,
            pathology_record_to_multihot,
            pathology_num_classes,
            max_weight=args.max_pathology_pos_weight,
        )
        print(f"Pathology pos_weight: {[round(x, 3) for x in pathology_pos_weights]}")
        print(f"Pathology labeled train windows: {pathology_pos_weight_stats.get('labeled_windows', 0)}")
        pathology_pos_weight_tensor = torch.tensor(pathology_pos_weights, dtype=torch.float32, device=device)
        criterion_pathology = nn.BCEWithLogitsLoss(pos_weight=pathology_pos_weight_tensor)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lr,
        steps_per_epoch=train_batches_per_epoch,
        epochs=args.epochs,
        pct_start=0.3,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    best_val_f1 = -1.0
    best_val_score = -1.0
    best_epoch = 0
    stale_epochs = 0
    training_log = []
    start_epoch = 1
    resume_batch_idx = 0
    global_step = 0
    resume_partial_epoch_state = None
    init_weights_loaded = False
    initial_val_metrics = None
    if args.resume:
        resume_path = os.path.abspath(args.resume)
        if not os.path.exists(resume_path):
            raise FileNotFoundError(f"Checkpoint resume no existe: {resume_path}")
        restored = load_training_checkpoint(
            resume_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
        )
        start_epoch = restored["start_epoch"]
        best_val_f1 = restored["best_val_f1"]
        best_val_score = restored["best_val_score"]
        best_epoch = restored["best_epoch"]
        stale_epochs = restored["stale_epochs"]
        training_log = restored["training_log"]
        resume_batch_idx = restored.get("resume_batch_idx", 0)
        global_step = restored.get("global_step", 0)
        resume_partial_epoch_state = restored.get("partial_epoch_state")
        print(
            f"[RESUME] {resume_path} | siguiente_epoca={start_epoch} "
            f"batch_inicial={resume_batch_idx + 1 if resume_batch_idx else 1} "
            f"best_epoch={best_epoch} best_val_f1={best_val_f1:.4f}",
            flush=True,
        )
    elif args.init_weights:
        init_path = os.path.abspath(args.init_weights)
        if not os.path.exists(init_path):
            raise FileNotFoundError(f"Pesos iniciales no existen: {init_path}")
        load_initial_weights(init_path, model=model, device=device)
        init_weights_loaded = True
        print(f"[INIT] Pesos iniciales cargados desde {init_path}", flush=True)
    elapsed_offset = float(training_log[-1].get("elapsed", 0.0)) if training_log else 0.0
    start_time = time.time() - elapsed_offset

    if init_weights_loaded and args.initial_best_score is not None:
        best_val_score = float(args.initial_best_score)
        best_val_f1 = float(args.initial_best_f1) if args.initial_best_f1 is not None else best_val_score
        best_epoch = 0
        print(
            "[INIT] Baseline inicial protegido desde metadata | "
            f"val_f1={best_val_f1:.4f} score={best_val_score:.4f}",
            flush=True,
        )
    elif init_weights_loaded and args.eval_init_weights:
        print("[INIT] Evaluando baseline inicial antes de fine-tuning...", flush=True)
        initial_val_metrics = evaluate_epoch(
            model,
            val_loader,
            device,
            criterion_rhythm,
            criterion_quality,
            criterion_bio,
            criterion_pathology=criterion_pathology,
            pathology_record_to_multihot=pathology_record_to_multihot,
            pathology_num_classes=pathology_num_classes,
            pathology_class_names=pathology_class_names,
            pathology_weight=float(args.pathology_weight) if args.pathology_head else 0.0,
            biometry_weight=float(args.biometry_weight),
            morphology_fusion=args.morphology_fusion,
        )
        best_val_f1 = float(initial_val_metrics["f1_macro"])
        best_val_score = best_val_f1
        if args.pathology_head and args.checkpoint_metric == "combined":
            pr_auc = initial_val_metrics.get("pathology_pr_auc_macro", float("nan"))
            if pr_auc == pr_auc:
                best_val_score = best_val_score + float(args.pathology_combined_alpha) * float(pr_auc)
        elif args.checkpoint_metric == "multi_task":
            best_val_score = multi_task_checkpoint_score(
                initial_val_metrics,
                pathology_alpha=float(args.pathology_combined_alpha),
                biometry_alpha=float(args.biometry_combined_alpha),
            )
        best_epoch = 0
        save_validation_artifacts(initial_val_metrics["y_true"], initial_val_metrics["y_pred"], out_dir, active_rhythm_names)
        print(
            "[INIT] Baseline inicial | "
            f"val_loss={initial_val_metrics['loss']:.4f} "
            f"val_f1={best_val_f1:.4f} "
            f"val_acc={initial_val_metrics['acc']:.2%} "
            f"val_nonfinite={initial_val_metrics.get('nonfinite_loss_batches', 0)}",
            flush=True,
        )

    from tqdm import tqdm

    use_tqdm = sys.stderr.isatty()
    auxiliary_freeze_summary = None
    if args.freeze_for_auxiliary_heads:
        auxiliary_freeze_summary = freeze_for_auxiliary_head_tuning(
            model,
            train_quality=bool(args.train_quality_head_with_auxiliary),
        )
        print(
            "[AuxHeadTuning] backbone/head_rhythm congelados | "
            f"trainable={auxiliary_freeze_summary['trainable_params']}/"
            f"{auxiliary_freeze_summary['total_params']} parametros",
            flush=True,
        )
    if args.freeze_early_backbone_epochs > 0:
        print(
            "[StagedFreeze] ACTIVO | "
            f"freeze_early_backbone_epochs={args.freeze_early_backbone_epochs} "
            f"train_last_n_resblocks={args.train_last_n_resblocks}",
            flush=True,
        )

    nsr_class_idx = active_rhythm_names.index("NSR") if "NSR" in active_rhythm_names else None
    baseline_nsr_recall = None
    if initial_val_metrics is not None and nsr_class_idx is not None:
        baseline_nsr_recall = recall_for_class(
            initial_val_metrics.get("y_true", []),
            initial_val_metrics.get("y_pred", []),
            nsr_class_idx,
        )

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        staged_freeze_summary = apply_early_backbone_freeze(
            model,
            epoch=epoch,
            freeze_epochs=args.freeze_early_backbone_epochs,
            train_last_n_resblocks=args.train_last_n_resblocks,
        )
        if staged_freeze_summary["active"]:
            print(
                f"[StagedFreeze] epoch={epoch} trainable_params={staged_freeze_summary['trainable_params']}",
                flush=True,
            )
        if args.freeze_batchnorm:
            frozen_bn = freeze_batchnorm_modules(model)
            if epoch == start_epoch:
                print(f"BatchNorm congelado para fine-tuning: {frozen_bn} modulos", flush=True)
        is_resume_epoch = epoch == start_epoch and resume_batch_idx > 0
        if is_resume_epoch:
            total_loss, total_batches, nan_skips, train_preds, train_targs, resumed_train_seconds = (
                restore_partial_epoch_state(resume_partial_epoch_state)
            )
        else:
            total_loss = 0.0
            total_batches = 0
            nan_skips = 0
            train_preds = []
            train_targs = []
            resumed_train_seconds = 0.0
            resume_batch_idx = 0

        epoch_positions = build_epoch_sample_indices(
            train_size=len(train_indices),
            labels=train_labels,
            sampler=args.sampler,
            num_classes=active_num_classes,
            epoch=epoch,
            seed=args.seed,
            num_samples=sampler_num_samples,
            max_class_weight=args.max_class_weight,
            min_class_weight=args.min_class_weight,
            sampler_alpha=args.sampler_alpha,
        )
        start_sample = int(resume_batch_idx) * int(args.batch_size)
        remaining_positions = epoch_positions[start_sample:sampler_num_samples]
        if not remaining_positions:
            raise RuntimeError(
                f"Checkpoint parcial invalido: epoch={epoch} batch_idx={resume_batch_idx} "
                f"train_batches={train_batches_per_epoch}"
            )
        epoch_dataset_indices = [train_indices[int(pos)] for pos in remaining_positions]
        epoch_subset = Subset(train_dataset_all, epoch_dataset_indices)
        train_loader = DataLoader(
            epoch_subset,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=True,
            **loader_kwargs,
        )
        boundary_aux_iter = iter(boundary_aux_loader) if boundary_aux_loader is not None else None
        epoch_start = time.time()

        progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{args.epochs}",
            leave=True,
            mininterval=10,
            disable=not use_tqdm,
        )
        if is_resume_epoch:
            print(
                f"[RESUME] Reanudando epoca {epoch} desde batch {resume_batch_idx + 1}/{train_batches_per_epoch}",
                flush=True,
            )
        for local_batch_idx, batch in enumerate(progress, start=1):
            batch_idx = int(resume_batch_idx) + int(local_batch_idx)
            batch_x, batch_ry, batch_qy, batch_by, batch_paths, batch_morph, batch_window_keys = _unpack_batch(batch)
            batch_x = batch_x.to(device)
            batch_ry = batch_ry.to(device)
            batch_qy = batch_qy.to(device)
            batch_by = batch_by.to(device)
            if args.morphology_fusion:
                if batch_morph is None:
                    raise ValueError("morphology_fusion activo pero el batch no contiene morphology_features")
                batch_morph = batch_morph.to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                need_student_embedding = bool(offline_teacher_tables and float(args.teacher_embedding_weight) > 0.0)
                outputs = (
                    model(batch_x, morphology_features=batch_morph, return_features=need_student_embedding)
                    if args.morphology_fusion
                    else model(batch_x, return_features=need_student_embedding)
                )
                student_embedding = None
                if need_student_embedding:
                    *outputs, student_embedding = outputs
                if isinstance(outputs, (list, tuple)) and len(outputs) == 4:
                    out_r, out_q, out_b, out_p = outputs
                else:
                    out_r, out_q, out_b = outputs
                    out_p = None
                loss_r = criterion_rhythm(out_r, batch_ry)
                loss_b = masked_age_sex_loss(out_b, batch_by)
            out_q_safe = torch.clamp(out_q.float(), min=1e-7, max=1.0 - 1e-7)
            loss_q = criterion_quality(out_q_safe, batch_qy.float())
            loss = loss_r + 0.1 * loss_q + float(args.biometry_weight) * loss_b

            if (
                criterion_pathology is not None
                and args.pathology_weight > 0
                and out_p is not None
                and batch_paths is not None
                and pathology_record_to_multihot
                and pathology_num_classes > 0
            ):
                abs_paths = [os.path.abspath(p) for p in batch_paths]
                has = [p in pathology_record_to_multihot for p in abs_paths]
                if any(has):
                    y = np.zeros((len(abs_paths), pathology_num_classes), dtype=np.float32)
                    for i, p in enumerate(abs_paths):
                        vec = pathology_record_to_multihot.get(p)
                        if vec is not None:
                            y[i, :] = vec
                    y_t = torch.from_numpy(y).to(device=device, dtype=torch.float32)
                    mask = torch.tensor(has, device=device, dtype=torch.bool)
                    loss_p = criterion_pathology(out_p[mask], y_t[mask])
                    loss = loss + float(args.pathology_weight) * loss_p

            if teacher_model is not None:
                with torch.no_grad():
                    with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                        teacher_outputs = (
                            teacher_model(batch_x, morphology_features=batch_morph)
                            if args.morphology_fusion
                            else teacher_model(batch_x)
                        )
                        teacher_out_r = teacher_outputs[0] if isinstance(teacher_outputs, (list, tuple)) else teacher_outputs
                teacher_out_r = maybe_project_teacher_logits(
                    out_r,
                    teacher_out_r,
                    primary9_mode=bool(args.primary9_mode),
                )
                loss_teacher = distillation_kl_loss(out_r, teacher_out_r, args.teacher_temperature)
                loss = loss + float(args.teacher_kl_weight) * loss_teacher

            if offline_teacher_tables and float(args.teacher_logits_kl_weight) > 0.0:
                loss_offline, _offline_stats = offline_teacher_distillation_loss(
                    out_r,
                    batch_window_keys,
                    offline_teacher_tables,
                    primary9_mode=bool(args.primary9_mode),
                    temperature=float(args.teacher_logits_temperature),
                )
                if loss_offline is not None:
                    loss = loss + float(args.teacher_logits_kl_weight) * loss_offline

            if offline_teacher_tables and float(args.teacher_embedding_weight) > 0.0:
                loss_embed, _embed_stats = offline_teacher_embedding_loss(
                    student_embedding,
                    batch_window_keys,
                    offline_teacher_tables,
                )
                if loss_embed is not None:
                    loss = loss + float(args.teacher_embedding_weight) * loss_embed

            if boundary_aux_loader is not None and boundary_aux_iter is not None:
                aux_batch, boundary_aux_iter = next_cyclic_batch(boundary_aux_loader, boundary_aux_iter)
                aux_x, aux_ry, _aux_qy, _aux_by, _aux_paths, aux_morph, _aux_window_keys = _unpack_batch(aux_batch)
                aux_x = aux_x.to(device)
                aux_ry = aux_ry.to(device)
                if args.morphology_fusion:
                    if aux_morph is None:
                        raise ValueError("morphology_fusion activo pero el batch auxiliar no contiene morphology_features")
                    aux_morph = aux_morph.to(device)
                with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                    aux_outputs = model(aux_x, morphology_features=aux_morph) if args.morphology_fusion else model(aux_x)
                    if isinstance(aux_outputs, (list, tuple)):
                        aux_out_r = aux_outputs[0]
                    else:
                        aux_out_r = aux_outputs
                    loss_boundary = criterion_rhythm(aux_out_r, aux_ry)
                loss = combine_boundary_aux_loss(loss, loss_boundary, args.boundary_aux_weight)

            if torch.isnan(loss) or torch.isinf(loss):
                nan_skips += 1
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            global_step += 1

            total_loss += float(loss.item())
            total_batches += 1
            preds = torch.argmax(out_r, dim=1)
            train_preds.append(preds.detach().cpu())
            train_targs.append(batch_ry.detach().cpu())
            if use_tqdm:
                progress.set_postfix(loss=f"{loss.item():.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")
            elif batch_idx == 1 or batch_idx % args.log_interval == 0 or batch_idx == train_batches_per_epoch:
                elapsed_epoch = time.time() - epoch_start
                segment_batches = max(batch_idx - int(resume_batch_idx), 1)
                batches_per_second = segment_batches / max(elapsed_epoch, 1e-9)
                eta_minutes = (train_batches_per_epoch - batch_idx) / max(batches_per_second, 1e-9) / 60.0
                memory = cgroup_memory_snapshot()
                print(
                    f"Epoch {epoch:02d}/{args.epochs} batch {batch_idx}/{train_batches_per_epoch} | "
                    f"loss={loss.item():.4f} lr={scheduler.get_last_lr()[0]:.2e} | "
                    f"{batches_per_second:.2f} batch/s ETA={eta_minutes:.1f} min | "
                    f"{format_memory_snapshot(memory)}",
                    flush=True,
                )

            if (
                args.step_checkpoint_interval > 0
                and batch_idx % max(args.step_checkpoint_interval, 1) == 0
                and batch_idx < train_batches_per_epoch
            ):
                partial_state = build_partial_epoch_state(
                    total_loss=total_loss,
                    total_batches=total_batches,
                    nan_skips=nan_skips,
                    train_preds=train_preds,
                    train_targs=train_targs,
                    train_epoch_seconds=resumed_train_seconds + (time.time() - epoch_start),
                )
                save_training_checkpoint(
                    args.step_checkpoint_path or os.path.join(out_dir, "checkpoint_step.pt"),
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    epoch=epoch,
                    batch_idx=batch_idx,
                    global_step=global_step,
                    epoch_complete=False,
                    partial_epoch_state=partial_state,
                    best_val_f1=best_val_f1,
                    best_val_score=best_val_score,
                    best_epoch=best_epoch,
                    stale_epochs=stale_epochs,
                    training_log=training_log,
                    args=args,
                )
                print(
                    f"[CHECKPOINT_STEP] epoch={epoch} batch={batch_idx}/{train_batches_per_epoch} "
                    f"global_step={global_step}",
                    flush=True,
                )

            if (
                args.stop_after_batches > 0
                and local_batch_idx >= args.stop_after_batches
                and batch_idx < train_batches_per_epoch
            ):
                partial_state = build_partial_epoch_state(
                    total_loss=total_loss,
                    total_batches=total_batches,
                    nan_skips=nan_skips,
                    train_preds=train_preds,
                    train_targs=train_targs,
                    train_epoch_seconds=resumed_train_seconds + (time.time() - epoch_start),
                )
                stop_path = args.step_checkpoint_path or os.path.join(out_dir, "checkpoint_step.pt")
                save_training_checkpoint(
                    stop_path,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    epoch=epoch,
                    batch_idx=batch_idx,
                    global_step=global_step,
                    epoch_complete=False,
                    partial_epoch_state=partial_state,
                    best_val_f1=best_val_f1,
                    best_val_score=best_val_score,
                    best_epoch=best_epoch,
                    stale_epochs=stale_epochs,
                    training_log=training_log,
                    args=args,
                )
                print(
                    f"[CONTROLLED_STOP] epoch={epoch} batch={batch_idx}/{train_batches_per_epoch} "
                    f"global_step={global_step} checkpoint={stop_path}",
                    flush=True,
                )
                return model, training_log

            if (
                args.memory_guard_percent > 0
                and batch_idx % max(args.memory_guard_interval, 1) == 0
            ):
                memory = cgroup_memory_snapshot()
                if memory and memory.get("max") and memory["max"] < 10 ** 15:
                    used_percent = 100.0 * memory["current"] / max(memory["max"], 1)
                    if used_percent >= args.memory_guard_percent:
                        state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
                        emergency_path = os.path.join(out_dir, "titan_v4_lite_weights_emergency_memory_guard.pth")
                        torch.save(state, emergency_path)
                        raise RuntimeError(
                            "Memory guard triggered: "
                            f"{used_percent:.1f}% >= {args.memory_guard_percent:.1f}%. "
                            f"Checkpoint guardado en {emergency_path}"
                        )

        train_epoch_seconds = resumed_train_seconds + (time.time() - epoch_start)
        if not train_preds:
            raise RuntimeError(f"Epoca {epoch}: no hubo batches validos")

        train_y_pred = torch.cat(train_preds).numpy()
        train_y_true = torch.cat(train_targs).numpy()
        train_loss = total_loss / max(total_batches, 1)
        train_acc = accuracy_score(train_y_true, train_y_pred)
        train_f1 = f1_score(train_y_true, train_y_pred, average="macro", zero_division=0)

        val_start = time.time()
        val = evaluate_epoch(
            model,
            val_loader,
            device,
            criterion_rhythm,
            criterion_quality,
            criterion_bio,
            criterion_pathology=criterion_pathology,
            pathology_record_to_multihot=pathology_record_to_multihot,
            pathology_num_classes=pathology_num_classes,
            pathology_class_names=pathology_class_names,
            pathology_weight=float(args.pathology_weight) if args.pathology_head else 0.0,
            biometry_weight=float(args.biometry_weight),
            morphology_fusion=args.morphology_fusion,
        )
        val_nsr_recall = (
            recall_for_class(val.get("y_true", []), val.get("y_pred", []), nsr_class_idx)
            if nsr_class_idx is not None
            else float("nan")
        )
        if baseline_nsr_recall is None and val_nsr_recall == val_nsr_recall:
            baseline_nsr_recall = val_nsr_recall
        val_epoch_seconds = time.time() - val_start
        epoch_seconds = train_epoch_seconds + val_epoch_seconds
        train_bps = train_batches_per_epoch / max(train_epoch_seconds, 1e-9)
        val_bps = len(val_loader) / max(val_epoch_seconds, 1e-9)
        full_train_batches = full_train_windows // args.batch_size
        full_val_batches = math.ceil(full_val_windows / args.batch_size)
        estimated_full_epoch_seconds = (
            full_train_batches / max(train_bps, 1e-9)
            + full_val_batches / max(val_bps, 1e-9)
        )
        elapsed = time.time() - start_time
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "train_f1_macro": train_f1,
            "val_loss": val["loss"],
            "val_acc": val["acc"],
            "val_f1_macro": val["f1_macro"],
            "val_f1_weighted": val["f1_weighted"],
            "val_nsr_recall": val_nsr_recall,
            "nan_skips": nan_skips,
            "val_nonfinite_loss_batches": val.get("nonfinite_loss_batches", 0),
            "lr": scheduler.get_last_lr()[0],
            "elapsed": elapsed,
            "train_epoch_seconds": train_epoch_seconds,
            "val_epoch_seconds": val_epoch_seconds,
            "epoch_seconds": epoch_seconds,
            "train_batch_per_sec": train_bps,
            "val_batch_per_sec": val_bps,
            "estimated_full_epoch_seconds": estimated_full_epoch_seconds,
        }
        if args.pathology_head:
            row.update(
                {
                    "pathology_weight": float(args.pathology_weight),
                    "val_pathology_pr_auc_macro": val.get("pathology_pr_auc_macro", float("nan")),
                    "val_pathology_f1_macro": val.get("pathology_f1_macro", float("nan")),
                    "val_pathology_labeled_windows": val.get("pathology_labeled_windows", 0),
                    "val_pathology_used_classes": val.get("pathology_used_classes", 0),
                    "val_pathology_thresholds": val.get("pathology_thresholds", {}),
                }
            )
        row.update(
            {
                "biometry_weight": float(args.biometry_weight),
                "val_biometry_sex_valid": val.get("biometry_sex_valid", 0),
                "val_biometry_sex_acc": val.get("biometry_sex_acc", float("nan")),
                "val_biometry_sex_f1": val.get("biometry_sex_f1", float("nan")),
                "val_biometry_age_valid": val.get("biometry_age_valid", 0),
                "val_biometry_age_mae_years": val.get("biometry_age_mae_years", float("nan")),
                "val_biometry_age_within5_acc": val.get("biometry_age_within5_acc", float("nan")),
                "val_biometry_age_within10_acc": val.get("biometry_age_within10_acc", float("nan")),
            }
        )
        training_log.append(row)
        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_f1={train_f1:.4f} train_acc={train_acc:.2%} | "
            f"val_loss={val['loss']:.4f} val_f1={val['f1_macro']:.4f} val_acc={val['acc']:.2%} | "
            f"NSR_recall={val_nsr_recall:.3f} NaN={nan_skips} val_nonfinite={val.get('nonfinite_loss_batches', 0)} | "
            f"train={train_bps:.2f} batch/s val={val_bps:.2f} batch/s | "
            f"full_epoch_eta={estimated_full_epoch_seconds / 60.0:.1f} min"
        )
        append_benchmark_row(
            args.benchmark_csv,
            {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "num_workers": args.num_workers,
                "prefetch_factor": loader_kwargs.get("prefetch_factor", ""),
                "persistent_workers": loader_kwargs.get("persistent_workers", False),
                "pin_memory": loader_kwargs["pin_memory"],
                "batch_size": args.batch_size,
                "max_train_windows": args.max_train_windows,
                "max_val_windows": args.max_val_windows,
                "train_windows_used": len(train_indices),
                "val_windows_used": len(val_indices),
                "full_train_windows": full_train_windows,
                "full_val_windows": full_val_windows,
                "train_batches": train_batches_per_epoch,
                "val_batches": len(val_loader),
                "train_batch_per_sec": f"{train_bps:.6f}",
                "val_batch_per_sec": f"{val_bps:.6f}",
                "epoch_seconds": f"{epoch_seconds:.3f}",
                "estimated_full_epoch_seconds": f"{estimated_full_epoch_seconds:.3f}",
                "val_f1_macro": f"{val['f1_macro']:.6f}",
                "val_acc": f"{val['acc']:.6f}",
            },
        )

        val_score = float(val["f1_macro"])
        if args.pathology_head and args.checkpoint_metric == "combined":
            pr_auc = val.get("pathology_pr_auc_macro", float("nan"))
            if pr_auc == pr_auc:  # not NaN
                val_score = val_score + float(args.pathology_combined_alpha) * float(pr_auc)
        elif args.checkpoint_metric == "multi_task":
            val_score = multi_task_checkpoint_score(
                val,
                pathology_alpha=float(args.pathology_combined_alpha),
                biometry_alpha=float(args.biometry_combined_alpha),
            )

        improved = val_score > best_val_score + args.min_delta
        if improved:
            best_val_score = val_score
            best_val_f1 = float(val["f1_macro"])
            best_epoch = epoch
            stale_epochs = 0
            torch.save(_model_state_dict(model), os.path.join(out_dir, "titan_v4_lite_weights_best.pth"))
            save_validation_artifacts(val["y_true"], val["y_pred"], out_dir, active_rhythm_names)
            if args.checkpoint_metric in {"combined", "multi_task"}:
                print(f"  Nuevo mejor checkpoint por score={best_val_score:.4f} (rhythm_f1={best_val_f1:.4f})")
            else:
                print(f"  Nuevo mejor checkpoint por val_f1={best_val_f1:.4f}")
        else:
            stale_epochs += 1
        should_stop = stale_epochs >= args.patience
        if (
            args.nsr_recall_collapse_tolerance > 0
            and baseline_nsr_recall is not None
            and baseline_nsr_recall == baseline_nsr_recall
            and val_nsr_recall == val_nsr_recall
            and val_nsr_recall < baseline_nsr_recall - float(args.nsr_recall_collapse_tolerance)
        ):
            should_stop = True
            print(
                "[EarlyStop] NSR recall colapso: "
                f"baseline={baseline_nsr_recall:.3f} actual={val_nsr_recall:.3f}",
                flush=True,
            )
        if should_stop:
            print(f"Early stopping: {stale_epochs} epocas sin mejorar val_f1")

        pathology_summary = None
        if args.pathology_head:
            pathology_summary = {
                "pathology_head": True,
                "pathology_num_classes": pathology_num_classes,
                "pathology_class_names": pathology_class_names,
                "pathology_weight": float(args.pathology_weight),
                "pathology_sidecar": pathology_sidecar_used,
                "pathology_records_labelled": len(pathology_record_to_multihot),
                "pathology_combined_alpha": float(args.pathology_combined_alpha),
                "pathology_pos_weight": pathology_pos_weights,
                "pathology_pos_weight_stats": pathology_pos_weight_stats,
            }
        biometry_summary = {
            "biometry_head": True,
            "biometry_targets": ["sex", "age"],
            "biometry_weight": float(args.biometry_weight),
            "biometry_combined_alpha": float(args.biometry_combined_alpha),
            "freeze_for_auxiliary_heads": bool(args.freeze_for_auxiliary_heads),
            "auxiliary_freeze_summary": auxiliary_freeze_summary,
        }
        write_training_summary(
            os.path.join(out_dir, "training_summary.json"),
            best_epoch=best_epoch,
            best_val_f1=best_val_f1,
            best_val_score=best_val_score,
            checkpoint_metric=args.checkpoint_metric,
            training_log=training_log,
            train_windows=len(train_indices),
            val_windows=len(val_indices),
            full_train_windows=full_train_windows,
            full_val_windows=full_val_windows,
            sampler=args.sampler,
            train_samples_per_epoch=sampler_num_samples,
            balanced_samples_per_class=args.balanced_samples_per_class,
            class_weights=class_weights,
            loss_class_weight_mode=args.loss_class_weight_mode,
            loss_class_weights=loss_class_weights,
            num_workers=args.num_workers,
            pin_memory=loader_kwargs["pin_memory"],
            persistent_workers=loader_kwargs.get("persistent_workers", False),
            prefetch_factor=loader_kwargs.get("prefetch_factor", None),
            initial_val_metrics=initial_val_metrics,
            pathology_summary=pathology_summary,
            biometry_summary=biometry_summary,
            boundary_aux_summary=boundary_aux_summary,
            teacher_distillation_summary=teacher_distillation_summary,
            offline_teacher_summary=offline_teacher_summary,
            morphology_fusion=args.morphology_fusion,
            morphology_dim=MORPHOLOGY_FEATURE_DIM if args.morphology_fusion else 0,
        )
        with open(os.path.join(out_dir, "training_log.json"), "w", encoding="utf-8") as handle:
            json.dump(training_log, handle, indent=2)
        save_training_checkpoint(
            os.path.join(out_dir, "checkpoint_last.pt"),
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            best_val_f1=best_val_f1,
            best_val_score=best_val_score,
            best_epoch=best_epoch,
            stale_epochs=stale_epochs,
            training_log=training_log,
            args=args,
            batch_idx=0,
            global_step=global_step,
            epoch_complete=True,
            partial_epoch_state=None,
        )
        plot_training_log(training_log, out_dir)
        if should_stop:
            break

    torch.save(_model_state_dict(model), os.path.join(out_dir, "titan_v4_lite_weights_final.pth"))
    with open(os.path.join(out_dir, "training_log.json"), "w", encoding="utf-8") as handle:
        json.dump(training_log, handle, indent=2)
    plot_training_log(training_log, out_dir)

    pathology_summary = None
    if args.pathology_head:
        pathology_summary = {
            "pathology_head": True,
            "pathology_num_classes": pathology_num_classes,
            "pathology_class_names": pathology_class_names,
            "pathology_weight": float(args.pathology_weight),
            "pathology_sidecar": pathology_sidecar_used,
            "pathology_records_labelled": len(pathology_record_to_multihot),
            "pathology_combined_alpha": float(args.pathology_combined_alpha),
            "pathology_pos_weight": pathology_pos_weights,
            "pathology_pos_weight_stats": pathology_pos_weight_stats,
        }
    biometry_summary = {
        "biometry_head": True,
        "biometry_targets": ["sex", "age"],
        "biometry_weight": float(args.biometry_weight),
        "biometry_combined_alpha": float(args.biometry_combined_alpha),
        "freeze_for_auxiliary_heads": bool(args.freeze_for_auxiliary_heads),
        "auxiliary_freeze_summary": auxiliary_freeze_summary,
    }
    write_training_summary(
        os.path.join(out_dir, "training_summary.json"),
        best_epoch=best_epoch,
        best_val_f1=best_val_f1,
        best_val_score=best_val_score,
        checkpoint_metric=args.checkpoint_metric,
        training_log=training_log,
        train_windows=len(train_indices),
        val_windows=len(val_indices),
        full_train_windows=full_train_windows,
        full_val_windows=full_val_windows,
        sampler=args.sampler,
        train_samples_per_epoch=sampler_num_samples,
        balanced_samples_per_class=args.balanced_samples_per_class,
        class_weights=class_weights,
        loss_class_weight_mode=args.loss_class_weight_mode,
        loss_class_weights=loss_class_weights,
        num_workers=args.num_workers,
        pin_memory=loader_kwargs["pin_memory"],
        persistent_workers=loader_kwargs.get("persistent_workers", False),
        prefetch_factor=loader_kwargs.get("prefetch_factor", None),
        initial_val_metrics=initial_val_metrics,
        pathology_summary=pathology_summary,
        biometry_summary=biometry_summary,
        boundary_aux_summary=boundary_aux_summary,
        teacher_distillation_summary=teacher_distillation_summary,
        offline_teacher_summary=offline_teacher_summary,
        morphology_fusion=args.morphology_fusion,
        morphology_dim=MORPHOLOGY_FEATURE_DIM if args.morphology_fusion else 0,
    )

    print("=" * 72)
    print("ENTRENAMIENTO COMPLETADO")
    print(f"Mejor epoca: {best_epoch} | Mejor val F1 macro: {best_val_f1:.4f}")
    print(f"Artefactos guardados en: {out_dir}")
    print("=" * 72)
    return model, training_log


def parse_args():
    parser = argparse.ArgumentParser(description="TITAN V4 - entrenamiento CEDIA con validacion real")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--base_lr", type=float, default=1e-4)
    parser.add_argument("--max_lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=2.0)
    parser.add_argument("--label_smoothing", type=float, default=0.05)
    parser.add_argument("--val_fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--min_delta", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--log_interval", type=int, default=500)
    parser.add_argument("--max_train_windows", type=int, default=0)
    parser.add_argument("--max_val_windows", type=int, default=0)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--persistent_workers", action="store_true")
    parser.add_argument("--no_pin_memory", action="store_true")
    parser.add_argument("--benchmark_csv", default=None)
    parser.add_argument("--memory_guard_percent", type=float, default=0.0)
    parser.add_argument("--memory_guard_interval", type=int, default=200)
    parser.add_argument("--sampler", choices=["balanced", "weighted", "tempered", "shuffle"], default="balanced")
    parser.add_argument("--sampler_alpha", type=float, default=0.5)
    parser.add_argument("--loss_class_weight_mode", choices=["auto", "inverse", "none"], default="auto")
    parser.add_argument("--loss_class_weights", default=None)
    parser.add_argument("--balanced_samples_per_class", type=int, default=4096)
    parser.add_argument("--max_class_weight", type=float, default=20.0)
    parser.add_argument("--min_class_weight", type=float, default=0.05)
    parser.add_argument("--min_train_records_per_class", type=int, default=10)
    parser.add_argument("--min_val_records_per_class", type=int, default=2)
    parser.add_argument("--min_train_windows_per_class", type=int, default=200)
    parser.add_argument("--min_val_windows_per_class", type=int, default=40)
    parser.add_argument("--allow_incomplete_classes", action="store_true")
    parser.add_argument("--data_root", default=None)
    parser.add_argument(
        "--extra_data_path",
        action="append",
        default=[],
        help="Ruta adicional escaneada por el dataset, usada para pools auxiliares fuera de DATA.",
    )
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--train_manifest_csv", default=None)
    parser.add_argument("--val_manifest_csv", default=None)
    parser.add_argument(
        "--primary9_mode",
        action="store_true",
        help="Entrena/evalua el head primario de 9 arritmias reportables.",
    )
    parser.add_argument("--boundary_aux_train_manifest_csv", default=None)
    parser.add_argument("--boundary_aux_val_manifest_csv", default=None)
    parser.add_argument("--boundary_aux_weight", type=float, default=0.0)
    parser.add_argument("--boundary_aux_batch_size", type=int, default=0)
    parser.add_argument("--boundary_aux_max_windows", type=int, default=0)
    parser.add_argument("--test_mode", action="store_true")
    parser.add_argument("--pathology_head", action="store_true")
    parser.add_argument("--pathology_sidecar", default=None)
    parser.add_argument("--pathology_weight", type=float, default=0.2)
    parser.add_argument("--max_pathology_pos_weight", type=float, default=20.0)
    parser.add_argument(
        "--morphology_fusion",
        action="store_true",
        help="Concatena 10 rasgos morfologicos deterministas al embedding de ritmo.",
    )
    parser.add_argument("--checkpoint_metric", choices=["rhythm_f1", "combined", "multi_task"], default="rhythm_f1")
    parser.add_argument("--pathology_combined_alpha", type=float, default=0.1)
    parser.add_argument("--biometry_weight", type=float, default=0.2)
    parser.add_argument("--biometry_combined_alpha", type=float, default=0.05)
    parser.add_argument("--resume", default=None, help="Checkpoint completo checkpoint_last.pt para reanudar entrenamiento.")
    parser.add_argument("--step_checkpoint_interval", type=int, default=0)
    parser.add_argument("--step_checkpoint_path", default=None)
    parser.add_argument(
        "--stop_after_batches",
        type=int,
        default=0,
        help="Corta la ejecucion tras N batches nuevos, guardando checkpoint parcial para relanzar por tramos.",
    )
    parser.add_argument("--init_weights", default=None, help="Pesos .pth para inicializar sin restaurar optimizer/scheduler.")
    parser.add_argument("--initial_best_score", type=float, default=None)
    parser.add_argument("--initial_best_f1", type=float, default=None)
    parser.add_argument("--eval_init_weights", action="store_true")
    parser.add_argument("--freeze_batchnorm", action="store_true")
    parser.add_argument("--freeze_for_auxiliary_heads", action="store_true")
    parser.add_argument("--train_quality_head_with_auxiliary", action="store_true")
    parser.add_argument("--freeze_early_backbone_epochs", type=int, default=0)
    parser.add_argument("--train_last_n_resblocks", type=int, default=1)
    parser.add_argument("--nsr_recall_collapse_tolerance", type=float, default=0.0)
    parser.add_argument("--teacher_weights", default=None, help="Pesos del modelo activo usado como teacher de consistencia.")
    parser.add_argument("--teacher_kl_weight", type=float, default=0.0)
    parser.add_argument("--teacher_temperature", type=float, default=2.0)
    parser.add_argument(
        "--teacher_logits_npz",
        action="append",
        default=[],
        help="NPZ con logits/embeddings de teacher precomputados por window_key. Puede repetirse.",
    )
    parser.add_argument(
        "--teacher_logits_weight",
        default=None,
        help="Pesos separados por coma para cada teacher_logits_npz; por defecto todos 1.0.",
    )
    parser.add_argument("--teacher_logits_kl_weight", type=float, default=0.0)
    parser.add_argument("--teacher_logits_temperature", type=float, default=2.0)
    parser.add_argument("--teacher_embedding_weight", type=float, default=0.0)
    args = parser.parse_args()

    if args.test_mode:
        args.epochs = 3
        args.batch_size = 64
        args.num_workers = 0
        args.patience = 3
    return args


if __name__ == "__main__":
    train_v4_model(parse_args())
