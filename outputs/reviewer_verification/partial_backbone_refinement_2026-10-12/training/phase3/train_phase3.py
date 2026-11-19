"""Warm-start the original CNN's last residual block and rhythm head on CEDIA."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import socket
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import contract as c

ALLOWED_PREFIXES = ("res_layers.4.", "head_rhythm.")
PROTECTED_CLASSES = ("SB", "LBBB", "RBBB", "PAC", "NSR")
PROTOCOL = json.loads((Path(__file__).resolve().parent / "protocol.json").read_text(encoding="utf-8"))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def configure_partial(model: nn.Module) -> list[nn.Parameter]:
    model.eval()
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith(ALLOWED_PREFIXES))
    model.res_layers[4].train()
    for module in model.res_layers[4].modules():
        if isinstance(module, nn.BatchNorm1d):
            module.eval()
    model.head_rhythm.train()
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def tensor_hashes_match(data: dict[str, np.ndarray], role: str) -> None:
    expected = data["tensor_sha256"]
    for index, (tensor, digest) in enumerate(zip(data["x"], expected, strict=True)):
        actual = hashlib.sha256(np.asarray(tensor, dtype="<f4").tobytes()).hexdigest()
        if actual != str(digest):
            raise ValueError(f"{role}: transferred raw tensor hash mismatch at row {index}")


def verify_gold_embeddings(model: nn.Module, data: dict[str, np.ndarray], role: str) -> float:
    model.eval()
    maximum_error = 0.0
    x = torch.from_numpy(data["x"])
    with torch.inference_mode():
        for start in range(0, len(x), 128):
            stop = min(start + 128, len(x))
            features = model.forward_features(x[start:stop].cuda()).cpu().numpy()
            difference = np.abs(features - data["embedding"][start:stop])
            maximum_error = max(maximum_error, float(difference.max(initial=0.0)))
            if not np.allclose(features, data["embedding"][start:stop], atol=2e-4, rtol=2e-4):
                raise ValueError(f"{role}: recomputed Gold embeddings differ from frozen feature cache")
    return maximum_error


def model_outputs(model: nn.Module, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
    outputs = model(x)
    if len(outputs) != 4:
        raise ValueError("Expected rhythm, quality, biometrics, and pathology outputs")
    return outputs


def cache_outputs(model: nn.Module, data: dict[str, np.ndarray]) -> tuple[np.ndarray, ...]:
    model.eval()
    values: list[list[np.ndarray]] = [[], [], [], []]
    x = torch.from_numpy(data["x"])
    with torch.inference_mode():
        for start in range(0, len(x), 128):
            outputs = model_outputs(model, x[start:start + 128].cuda())
            for bucket, output in zip(values, outputs, strict=True):
                bucket.append(output.float().cpu().numpy())
    return tuple(np.concatenate(parts, axis=0) for parts in values)


def normalized_auxiliary_errors(outputs: tuple[np.ndarray, ...], reference: tuple[np.ndarray, ...]) -> dict[str, float]:
    result: dict[str, float] = {}
    for name, output, target in zip(("quality", "biometrics", "pathology"), outputs[1:], reference[1:], strict=True):
        scale = np.maximum(target.std(axis=0), 0.05)
        result[name] = float(np.sqrt(np.mean(((output - target) / scale) ** 2)))
    return result


def predict(model: nn.Module, data: dict[str, np.ndarray]) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    outputs = cache_outputs(model, data)
    probabilities = torch.softmax(torch.from_numpy(outputs[0]), dim=1).numpy()
    c.validate_probabilities(probabilities)
    return probabilities, outputs


def protected_recall_floor(
    data: dict[str, np.ndarray], probabilities: np.ndarray, starting_probabilities: np.ndarray,
    minimum_support: int,
) -> tuple[bool, dict[str, dict[str, float]]]:
    candidate = c.source_metrics(data, probabilities)
    starting = c.source_metrics(data, starting_probabilities)
    checks: dict[str, dict[str, float]] = {}
    eligible = True
    for source, metrics in candidate.items():
        for label in PROTECTED_CLASSES:
            support = metrics["support"][label]
            if support < minimum_support:
                continue
            before = starting[source]["recall"][label]
            after = metrics["recall"][label]
            checks[f"{source}:{label}"] = {"support": support, "starting_recall": before, "candidate_recall": after}
            eligible &= after >= before - PROTOCOL["selection"]["max_protected_recall_drop"] - 1e-12
    return bool(eligible), checks


def metric_record(data: dict[str, np.ndarray], probabilities: np.ndarray, gold_probabilities: np.ndarray,
                  start_probabilities: np.ndarray, aux_errors: dict[str, float]) -> dict[str, object]:
    score, preserve_gold, sources = c.validation_score(data, probabilities, gold_probabilities)
    preserve_start, recall_checks = protected_recall_floor(
        data, probabilities, start_probabilities, PROTOCOL["selection"]["minimum_class_support"])
    auxiliary_ok = max(aux_errors.values(), default=0.0) <= PROTOCOL["selection"]["maximum_auxiliary_nrmse"]
    return {
        "score_source_mean_macro_f1": score,
        "eligible": preserve_gold and preserve_start and auxiliary_ok,
        "criteria": {"preserve_gold_SB_LBBB": preserve_gold, "preserve_start_weak_classes": preserve_start,
                     "preserve_auxiliary_outputs": auxiliary_ok},
        "protected_recall_checks": recall_checks,
        "auxiliary_normalized_rmse": aux_errors,
        "metrics": c.metrics(data["y"], probabilities),
        "sources": sources,
    }


def frozen_state_check(model: nn.Module, initial: dict[str, torch.Tensor],
                       allowed_parameters: set[str] | None = None) -> list[str]:
    state = model.state_dict()
    if set(state) != set(initial):
        raise ValueError("Model state keys changed")
    trainable = allowed_parameters or {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    changed: list[str] = []
    for name, value in state.items():
        if not torch.equal(value.detach().cpu(), initial[name].detach().cpu()):
            changed.append(name)
            if name not in trainable:
                raise ValueError(f"Frozen parameter or buffer changed: {name}")
    return changed


def run_trial(
    trial: dict[str, float], epoch_limit: int, train: dict[str, np.ndarray], validation: dict[str, np.ndarray],
    starting_state: dict[str, torch.Tensor], train_reference: tuple[np.ndarray, ...],
    validation_reference: tuple[np.ndarray, ...], starting_validation_probabilities: np.ndarray,
    output: Path | None,
) -> tuple[dict[int, dict[str, torch.Tensor]], list[dict[str, object]]]:
    set_seed(PROTOCOL["optimization"]["seed"])
    model = c.load_model(c.ROOT / "phase2_selected.pth").cuda()
    parameters = configure_partial(model)
    if sum(parameter.numel() for parameter in parameters) != PROTOCOL["architecture"]["trainable_parameter_count"]:
        raise ValueError("Trainable parameter contract changed")
    optimizer = torch.optim.AdamW(parameters, lr=trial["learning_rate"], weight_decay=0.01)
    x = torch.from_numpy(train["x"]).cuda()
    y = torch.from_numpy(train["y"]).long().cuda()
    teacher = torch.from_numpy(train["teacher"]).cuda()
    gold = torch.from_numpy(train["gold"]).cuda()
    weights = torch.from_numpy(c.sample_weights(train)).cuda()
    focus = torch.ones_like(weights)
    focus[y == c.CLASSES.index("SB")] = PROTOCOL["optimization"]["sb_lbbb_sample_weight"]
    focus[y == c.CLASSES.index("LBBB")] = PROTOCOL["optimization"]["sb_lbbb_sample_weight"]
    focus[y == c.CLASSES.index("RBBB")] = trial["weak_class_multiplier"]
    focus[y == c.CLASSES.index("PAC")] = trial["weak_class_multiplier"]
    weights = weights * focus
    weights = weights / weights.mean()
    gold_anchor = trial["gold_anchor"]
    start_train_aux = tuple(torch.from_numpy(value).cuda() for value in train_reference[1:])
    aux_scales = tuple(value.std(dim=0, unbiased=False).clamp_min(0.05) for value in start_train_aux)
    history: list[dict[str, object]] = []
    states: dict[int, dict[str, torch.Tensor]] = {}
    checkpoints = set(PROTOCOL["optimization"]["checkpoints"])
    for epoch in range(1, epoch_limit + 1):
        configure_partial(model)
        order = torch.randperm(len(y), device="cuda")
        total = 0.0
        for start in range(0, len(y), PROTOCOL["optimization"]["batch_size"]):
            index = order[start:start + PROTOCOL["optimization"]["batch_size"]]
            rhythm, quality, biometrics, pathology = model_outputs(model, x[index])
            logp = F.log_softmax(rhythm, dim=1)
            ce = F.cross_entropy(rhythm, y[index], reduction="none")
            teacher_kl = F.kl_div(logp, teacher[index], reduction="none").sum(dim=1)
            gold_kl = F.kl_div(logp, gold[index], reduction="none").sum(dim=1)
            auxiliary_predictions = (quality, biometrics, pathology)
            auxiliary_loss = sum(
                (((prediction - target[index]) / scale) ** 2).mean()
                for prediction, target, scale in zip(auxiliary_predictions, start_train_aux, aux_scales, strict=True)
            )
            loss = ((ce + PROTOCOL["optimization"]["teacher_kl_weight"] * teacher_kl + gold_anchor * gold_kl) * weights[index]).mean()
            loss = loss + PROTOCOL["optimization"]["auxiliary_head_distillation_weight"] * auxiliary_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            total += float(loss.detach()) * len(index)
        changed = frozen_state_check(model, starting_state)
        history.append({"epoch": epoch, "mean_loss": total / len(y), "changed_state_tensors": changed})
        if epoch not in checkpoints:
            continue
        probabilities, outputs = predict(model, validation)
        aux_errors = normalized_auxiliary_errors(outputs, validation_reference)
        result = metric_record(validation, probabilities, validation["gold"],
                               starting_validation_probabilities, aux_errors)
        name = (f"partial_lr{trial['learning_rate']}_anchor{gold_anchor}_focus"
                f"{trial['weak_class_multiplier']}_e{epoch:02d}")
        state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        states[epoch] = state
        row = {"name": name, **trial, "epoch": epoch, **result}
        print(json.dumps({"candidate": name, "score": row["score_source_mean_macro_f1"],
                          "eligible": row["eligible"], "accuracy": row["metrics"]["accuracy"],
                          "macro_f1": row["metrics"]["macro_f1"],
                          "auxiliary_nrmse": row["auxiliary_normalized_rmse"]}), flush=True)
        if output is not None:
            torch.save({"state_dict": state, "classes": c.CLASSES, "parameters": c.PARAMETERS},
                       output / f"{name}.pth")
            c.save(output / f"{name}.json", row)
            np.savez_compressed(output / f"{name}_validation.npz", probabilities=probabilities,
                                y=validation["y"], dataset=validation["dataset"], ecg_id=validation["ecg_id"])
    return states, history


def labels_metrics(y: np.ndarray, labels: np.ndarray) -> dict[str, object]:
    return c.metrics(y, np.eye(9, dtype=np.float32)[labels])


def evaluate_follow_up(model: nn.Module, data: dict[str, np.ndarray], output: Path) -> dict[str, object]:
    probabilities, _ = predict(model, data)
    prediction = probabilities.argmax(axis=1)
    rows: list[dict[str, object]] = []
    for index in range(len(prediction)):
        row: dict[str, object] = {
            "cohort": str(data["cohort"][index]), "dataset": str(data["dataset"][index]),
            "ecg_id": str(data["ecg_id"][index]), "reference": c.CLASSES[int(data["y"][index])],
            "phase3_prediction": c.CLASSES[int(prediction[index])],
            "phase2_prediction": c.CLASSES[int(data["phase2_label"][index])],
            "previous_partial_prediction": c.CLASSES[int(data["previous_partial_label"][index])],
            "gold_prediction": c.CLASSES[int(data["gold_label"][index])],
            "tensor_sha256": str(data["tensor_sha256"][index]),
        }
        row.update({f"prob_{label}": float(probabilities[index, j]) for j, label in enumerate(c.CLASSES)})
        rows.append(row)
    with (output / "followup_predictions.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    cohort_rows: dict[str, object] = {}
    for cohort in sorted(set(data["cohort"].tolist())):
        index = data["cohort"] == cohort
        y = data["y"][index]
        phase2 = data["phase2_label"][index]
        previous = data["previous_partial_label"][index]
        gold = data["gold_label"][index]
        cohort_rows[str(cohort)] = {
            "records": int(index.sum()),
            "phase3": c.metrics(y, probabilities[index]),
            "phase2": labels_metrics(y, phase2),
            "previous_partial": labels_metrics(y, previous),
            "gold": labels_metrics(y, gold),
        }
    return {"cohorts": cohort_rows, "records": len(prediction),
            "role": "previously examined follow-up; not used for model selection"}


def main() -> None:
    c.require_cedia_gpu()
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(2)
    if (c.ROOT / "results").exists():
        raise FileExistsError("Results directory exists; refusing to overwrite")
    for name, expected in json.loads((c.ROOT / "INPUT_SHA256.json").read_text(encoding="utf-8")).items():
        if c.digest(c.ROOT / name) != expected:
            raise ValueError(f"Frozen input hash mismatch: {name}")
    train = dict(np.load(c.ROOT / "train.npz", allow_pickle=False))
    validation = dict(np.load(c.ROOT / "validation.npz", allow_pickle=False))
    evaluation = dict(np.load(c.ROOT / "evaluation.npz", allow_pickle=False))
    for role, data in (("train", train), ("validation", validation), ("evaluation", evaluation)):
        tensor_hashes_match(data, role)
    gold_model = c.load_model(c.ROOT / "gold.pth").cuda()
    start_model = c.load_model(c.ROOT / "phase2_selected.pth").cuda()
    if c.parameter_count(gold_model) != c.PARAMETERS or c.parameter_count(start_model) != c.PARAMETERS:
        raise ValueError("Original model parameter count changed")
    c.check_frozen(start_model, gold_model.state_dict())
    initial_state = {key: value.detach().cpu().clone() for key, value in start_model.state_dict().items()}
    max_train_feature_error = verify_gold_embeddings(gold_model, train, "train")
    max_validation_feature_error = verify_gold_embeddings(gold_model, validation, "validation")
    train_reference = cache_outputs(start_model, train)
    validation_reference = cache_outputs(start_model, validation)
    gold_validation_probabilities = validation["gold"]
    starting_validation_probabilities = torch.softmax(torch.from_numpy(validation_reference[0]), dim=1).numpy()
    start_score, start_gold_eligible, start_sources = c.validation_score(
        validation, starting_validation_probabilities, gold_validation_probabilities)
    if not start_gold_eligible:
        raise ValueError("The warm-start checkpoint does not satisfy its Gold recall-preservation contract")
    result_dir = c.ROOT / "results"
    result_dir.mkdir()
    c.save(result_dir / "execution.json", {
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hostname": socket.gethostname(), "job_id": os.environ["SLURM_JOB_ID"],
        "torch": torch.__version__, "numpy": np.__version__, "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0), "input_manifest_sha256": c.digest(c.ROOT / "INPUT_SHA256.json"),
        "initialization": "phase2_selected.pth, itself strict warm-start from Gold; no layer reinitialization",
        "trainable_modules": ["res_layers.4", "head_rhythm"],
        "training": "CEDIA GPU; original architecture and total parameters unchanged",
        "gold_embedding_max_abs_error": {"train": max_train_feature_error,
                                          "validation": max_validation_feature_error},
        "starting_validation_score": start_score,
    })
    trial_results: list[dict[str, object]] = []
    trial_histories: dict[str, object] = {}
    best_trial: dict[str, object] | None = None
    for trial in PROTOCOL["optimization"]["trials"]:
        states, history = run_trial(trial, PROTOCOL["optimization"]["epochs"], train, validation, initial_state,
                                    train_reference, validation_reference,
                                    starting_validation_probabilities, result_dir)
        trial_histories[str(trial)] = history
        for epoch in PROTOCOL["optimization"]["checkpoints"]:
            name = (f"partial_lr{trial['learning_rate']}_anchor{trial['gold_anchor']}_focus"
                    f"{trial['weak_class_multiplier']}_e{epoch:02d}")
            report = json.loads((result_dir / f"{name}.json").read_text(encoding="utf-8"))
            trial_results.append(report)
            if not report["eligible"] or report["score_source_mean_macro_f1"] <= start_score + PROTOCOL["selection"]["minimum_score_gain"]:
                continue
            if best_trial is None or report["score_source_mean_macro_f1"] > best_trial["score_source_mean_macro_f1"]:
                best_trial = report
    c.save(result_dir / "training_histories.json", trial_histories)
    selection = {
        "selected": "phase2_selected" if best_trial is None else best_trial["name"],
        "phase2_selected_validation_score": start_score,
        "phase2_selected_gold_recall_contract": start_gold_eligible,
        "gold_validation_metrics": c.metrics(validation["y"], gold_validation_probabilities),
        "phase2_selected_validation_metrics": c.metrics(validation["y"], starting_validation_probabilities),
        "phase2_selected_sources": start_sources,
        "candidates": trial_results,
        "selection_roles": ["train", "validation"],
        "followup_opened_only_after_selection": True,
        "minimum_validation_gain": PROTOCOL["selection"]["minimum_score_gain"],
        "labels_modified": False,
    }
    if best_trial is None:
        selected_state = initial_state
        refit = {"selected": "phase2_selected", "refit_needed": False,
                 "reason": "No partial-backbone candidate exceeded the predeclared validation selection rule"}
    else:
        selected_trial = {key: float(best_trial[key]) for key in
                          ("learning_rate", "gold_anchor", "weak_class_multiplier")}
        epoch = int(best_trial["epoch"])
        states, _ = run_trial(selected_trial, epoch, train, validation, initial_state, train_reference,
                              validation_reference, starting_validation_probabilities, None)
        selected_state = states[epoch]
        saved_state = torch.load(result_dir / f"{best_trial['name']}.pth", map_location="cpu", weights_only=True)["state_dict"]
        if set(saved_state) != set(selected_state) or not all(torch.equal(saved_state[k], selected_state[k]) for k in saved_state):
            raise ValueError("Deterministic refit differs from the selected validation checkpoint")
        refit_model = c.load_model(result_dir / f"{best_trial['name']}.pth")
        refit = {"selected": best_trial["name"], "refit_needed": True,
                 "all_state_tensors_bitwise_equal": True,
                 "changed_state_tensors": frozen_state_check(
                     refit_model, initial_state,
                     {name for name, _ in refit_model.named_parameters() if name.startswith(ALLOWED_PREFIXES)})}
    selected_path = result_dir / "selected.pth"
    torch.save({"state_dict": selected_state, "classes": c.CLASSES, "parameters": c.PARAMETERS}, selected_path)
    selected_model = c.load_model(selected_path).cuda()
    differences = frozen_state_check(
        selected_model, initial_state,
        {name for name, _ in selected_model.named_parameters() if name.startswith(ALLOWED_PREFIXES)})
    selection["changed_state_tensors"] = differences
    selection["allowed_trainable_parameter_count"] = sum(
        parameter.numel() for name, parameter in selected_model.named_parameters()
        if name.startswith(ALLOWED_PREFIXES))
    selection["changed_parameter_count"] = sum(
        parameter.numel() for name, parameter in selected_model.named_parameters() if name in differences)
    selection["selected_checkpoint_sha256"] = c.digest(selected_path)
    c.save(result_dir / "selection.json", selection)
    c.save(result_dir / "refit.json", refit)
    followup = evaluate_follow_up(selected_model, evaluation, result_dir)
    c.save(result_dir / "followup_metrics.json", followup)
    c.save(result_dir / "SHA256SUMS.json", {path.name: c.digest(path) for path in sorted(result_dir.iterdir())
                                             if path.is_file()})
    print(json.dumps({"COMPLETED": selection["selected"], "parameters": c.PARAMETERS,
                      "allowed_trainable_parameters": selection["allowed_trainable_parameter_count"],
                      "changed_parameters": selection["changed_parameter_count"],
                      "followup_records": followup["records"]}), flush=True)


if __name__ == "__main__":
    main()
