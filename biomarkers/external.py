"""External evaluation with one best-validation checkpoint from each fold."""

import json
from pathlib import Path
import re

import pandas as pd
import torch

from biomarkers import TASKS
from biomarkers.data import assert_disjoint, read_manifest
from biomarkers.engine import make_loader, predict, save_evaluation
from biomarkers.metrics import aggregate_patients, compute_metrics
from biomarkers.model import QueryDecoupledMIL
from biomarkers.utils import file_hash, resolve_device, save_json


def read_external_cohorts(specifications, development, group_col="patient_id", feature_root=None):
    cohorts = {}
    for specification in specifications:
        if "=" not in specification:
            raise ValueError("External cohort must be CENTER=path/to/manifest.csv")
        name, csv_path = specification.split("=", 1)
        if not re.fullmatch(r"[A-Za-z0-9_-]+", name) or name in cohorts:
            raise ValueError("External center names must be unique and use letters, digits, _ or -")
        cohorts[name] = read_manifest(csv_path, group_col, feature_root, require_patient=False,
                                      check_files=True)
    if not cohorts:
        raise ValueError("Provide at least one external CENTER=CSV")
    assert_disjoint({"development": development, **{f"external:{n}": f for n, f in cohorts.items()}})
    return cohorts


def evaluate_external(run_dir, specifications, group_col="patient_id", feature_root=None,
                      device="auto", batch_size=1, num_workers=0, expected_folds=5):
    if expected_folds < 2 or batch_size < 1 or num_workers < 0:
        raise ValueError("Invalid evaluation parameters")
    run_dir = Path(run_dir)
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    if config["n_folds"] != expected_folds or sorted(config["trained_folds"]) != list(range(expected_folds)):
        raise ValueError(f"Expected all {expected_folds} trained folds in the same run")
    development_path = run_dir / "development_manifest.csv"
    if file_hash(development_path) != config["development_sha256"]:
        raise ValueError("Development manifest changed; cannot reliably audit external leakage")
    development = read_manifest(development_path)
    cohorts = read_external_cohorts(specifications, development, group_col, feature_root)
    checkpoints = []
    for fold in range(expected_folds):
        path = run_dir / f"fold_{fold}" / "best.pt"
        if not path.is_file():
            raise FileNotFoundError(f"Missing fold {fold} best checkpoint: {path}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if (checkpoint["fold"] != fold or checkpoint["run_id"] != config["run_id"]
                or checkpoint["development_sha256"] != config["development_sha256"]
                or checkpoint["tasks"] != list(TASKS)
                or checkpoint["model_config"] != config["model_config"]):
            raise ValueError("Checkpoints have inconsistent folds, runs, training cohorts, tasks or architectures")
        if checkpoints and (checkpoint["feature_config"] != checkpoints[0]["feature_config"]
                            or checkpoint["threshold"] != checkpoints[0]["threshold"]):
            raise ValueError("Checkpoints have inconsistent feature settings or thresholds")
        checkpoints.append(checkpoint)
    device = resolve_device(device)
    threshold = checkpoints[0]["threshold"]
    output = run_dir / "external"
    pooled_patients, cohort_summaries = [], {}
    for name, frame in cohorts.items():
        tables, fold_metrics = [], []
        feature_config = checkpoints[0]["feature_config"]
        loader = make_loader(frame, config["model_config"], feature_config["h5_key"],
                              feature_config["eval_max_patches"], batch_size, num_workers)
        for checkpoint in checkpoints:
            fold = checkpoint["fold"]
            model = QueryDecoupledMIL(**checkpoint["model_config"]).to(device)
            model.load_state_dict(checkpoint["model_state"])
            slides, patients, metrics = predict(model, loader, device, threshold,
                                                frame.attrs["evaluation_unit"])
            save_evaluation(output / name / f"fold_{fold}", slides, patients, metrics)
            tables.append(slides)
            fold_metrics.append({"fold": fold, "selected_epoch": checkpoint["epoch"], "metrics": metrics})
            print(f"External {name}, fold {fold}: macro AUROC={metrics['macro_auroc']}", flush=True)
            del model
        ensemble_slides = tables[0].copy()
        for table in tables[1:]:
            if not table[["patient_id", "feature_path"]].equals(tables[0][["patient_id", "feature_path"]]):
                raise RuntimeError("External predictions are not aligned across models")
        for task in TASKS:
            ensemble_slides[f"{task}_prob"] = sum(t[f"{task}_prob"] for t in tables) / expected_folds
            ensemble_slides[f"{task}_pred"] = (ensemble_slides[f"{task}_prob"] >= threshold).astype(int)
        ensemble_patients = aggregate_patients(ensemble_slides)
        for task in TASKS:
            ensemble_patients[f"{task}_pred"] = (ensemble_patients[f"{task}_prob"] >= threshold).astype(int)
        ensemble_metrics = compute_metrics(ensemble_patients, threshold)
        ensemble_metrics["evaluation_unit"] = frame.attrs["evaluation_unit"]
        save_evaluation(output / name / "ensemble", ensemble_slides, ensemble_patients, ensemble_metrics)
        cohort_summaries[name] = {"fold_results": fold_metrics, "ensemble": ensemble_metrics}
        ensemble_patients["center"] = name
        pooled_patients.append(ensemble_patients)
    pooled = pd.concat(pooled_patients, ignore_index=True)
    pooled_folder = output / "pooled"
    pooled_folder.mkdir(parents=True, exist_ok=True)
    pooled.to_csv(pooled_folder / "patient_predictions.csv", index=False)
    pooled_metrics = compute_metrics(pooled, threshold)
    units = {frame.attrs["evaluation_unit"] for frame in cohorts.values()}
    pooled_metrics["evaluation_unit"] = next(iter(units)) if len(units) == 1 else "mixed_patient_and_slide"
    pooled_metrics["note"] = "Pooled metrics can differ from center-wise performance; report both."
    save_json(pooled_folder / "metrics.json", pooled_metrics)
    save_json(output / "summary.json", {"cohorts": cohort_summaries, "pooled": pooled_metrics,
                                       "n_models": expected_folds, "ensemble_method": "mean_probability",
                                       "checkpoints": [{"fold": c["fold"], "epoch": c["epoch"]}
                                                       for c in checkpoints]})
    save_json(output / "evaluation_config.json", {"external_csvs": specifications,
              "group_col": group_col, "feature_root": feature_root, "threshold": threshold,
              "run_id": config["run_id"], "expected_folds": expected_folds,
              "source_csv_sha256": {s.split("=", 1)[0]: file_hash(s.split("=", 1)[1]) for s in specifications}})
    return cohort_summaries
