"""Train precomputed folds; select one best validation checkpoint per fold."""

import argparse
import json
from pathlib import Path
import uuid
import warnings

import numpy as np
import pandas as pd
import torch

from biomarkers import TASKS
from biomarkers.data import assert_disjoint, feature_shape, patient_table, read_manifest
from biomarkers.engine import make_loader, predict, save_evaluation
from biomarkers.model import QueryDecoupledMIL, masked_bce
from biomarkers.utils import file_hash, resolve_device, save_json, seed_everything


def train_fold(args, fold, parts, model_config, run_config, device):
    fold_dir = Path(args.run_dir) / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed + fold)
    train_patients = patient_table(parts["train"])
    pos_weights = []
    for task in TASKS:
        positive = int((train_patients[task] == 1).sum())
        negative = int((train_patients[task] == 0).sum())
        if min(positive, negative) == 0:
            raise ValueError(f"Fold {fold}: training {task} lacks positive or negative patients. "
                             "Change split settings or add patients; cannot train this binary task.")
        pos_weights.append(negative / positive)
    weight = (torch.tensor(pos_weights, dtype=torch.float32, device=device)
              if args.class_weight == "balanced" else None)
    if (parts["train"][list(TASKS)] < 0).all(axis=1).any():
        raise ValueError("Training contains patients without any observed biomarker label")
    train_loader = make_loader(parts["train"], model_config, args.h5_key, args.max_patches,
                               args.batch_size, args.num_workers, True, args.seed + fold)
    val_loader = make_loader(parts["val"], model_config, args.h5_key, args.eval_max_patches,
                             args.batch_size, args.num_workers)
    val_patients = patient_table(parts["val"])
    val_auc_tasks = [t for t in TASKS if val_patients.loc[val_patients[t] >= 0, t].nunique() == 2]
    if not (val_patients[list(TASKS)] >= 0).any().any():
        raise ValueError(f"Fold {fold}: validation has no observed labels")
    selection = args.selection_metric
    if selection == "macro_auroc" and not val_auc_tasks:
        warnings.warn(f"Fold {fold}: no validation task supports AUROC; selecting by log_loss")
        selection = "log_loss"
    elif selection == "macro_auroc" and len(val_auc_tasks) < len(TASKS):
        warnings.warn(f"Fold {fold}: validation macro AUROC uses only {val_auc_tasks}")
    model = QueryDecoupledMIL(**model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    amp_enabled = args.amp and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    best_score, stale, history = -float("inf"), 0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, total_supervised, total_penalty, seen = 0.0, 0.0, 0.0, 0
        for batch in train_loader:
            features = batch["features"].to(device, non_blocking=True)
            mask = batch["padding_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                logits = model(features, mask)
                supervised = masked_bce(logits, labels, weight)
            penalty = model.query_penalty()
            loss = supervised + args.query_lambda * penalty
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Fold {fold}, epoch {epoch}: non-finite training loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            count = len(labels)
            seen += count
            total_loss += float(loss.detach()) * count
            total_supervised += float(supervised.detach()) * count
            total_penalty += float(penalty.detach()) * count
        evaluation_unit = parts["train"].attrs.get("evaluation_unit", "patient")
        slides, patients, metrics = predict(model, val_loader, device, args.threshold, evaluation_unit)
        score = (metrics["macro_auroc"] if selection == "macro_auroc"
                 else -metrics["macro_log_loss"])
        entry = {"epoch": epoch, "train_loss": total_loss / seen,
                 "train_bce": total_supervised / seen, "query_penalty": total_penalty / seen,
                 "val_macro_auroc": metrics["macro_auroc"],
                 "val_macro_log_loss": metrics["macro_log_loss"],
                 "selection_metric": selection, "selection_score": score}
        history.append(entry)
        save_json(fold_dir / "history.json", history)
        print(f"Fold {fold} epoch {epoch:03d}: loss={entry['train_loss']:.4f}, "
              f"val_auc={metrics['macro_auroc']}, val_log_loss={metrics['macro_log_loss']:.4f}", flush=True)
        if score > best_score:
            best_score, stale = score, 0
            checkpoint = {
                "schema_version": 1, "model_state": model.state_dict(), "model_config": model_config,
                "feature_config": {"h5_key": args.h5_key, "eval_max_patches": args.eval_max_patches},
                "tasks": list(TASKS), "fold": fold, "epoch": epoch,
                "selection_metric": selection, "selection_score": score,
                "val_auc_tasks": val_auc_tasks, "threshold": args.threshold,
                "run_id": run_config["run_id"],
                "development_sha256": run_config["development_sha256"],
            }
            temporary = fold_dir / "best.pt.tmp"
            torch.save(checkpoint, temporary)
            temporary.replace(fold_dir / "best.pt")
            save_evaluation(fold_dir / "val", slides, patients, metrics)
        else:
            stale += 1
        if args.patience > 0 and stale >= args.patience:
            print(f"Fold {fold}: early stopping at epoch {epoch}", flush=True)
            break
    checkpoint = torch.load(fold_dir / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    test_loader = make_loader(parts["test"], model_config, args.h5_key, args.eval_max_patches,
                              args.batch_size, args.num_workers)
    slides, patients, test_metrics = predict(model, test_loader, device, args.threshold, evaluation_unit)
    save_evaluation(fold_dir / "test", slides, patients, test_metrics)
    summary = {"fold": fold, "best_epoch": checkpoint["epoch"], "selection_metric": selection,
               "best_selection_score": best_score,
               "val": json.loads((fold_dir / "val" / "metrics.json").read_text(encoding="utf-8")),
               "test": test_metrics}
    save_json(fold_dir / "summary.json", summary)
    return summary, patients


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits-dir", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--folds", default="all", help="all, or comma-separated fold IDs (0,1,2...)")
    parser.add_argument("--external", action="append", default=[], metavar="CENTER=CSV")
    parser.add_argument("--external-group-col", default="patient_id")
    parser.add_argument("--external-feature-root")
    parser.add_argument("--h5-key", default="features")
    parser.add_argument("--input-dim", type=int, help="Inferred from the first training H5 if omitted")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--query-lambda", type=float, default=0.01)
    parser.add_argument("--max-patches", type=int, default=4096, help="Training patch cap; 0 means all")
    parser.add_argument("--eval-max-patches", type=int, default=0, help="Deterministic eval cap; 0 means all")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20, help="0 disables early stopping")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--class-weight", choices=["none", "balanced"], default="none")
    parser.add_argument("--selection-metric", choices=["macro_auroc", "log_loss"], default="macro_auroc")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", action="store_true")
    args = parser.parse_args()
    if (args.epochs < 1 or args.batch_size < 1 or args.num_workers < 0 or args.patience < 0
            or min(args.max_patches, args.eval_max_patches, args.query_lambda, args.weight_decay) < 0
            or args.lr <= 0 or args.grad_clip <= 0 or not 0 < args.threshold < 1):
        parser.error("Invalid training parameters")
    split_dir, run_dir = Path(args.splits_dir), Path(args.run_dir)
    metadata_path = split_dir / "split_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    all_folds = list(range(metadata["n_folds"]))
    folds = all_folds if args.folds == "all" else [int(x) for x in args.folds.split(",")]
    if not folds or len(folds) != len(set(folds)) or not set(folds).issubset(all_folds):
        parser.error("Requested folds are duplicated or outside the prepared split")
    if args.external and folds != all_folds:
        parser.error("Automatic external evaluation requires training all folds; otherwise use evaluate.py later")
    if args.external and len(all_folds) != 5:
        parser.error("Automatic external evaluation requires 5 prepared folds")
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError("Run directory is not empty; use a fresh --run-dir to avoid mixing checkpoints")
    if file_hash(split_dir / "all.csv") != metadata["all_csv_sha256"]:
        raise ValueError("all.csv changed after split preparation; regenerate splits")
    development = read_manifest(split_dir / "all.csv")
    evaluation_unit = metadata.get("evaluation_unit", "patient")
    development.attrs["evaluation_unit"] = evaluation_unit
    all_parts = {}
    for fold in folds:
        entry = metadata["folds"][fold]
        parts = {}
        for role in ("train", "val", "test"):
            path = split_dir / f"fold_{fold}" / f"{role}.csv"
            if file_hash(path) != entry["sets"][role]["sha256"]:
                raise ValueError(f"Fold {fold} {role} CSV changed after preparation")
            parts[role] = read_manifest(path)
            parts[role].attrs["evaluation_unit"] = evaluation_unit
        assert_disjoint(parts)
        union = pd.concat(list(parts.values()), ignore_index=True)
        if set(union.feature_path) != set(development.feature_path):
            raise ValueError(f"Fold {fold} does not partition the full development cohort")
        all_parts[fold] = parts
    # Validate external schema and patient/path overlap BEFORE spending time training.
    if args.external:
        from biomarkers.external import read_external_cohorts
        read_external_cohorts(args.external, development, args.external_group_col,
                              args.external_feature_root)
    first = all_parts[folds[0]]["train"].feature_path.iloc[0]
    inferred_dim = feature_shape(first, args.h5_key)[1]
    if args.input_dim is not None and args.input_dim != inferred_dim:
        raise ValueError(f"--input-dim {args.input_dim} does not match H5 dimension {inferred_dim}")
    model_config = {"input_dim": inferred_dim, "hidden_dim": args.hidden_dim,
                    "num_heads": args.num_heads, "dropout": args.dropout}
    # Validate architecture before creating output files.
    QueryDecoupledMIL(**model_config)
    device = resolve_device(args.device)
    run_dir.mkdir(parents=True, exist_ok=True)
    development.to_csv(run_dir / "development_manifest.csv", index=False)
    run_config = {"schema_version": 1, "run_id": str(uuid.uuid4()), "args": vars(args),
                  "model_config": model_config, "n_folds": metadata["n_folds"],
                  "trained_folds": folds, "tasks": list(TASKS),
                  "evaluation_unit": evaluation_unit,
                  "development_sha256": file_hash(run_dir / "development_manifest.csv"),
                  "split_metadata_sha256": file_hash(metadata_path),
                  "versions": {"torch": torch.__version__, "numpy": np.__version__, "pandas": pd.__version__}}
    save_json(run_dir / "config.json", run_config)
    summaries, test_predictions = [], []
    for fold in folds:
        summary, patients = train_fold(args, fold, all_parts[fold], model_config, run_config, device)
        summaries.append(summary)
        patients["fold"] = fold
        test_predictions.append(patients)
    save_json(run_dir / "fold_summaries.json", summaries)
    if folds == all_folds:
        from biomarkers.metrics import compute_metrics
        out_of_fold = pd.concat(test_predictions, ignore_index=True)
        if out_of_fold.patient_id.duplicated().any():
            raise RuntimeError("Duplicate patients in out-of-fold test predictions")
        out_of_fold.to_csv(run_dir / "oof_test_predictions.csv", index=False)
        metrics = compute_metrics(out_of_fold, args.threshold)
        metrics["evaluation_unit"] = evaluation_unit
        save_json(run_dir / "oof_test_metrics.json", metrics)
    if args.external:
        from biomarkers.external import evaluate_external
        evaluate_external(args.run_dir, args.external, args.external_group_col,
                           args.external_feature_root, args.device, args.batch_size,
                           args.num_workers, expected_folds=5)


if __name__ == "__main__":
    main()
