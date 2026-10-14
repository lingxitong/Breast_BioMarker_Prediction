"""Precompute patient-grouped outer K-fold tests and inner validation splits."""

import argparse
from pathlib import Path
import warnings

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

from biomarkers import TASKS
from biomarkers.data import assert_disjoint, patient_table, read_manifest
from biomarkers.utils import file_hash, save_json


def choose_strata(patients, min_count, val_size=None, train_size=None):
    candidates = [("joint_ER_PR_HER2", patients[list(TASKS)].astype(str).agg("_".join, axis=1))]
    candidates += [(task, patients[task].astype(str)) for task in ("HER2", "ER", "PR")]
    for name, labels in candidates:
        counts = labels.value_counts()
        if len(counts) < 2 or counts.min() < min_count:
            continue
        if val_size is not None and min(val_size, train_size) < len(counts):
            continue
        return name, labels
    return "unstratified", None


def describe(frame):
    patients = patient_table(frame)
    return {"n_slides": len(frame), "n_patients": len(patients), "labels": {
        task: {"positive": int((patients[task] == 1).sum()),
               "negative": int((patients[task] == 0).sum()),
               "missing": int((patients[task] < 0).sum())} for task in TASKS}}


def prepare(csv_path, output_dir, n_folds=5, val_fraction=0.2, seed=42,
            group_col="patient_id", feature_root=None, check_files=False, overwrite=False):
    if n_folds < 2 or not 0 < val_fraction < 1:
        raise ValueError("n_folds must be >=2 and val_fraction must be between 0 and 1")
    output_dir = Path(output_dir)
    if (output_dir / "split_metadata.json").exists() and not overwrite:
        raise FileExistsError("Split metadata already exists; choose another directory or --overwrite")
    frame = read_manifest(csv_path, group_col=group_col, feature_root=feature_root,
                          check_files=check_files)
    patients = patient_table(frame)
    if len(patients) < n_folds:
        raise ValueError("Not enough patients for the requested K-fold split")
    name, strata = choose_strata(patients, n_folds)
    if name != "joint_ER_PR_HER2":
        warnings.warn(f"Joint-label strata are too sparse; outer stratification: {name}")
    splitter = (StratifiedKFold(n_folds, shuffle=True, random_state=seed) if strata is not None
                else KFold(n_folds, shuffle=True, random_state=seed))
    planned, test_patient_ids = [], []
    for fold, (dev_indices, test_indices) in enumerate(splitter.split(patients, strata)):
        dev = patients.iloc[dev_indices].reset_index(drop=True)
        n_val = int(np.ceil(len(dev) * val_fraction))
        if n_val == 0 or len(dev) - n_val == 0:
            raise ValueError("Inner train/validation sets would be empty; add more patients")
        inner_name, inner_strata = choose_strata(dev, 2, n_val, len(dev) - n_val)
        if inner_name != "joint_ER_PR_HER2":
            warnings.warn(f"Fold {fold}: inner stratification: {inner_name}")
        train_ids, val_ids = train_test_split(dev.patient_id.to_numpy(), test_size=n_val,
                                             random_state=seed + fold + 1, stratify=inner_strata)
        test_ids = patients.iloc[test_indices].patient_id.to_numpy()
        parts = {"train": frame.loc[frame.patient_id.isin(train_ids)].copy(),
                 "val": frame.loc[frame.patient_id.isin(val_ids)].copy(),
                 "test": frame.loc[frame.patient_id.isin(test_ids)].copy()}
        assert_disjoint(parts)
        planned.append((fold, inner_name, parts))
        test_patient_ids.extend(test_ids.tolist())
    if len(test_patient_ids) != len(patients) or len(set(test_patient_ids)) != len(patients):
        raise RuntimeError("Every patient must occur in exactly one outer test fold")
    metadata = {"schema_version": 1, "n_folds": n_folds, "seed": seed,
                "val_fraction_of_outer_development": val_fraction,
                "outer_stratification": name, "original_group_column": group_col,
                "effective_grouping_column": frame.attrs.get("grouping_column", group_col),
                "evaluation_unit": frame.attrs.get("evaluation_unit", "patient"),
                "source_csv": str(Path(csv_path).resolve()), "source_sha256": file_hash(csv_path),
                "source": describe(frame), "folds": []}
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "all.csv", index=False)
    for fold, inner_name, parts in planned:
        folder = output_dir / f"fold_{fold}"
        folder.mkdir(parents=True, exist_ok=True)
        entry = {"fold": fold, "inner_stratification": inner_name, "sets": {}}
        for role, part in parts.items():
            path = folder / f"{role}.csv"
            part.to_csv(path, index=False)
            entry["sets"][role] = {**describe(part), "sha256": file_hash(path)}
        metadata["folds"].append(entry)
        print(f"Fold {fold}: " + ", ".join(f"{r}={p.patient_id.nunique()} patients/{len(p)} slides"
                                            for r, p in parts.items()))
    metadata["all_csv_sha256"] = file_hash(output_dir / "all.csv")
    save_json(output_dir / "split_metadata.json", metadata)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--val-fraction", type=float, default=0.2,
                        help="Validation fraction WITHIN the remaining outer development patients")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--group-col", default="patient_id")
    parser.add_argument("--feature-root")
    parser.add_argument("--check-files", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    prepare(args.csv, args.output_dir, args.n_folds, args.val_fraction, args.seed,
            args.group_col, args.feature_root, args.check_files, args.overwrite)


if __name__ == "__main__":
    main()
