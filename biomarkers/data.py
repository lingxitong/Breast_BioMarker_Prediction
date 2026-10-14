"""Manifest validation, patient grouping and variable-length HDF5 bags."""

from pathlib import Path
import warnings

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from biomarkers import TASKS


def read_manifest(csv_path, group_col="patient_id", feature_root=None,
                  require_patient=False, check_files=False):
    csv_path = Path(csv_path).resolve()
    frame = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    columns = [str(c).strip() for c in frame.columns]
    if len(columns) < 4 or [c.upper() for c in columns[1:4]] != list(TASKS):
        raise ValueError(f"{csv_path}: CSV must have a header: first column H5 path, "
                         "followed by ER,PR,HER2, then patient_id (and optional metadata).")
    if len(set(columns)) != len(columns):
        raise ValueError(f"{csv_path}: duplicate column names")
    if columns[0] in TASKS or columns[0] == group_col:
        raise ValueError("The first column must contain feature paths, not labels/IDs")
    frame.columns = columns
    frame = frame.rename(columns={columns[0]: "feature_path", **dict(zip(columns[1:4], TASKS))})
    if frame.empty:
        raise ValueError(f"{csv_path}: empty manifest")
    root = Path(feature_root).resolve() if feature_root else csv_path.parent
    paths = []
    for value in frame.feature_path:
        value = value.strip()
        if not value:
            raise ValueError(f"{csv_path}: blank feature path")
        path = Path(value).expanduser()
        paths.append(str((path if path.is_absolute() else root / path).resolve()))
    frame["feature_path"] = paths
    if frame.feature_path.duplicated().any():
        raise ValueError(f"{csv_path}: duplicate H5 paths; deduplicate before splitting")
    if check_files:
        missing = [p for p in paths if not Path(p).is_file()]
        if missing:
            raise FileNotFoundError(f"{len(missing)} H5 files are missing; first: {missing[0]}")
    if group_col not in frame:
        if require_patient:
            raise ValueError(f"{csv_path}: missing group column {group_col!r}")
        warnings.warn(
            f"{csv_path}: no {group_col}; treating each H5/slide as one independent unit. "
            "If multiple slides belong to one patient, add a patient_id column to prevent leakage."
        )
        frame["patient_id"] = frame.feature_path
        frame.attrs["evaluation_unit"] = "slide"
        frame.attrs["grouping_column"] = "feature_path"
    else:
        ids = frame[group_col].str.strip()
        if ids.eq("").any():
            raise ValueError(f"{csv_path}: blank patient/group IDs")
        frame["patient_id"] = ids
        frame.attrs["evaluation_unit"] = "patient"
        frame.attrs["grouping_column"] = group_col
    for task in TASKS:
        text = frame[task].str.strip()
        missing = text.str.lower().isin(["", "na", "nan", "null", "none", "-1"])
        values = pd.to_numeric(text.mask(missing), errors="coerce")
        if ((~missing & values.isna()) | (values.notna() & ~values.isin([0, 1]))).any():
            raise ValueError(f"{csv_path}: {task} must be 0, 1, or missing (blank/-1)")
        frame[task] = values.fillna(-1).astype(int)
    # Multiple slides from one patient may have missing labels, but known labels must agree.
    for task in TASKS:
        known = frame.loc[frame[task] >= 0]
        if (known.groupby("patient_id")[task].nunique() > 1).any():
            raise ValueError(f"{csv_path}: conflicting {task} labels within patient_id; "
                             "resolve labels or use a globally unique tumor/specimen group ID")
        mapping = known.groupby("patient_id")[task].first()
        frame[task] = frame.patient_id.map(mapping).fillna(-1).astype(int)
    return frame


def patient_table(frame):
    return frame.groupby("patient_id", sort=True)[list(TASKS)].first().reset_index()


def assert_disjoint(named_frames):
    items = list(named_frames.items())
    for i, (left_name, left) in enumerate(items):
        for right_name, right in items[i + 1:]:
            patient_overlap = set(left.patient_id) & set(right.patient_id)
            path_overlap = set(left.feature_path) & set(right.feature_path)
            if patient_overlap or path_overlap:
                raise ValueError(f"Leakage between {left_name} and {right_name}: "
                                 f"{len(patient_overlap)} patient IDs, {len(path_overlap)} H5 paths. "
                                 "Patient IDs must be globally unique across centers.")


def feature_shape(path, key="features"):
    with h5py.File(path, "r") as handle:
        if key not in handle or not isinstance(handle[key], h5py.Dataset):
            raise ValueError(f"{path}: missing HDF5 dataset {key!r}; available keys: {list(handle.keys())}")
        shape = handle[key].shape
        if len(shape) != 2 or min(shape) <= 0:
            raise ValueError(f"{path}: expected non-empty [patches, feature_dim], got {shape}")
        return shape


class FeatureDataset(Dataset):
    def __init__(self, frame, input_dim, h5_key="features", max_patches=0, training=False):
        self.frame = frame.reset_index(drop=True)
        self.input_dim = input_dim
        self.h5_key = h5_key
        self.max_patches = max_patches
        self.training = training

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        # Each worker opens its own handle; never share live HDF5 handles across workers.
        with h5py.File(row.feature_path, "r") as handle:
            if self.h5_key not in handle:
                raise ValueError(f"{row.feature_path}: missing dataset {self.h5_key!r}")
            dataset = handle[self.h5_key]
            if not isinstance(dataset, h5py.Dataset) or dataset.ndim != 2:
                raise ValueError(f"{row.feature_path}: features must be a 2D HDF5 dataset")
            n, dim = dataset.shape
            if n == 0 or dim != self.input_dim:
                raise ValueError(f"{row.feature_path}: expected [N>0,{self.input_dim}], got {dataset.shape}")
            if 0 < self.max_patches < n:
                if self.training:
                    indices = torch.randperm(n)[:self.max_patches].sort().values.numpy()
                else:
                    indices = np.linspace(0, n - 1, self.max_patches, dtype=np.int64)
                features = np.asarray(dataset[indices], dtype=np.float32)
            else:
                features = np.asarray(dataset[:], dtype=np.float32)
        if not np.isfinite(features).all():
            raise ValueError(f"{row.feature_path}: non-finite features")
        return {
            "features": torch.from_numpy(features),
            "labels": torch.tensor([row[t] for t in TASKS], dtype=torch.float32),
            "patient_id": row.patient_id,
            "feature_path": row.feature_path,
        }


def collate_bags(samples):
    lengths = [s["features"].shape[0] for s in samples]
    features = torch.zeros(len(samples), max(lengths), samples[0]["features"].shape[1])
    padding_mask = torch.ones(len(samples), max(lengths), dtype=torch.bool)
    for i, sample in enumerate(samples):
        features[i, :lengths[i]] = sample["features"]
        padding_mask[i, :lengths[i]] = False
    return {
        "features": features, "padding_mask": padding_mask,
        "labels": torch.stack([s["labels"] for s in samples]),
        "patient_id": [s["patient_id"] for s in samples],
        "feature_path": [s["feature_path"] for s in samples],
    }
