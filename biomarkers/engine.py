import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from biomarkers import TASKS
from biomarkers.data import FeatureDataset, collate_bags
from biomarkers.metrics import aggregate_patients, compute_metrics
from biomarkers.utils import save_json, seed_worker


def make_loader(frame, model_config, h5_key, max_patches, batch_size, num_workers,
                training=False, seed=42):
    dataset = FeatureDataset(frame, model_config["input_dim"], h5_key, max_patches, training)
    generator = torch.Generator().manual_seed(seed)
    sampler = None
    if training:
        # Give each patient equal expected total weight despite different slide counts.
        counts = frame.patient_id.value_counts()
        weights = frame.patient_id.map(lambda patient: 1 / counts[patient]).to_numpy()
        sampler = WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double),
                                         len(frame), replacement=True, generator=generator)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, shuffle=False,
                      num_workers=num_workers, collate_fn=collate_bags,
                      worker_init_fn=seed_worker, generator=generator,
                      pin_memory=torch.cuda.is_available(), persistent_workers=num_workers > 0)


@torch.inference_mode()
def predict(model, loader, device, threshold=0.5, evaluation_unit="patient"):
    model.eval()
    rows = []
    for batch in loader:
        logits = model(batch["features"].to(device), batch["padding_mask"].to(device))
        probabilities = logits.sigmoid().cpu().numpy()
        if not np.isfinite(probabilities).all():
            raise FloatingPointError("Non-finite prediction probabilities")
        labels = batch["labels"].numpy()
        for i, patient_id in enumerate(batch["patient_id"]):
            row = {"patient_id": patient_id, "feature_path": batch["feature_path"][i]}
            for j, task in enumerate(TASKS):
                row[task] = int(labels[i, j])
                row[f"{task}_prob"] = float(probabilities[i, j])
            rows.append(row)
    slide_predictions = pd.DataFrame(rows)
    patient_predictions = aggregate_patients(slide_predictions)
    for table in (slide_predictions, patient_predictions):
        for task in TASKS:
            table[f"{task}_pred"] = (table[f"{task}_prob"] >= threshold).astype(int)
    metrics = compute_metrics(patient_predictions, threshold)
    metrics["evaluation_unit"] = evaluation_unit
    return slide_predictions, patient_predictions, metrics


def save_evaluation(directory, slides, patients, metrics):
    directory.mkdir(parents=True, exist_ok=True)
    slides.to_csv(directory / "slide_predictions.csv", index=False)
    patients.to_csv(directory / "patient_predictions.csv", index=False)
    save_json(directory / "metrics.json", metrics)
