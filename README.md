# Breast biomarker prediction from H&E features

Patient-level, three-task prediction of ER, PR, and HER2 from precomputed H&E whole-slide features. The first version uses a query-decoupled multiple-instance learning (MIL) model: each biomarker has its own learned query and prediction head, while all three queries read a shared bag of patch features. A query orthogonality penalty reduces redundancy between the task queries.

The workflow has three separate stages:

1. Prepare and save patient-grouped K-fold train/validation/test manifests once.
2. Train one model per fold, select its best checkpoint using validation data, and evaluate that checkpoint on its held-out internal test fold.
3. Evaluate the five fold-best checkpoints on each provided external-center CSV. The external ensemble is the mean of the five models' probabilities.

## 1. Install

Python 3.10 or newer is required.

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -e .
```

Install a PyTorch build suitable for the machine (CPU or the installed CUDA version) before or after installing the project dependencies. See the [official PyTorch installation selector](https://pytorch.org/get-started/locally/).

## 2. Input data

Provide one CSV manifest for development and one CSV per external center. The first four columns are:

```csv
h5_path,ER,PR,HER2,patient_id
features/case001.h5,1,1,0,P001
features/case002.h5,0,0,1,P002
```

The first column may have any name and contains the HDF5 feature path. The next columns must be named `ER`, `PR`, and `HER2` in that order; `patient_id` is optional when each row is an independent patient. Labels are `0` or `1`. Unknown labels may be left blank or set to `-1`; their task loss and metrics are masked. Extra columns are allowed.

Each HDF5 file must contain a two-dimensional dataset named `features`, with shape `[number_of_patches, feature_dimension]`. Change the dataset name with `--h5-key` if needed. Feature paths are resolved relative to the CSV file's directory unless `--feature-root` is supplied.

**Use globally unique patient IDs across all centers whenever a patient has multiple slides.** The split is grouped by this ID, and all known biomarker labels for a patient must agree. If `patient_id` is absent, the pipeline treats each H5/slide as one independent unit and warns; this is only safe when rows do not share a patient. For external CSVs without a patient ID, performance is reported at slide level.

If you have only four columns, the CSV can be:

```csv
h5_path,ER,PR,HER2
features/case001.h5,1,1,0
features/case002.h5,0,0,1
```

## 3. Prepare folds once

The default is five outer folds. In each fold, one fifth of patients is the untouched internal test set. From the remaining four fifths, 20% is reserved for validation; the rest is used for training. Joint ER/PR/HER2 stratification is used when label combinations have enough patients, with a fallback to single-task or unstratified splitting when necessary.

```bash
python prepare_splits.py \
  --csv data/development.csv \
  --output-dir splits/development_5fold \
  --n-folds 5 \
  --val-fraction 0.2 \
  --seed 42 \
  --check-files
```

On PowerShell, use backticks instead of backslashes to continue a command, or enter it on one line. `--check-files` checks that all referenced H5 files exist. The output includes `fold_0` through `fold_4`, each with `train.csv`, `val.csv`, and `test.csv`, plus `all.csv` and `split_metadata.json`. File hashes are saved and checked by training to detect changed split files. Prepare once and reuse the same split directory for all experiments; use a new output directory for a different seed or split.

If a patient has multiple H&E slides, every slide stays in the same partition. Without patient IDs, only slide-level separation can be guaranteed.

## 4. Train, evaluate internal folds, and run external centers

```bash
python train.py \
  --splits-dir splits/development_5fold \
  --run-dir runs/query_mil_5fold \
  --external center_a=data/external/center_a.csv \
  --external center_b=data/external/center_b.csv \
  --epochs 100 \
  --patience 20 \
  --device auto
```

Each `--external` value is `CENTER_NAME=CSV_PATH`; pass the option once per center. External CSVs must contain the same three biomarker label columns and feature paths. They may omit `patient_id`, in which case slide-level metrics are reported. Their H5 paths are resolved relative to each external CSV unless `--external-feature-root` is set. The pipeline checks for overlapping patient IDs and H5 paths between development and external cohorts before training.

For every fold, the checkpoint with the best validation macro-AUROC is saved as `fold_N/best.pt`. AUROC is computed only for tasks with both classes in validation. If no task has both classes, validation macro log-loss is used for checkpoint selection. Internal test data are evaluated only after checkpoint selection. External evaluation then loads the five fold-best checkpoints, reports each model separately, and averages their predicted probabilities for an ensemble. The external CSV must not be used to select epochs, thresholds, or models.

Useful options:

```text
--h5-key features           HDF5 dataset key
--hidden-dim 256            Projected feature/query dimension
--num-heads 4               Cross-attention heads
--query-lambda 0.01          Query decorrelation penalty weight
--max-patches 4096           Random patch cap in training; 0 uses all patches
--eval-max-patches 0         Deterministic patch cap in evaluation; 0 uses all
--class-weight none          Or balanced, based on patient labels in each train fold
--selection-metric macro_auroc  Or log_loss
--device auto                auto, cpu, cuda, or another torch device
--amp                       Enable mixed precision on CUDA
```

To rerun external evaluation later with the same five saved checkpoints:

```bash
python evaluate.py \
  --run-dir runs/query_mil_5fold \
  --external center_a=data/external/center_a.csv \
  --external center_b=data/external/center_b.csv \
  --device auto
```

## 5. Outputs

```text
runs/query_mil_5fold/
├── config.json
├── development_manifest.csv
├── fold_0/
│   ├── best.pt
│   ├── history.json
│   ├── val/                 # slide/patient predictions and metrics
│   ├── test/                # held-out internal test predictions/metrics
│   └── summary.json
├── ... fold_1 through fold_4
├── fold_summaries.json
├── oof_test_predictions.csv # each development patient in exactly one test fold
├── oof_test_metrics.json
└── external/
    ├── center_a/fold_0 ... fold_4/
    ├── center_a/ensemble/
    ├── center_b/...
    ├── pooled/
    └── summary.json
```

Metrics include AUROC, AUPRC, accuracy, sensitivity, specificity, PPV, NPV, F1, Brier score, and log-loss. Metrics are calculated at patient level when IDs are available, by averaging slide probabilities within patient; otherwise they are calculated at slide level. Center-specific results and pooled results are both saved. Report per-center results because pooled performance can hide center-to-center variation. AUROC and AUPRC are undefined for a center with only one observed class and are written as `null` with an explanation.

## Model summary

The HDF5 patch features are projected to a shared embedding. Three independent learned task queries perform cross-attention over the patch bag; each query has its own classifier head. There is no query-to-query attention. The training objective is the mean masked binary cross-entropy across available tasks plus a weighted squared cosine-similarity penalty between task queries. This is a research baseline for patient-level biomarker prediction, not a replacement for IHC/ISH testing.

## Reproducibility and leakage safeguards

- Patient IDs, not slides, are split when `patient_id` is provided.
- Every patient appears in exactly one outer test fold across the five folds.
- Training checks the saved split hashes and rejects changed fold manifests.
- External cohorts are checked against the development manifest for overlapping IDs and H5 paths.
- Checkpoints record the run ID, development-manifest hash, fold, best epoch, model configuration, and validation selection score.
- Threshold defaults to 0.5 and is recorded. Do not tune a reported external result on that external cohort.

## Project files

- `prepare_splits.py`: generate and freeze patient-grouped folds.
- `train.py`: train all or selected folds, evaluate internal test folds, optionally run external cohorts.
- `evaluate.py`: rerun external evaluation from a completed five-fold run.
- `biomarkers/model.py`: query-decoupled MIL, query penalty, and masked multi-task loss.
- `biomarkers/data.py`: CSV validation and HDF5 bag loading.
- `biomarkers/metrics.py`: patient aggregation and classification metrics.

## Planned cohort use

The scripts accept any cohort converted to the CSV/HDF5 format above. Candidate roles discussed for the broader project are:

| Cohort | ER/PR/HER2 availability | Potential role |
| --- | --- | --- |
| Hebei Fourth Hospital development cohort | All three | Development and internal cross-validation |
| Hebei Fourth Hospital held-out cohort | All three | Additional internal validation, if kept outside all model selection |
| Participating multicenter cohorts | All three | Center-specific external validation |
| BCNB, TCGA-BRCA, CPTAC-BRCA | Confirm labels and feature compatibility | Public external validation candidates |
| HEROHE, HER2 Warwick | HER2-focused | HER2 external benchmark candidates |
| Post-NAT-BRCA | Confirm treatment timing and label definition | Separate post-treatment validation |

These are candidate cohort roles, not preconfigured downloads or guarantees that every listed dataset is compatible. Keep post-treatment specimens separate from pretreatment cohorts, and confirm that each public dataset's labels and specimen timing match the intended prediction task.
