# 基于 H&E 特征的乳腺癌 Biomarker 预测

本项目提供一个用于 ER、PR、HER2 三任务预测的多实例学习（MIL）基线。输入为预先提取的 H&E 全视野切片（WSI）HDF5 patch 特征。模型为每个 biomarker 设置独立的可学习 query 和分类头；三个 query 从共享的 patch 特征集合中分别读取信息，并通过 query 正交惩罚减少任务间冗余。

整体流程分为三步：

1. 预先生成并保存按患者分组的 K-fold train/validation/test 清单。
2. 每个 fold 训练一个模型，根据 validation 指标选择最佳权重，并在对应的内部 test fold 上评估。
3. 将每个 fold 的最佳权重用于各中心外部 CSV；默认使用 5 个模型的平均预测概率作为集成结果。

## 1. 安装

需要 Python 3.10 或更新版本。

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -e .
```

请根据机器环境安装合适的 PyTorch CPU 或 CUDA 版本。安装方式见 [PyTorch 官方安装页面](https://pytorch.org/get-started/locally/)。

## 2. 输入数据

开发集使用一个 CSV，每个外部中心各使用一个 CSV。前四列格式如下：

```csv
h5_path,ER,PR,HER2,patient_id
features/case001.h5,1,1,0,P001
features/case002.h5,0,0,1,P002
```

第一列是 HDF5 特征文件路径，列名可以任意；接下来三列必须依次命名为 `ER`、`PR`、`HER2`。`patient_id` 是可选列；若 CSV 每行都对应不同患者，可以不提供。标签取值为 `0` 或 `1`。未知标签可以留空或设为 `-1`，相应任务的损失和指标计算会忽略该标签。允许包含其他额外列。

每个 HDF5 文件需要包含一个二维数据集，默认键名为 `features`，形状为 `[patch 数量, 特征维度]`。如键名不同，可通过 `--h5-key` 指定。特征文件路径默认相对于 CSV 所在目录解析；也可以通过 `--feature-root` 指定统一的根目录。

**如果同一患者有多张切片，请在所有中心使用全局唯一的患者 ID。** 划分时会按该 ID 分组，同一患者已知的 biomarker 标签必须一致。如果没有 `patient_id`，程序会将每个 H5/切片视为一个独立样本并发出警告；这只适用于每行确实对应不同患者的情况。外部 CSV 不含患者 ID 时，指标按切片级计算。

只有四列时，CSV 可以写成：

```csv
h5_path,ER,PR,HER2
features/case001.h5,1,1,0
features/case002.h5,0,0,1
```

## 3. 预先划分 K-fold

默认生成 5 个外层 fold。每个 fold 将五分之一患者留作内部 test；剩余患者中再划出默认 20% 作为 validation，其余用于训练。标签组合样本数足够时采用 ER/PR/HER2 联合分层；样本不足时依次回退到单任务分层或非分层划分。

```bash
python prepare_splits.py \
  --csv data/development.csv \
  --output-dir splits/development_5fold \
  --n-folds 5 \
  --val-fraction 0.2 \
  --seed 42 \
  --check-files
```

PowerShell 中可使用反引号续行，也可以将命令写成一行。`--check-files` 会检查所有 H5 文件是否存在。输出包括 `fold_0` 至 `fold_4`，每个目录包含 `train.csv`、`val.csv` 和 `test.csv`；此外还会生成 `all.csv` 和 `split_metadata.json`。训练时会校验保存的文件哈希，检测划分清单是否被修改。每组实验应固定并复用同一个 split 目录；更换随机种子或划分方案时请使用新的输出目录。

如果同一患者有多张 H&E 切片，提供 `patient_id` 后这些切片会被放在同一数据分区。没有患者 ID 时，只能保证切片级划分。

## 4. 训练、内部评估和多中心外测

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

每个外测中心通过一次 `--external CENTER_NAME=CSV_PATH` 传入，可重复指定多个中心。外部 CSV 需要包含同样的三个 biomarker 标签列和特征文件路径；可以不含 `patient_id`，此时按切片级计算指标。H5 路径默认相对于各自外部 CSV 所在目录解析，也可以通过 `--external-feature-root` 设置统一根目录。正式训练前，程序会检查外部队列与开发队列之间是否存在重复患者 ID 或 H5 路径。

每个 fold 的 `fold_N/best.pt` 保存 validation 表现最佳的 checkpoint。只有 validation 中同时包含阳性和阴性的任务才参与 AUROC；如果没有任务满足这一条件，则按 validation macro log-loss 选择权重。选定 checkpoint 后才会评估内部 test。外部评估会分别报告五个 fold 模型的结果，并对预测概率取平均作为集成结果。外部 CSV 不应用于选择 epoch、阈值或模型。

常用参数：

```text
--h5-key features              HDF5 数据集键名
--hidden-dim 256               投影后的特征/query 维度
--num-heads 4                  cross-attention 头数
--query-lambda 0.01             query 去相关惩罚系数
--max-patches 4096              训练时随机抽取的 patch 上限；0 表示使用全部
--eval-max-patches 0            评估时确定性抽取的 patch 上限；0 表示使用全部
--class-weight none             也可选 balanced，按每折训练患者标签计算
--selection-metric macro_auroc  也可选 log_loss
--device auto                   auto、cpu、cuda 或其他 torch device
--amp                           在 CUDA 上启用混合精度
```

如果已经完成训练，之后可用相同的五个 checkpoint 重新评估外部中心：

```bash
python evaluate.py \
  --run-dir runs/query_mil_5fold \
  --external center_a=data/external/center_a.csv \
  --external center_b=data/external/center_b.csv \
  --device auto
```

## 5. 输出文件

```text
runs/query_mil_5fold/
├── config.json
├── development_manifest.csv
├── fold_0/
│   ├── best.pt
│   ├── history.json
│   ├── val/                 # 切片/患者预测和指标
│   ├── test/                # 留出的内部测试预测和指标
│   └── summary.json
├── ... fold_1 至 fold_4
├── fold_summaries.json
├── oof_test_predictions.csv # 每位开发集患者恰好出现在一个 test fold
├── oof_test_metrics.json
└── external/
    ├── center_a/fold_0 ... fold_4/
    ├── center_a/ensemble/
    ├── center_b/...
    ├── pooled/
    └── summary.json
```

指标包括 AUROC、AUPRC、准确率、敏感度、特异度、PPV、NPV、F1、Brier score 和 log-loss。有患者 ID 时，先对同一患者的切片概率取平均，再按患者级计算指标；没有患者 ID 时按切片级计算。结果同时保存中心内指标和合并指标。建议报告各中心结果，因为合并结果可能掩盖中心间差异。如果某个中心的某个 biomarker 只有一个类别，AUROC 和 AUPRC 无法定义，结果会记为 `null` 并说明原因。

## 模型说明

HDF5 patch 特征先投影到共享 embedding。ER、PR、HER2 分别使用独立的可学习任务 query 对 patch 特征做 cross-attention，并由各自的分类头输出预测；任务 query 之间不做注意力交互。训练目标为可用标签上的 masked binary cross-entropy，加上带权重的任务 query 两两平方余弦相似度惩罚。这是用于研究的基线流程，不能替代 IHC/ISH 检测。

## 可复现性与数据泄漏防护

- 提供 `patient_id` 时按患者而不是切片划分。
- 五折外层划分中，每位患者恰好进入一个外层 test fold。
- 训练时检查 split 文件哈希；清单被修改后会停止运行。
- 外部队列会与开发队列核对患者 ID 和 H5 路径是否重叠。
- Checkpoint 记录 run ID、开发集清单哈希、fold、最佳 epoch、模型配置和 validation 选模指标。
- 默认阈值为 0.5，并记录在配置中。不要根据外部队列结果调节阈值或选择模型。

## 项目文件

- `prepare_splits.py`：生成并固定患者级 fold 划分。
- `train.py`：训练所有或指定 fold，评估内部测试集，可选运行多中心外测。
- `evaluate.py`：使用已完成的五折实验重新评估外部队列。
- `biomarkers/model.py`：query-decoupled MIL、query 惩罚和多任务 masked loss。
- `biomarkers/data.py`：CSV 校验和 HDF5 特征读取。
- `biomarkers/metrics.py`：患者级聚合和分类指标。

## 多中心队列的用途规划

所有队列需先整理为上述 CSV/HDF5 格式。下表为候选用途，实际使用前需核实标签和样本信息：

| 队列 | ER/PR/HER2 标签 | 潜在用途 |
| --- | --- | --- |
| 河北省第四医院开发队列 | 三项均有 | 开发和内部交叉验证 |
| 河北省第四医院独立留出队列 | 三项均有 | 额外内部验证；需确保不参与任何模型选择 |
| 参与研究的多中心队列 | 三项均有 | 按中心报告的外部验证 |
| BCNB、TCGA-BRCA、CPTAC-BRCA | 使用前核实标签和特征兼容性 | 公共外部验证候选 |
| HEROHE、HER2 Warwick | 主要针对 HER2 | HER2 外部基准候选 |
| Post-NAT-BRCA | 核实治疗时点和标签定义 | 新辅助治疗后队列的独立验证 |

这些队列仅为候选，并不代表本项目已下载或保证其与当前任务兼容。新辅助治疗后的样本应与治疗前队列分开分析，并确认公开数据集的标签定义和样本时点符合研究问题。
