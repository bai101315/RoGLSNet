# Vaihingen IRRG + DSM 实验

这里保存了可脱离 AI-Scientist 运行的多模态数据读取、模型、训练、诊断和评测代码。输入始终是离线生成的 `1024×1024` 对齐块；训练框架不负责切图。

## 已实现模型

- `InputFusionRoGLSNet`：把 DSM 映射到三通道，在输入端以门控残差注入 IRRG。它的绝对指标最高，但置零/打乱对照表明 DSM 几乎没有贡献，因此只适合作为较强的光学核心。
- `AsymmetricFusionRoGLSNet`：在 CDFormer 四层特征上加入 DSM 金字塔，实验表明 DSM 没有被有效利用。
- `HeightSelectiveRoGLSNet`：只在全局语义、局部语义和边界处选择性注入高度，并带高度边界辅助监督。DSM 利用通过了置零/打乱对照，但绝对精度仍低于输入级基线。

完整结果和逐轮记录见 [`../experiments/vaihingen_multimodal_20261113`](../experiments/vaihingen_multimodal_20261113)。大权重不进入 Git 历史，发布在 GitHub Release `vaihingen-multimodal-20261113`。

## 环境和数据

先安装与显卡匹配的 PyTorch，再安装：

```bash
cd GeoSeg2
python -m pip install -r requirements.txt
```

数据目录必须包含 `manifest.json`，以及如下完全同名、逐像素对齐的数据：

```text
dataset_paper_1024/
├── manifest.json
├── train/{primary,auxiliary,masks}/
├── val/{primary,auxiliary,masks}/
└── test/{primary,auxiliary,masks}/
```

Vaihingen 数据受官方许可约束，因此仓库和 Release 均不包含原图或切块。论文划分、样本数和 DSM 归一化参数已保存在实验目录的 manifest 快照中。

从原始 DSM 生成按区域 `5%–95%` 截断的高度版本：

```bash
python -m multimodal.prepare_height_dataset \
  --source-manifest /path/dataset_paper_1024/manifest.json \
  --raw-dsm-dir /path/DSM \
  --output-dir /path/dataset_heightnorm_1024
```

该命令只创建新的 DSM 块，并用相对符号链接复用 IRRG 和标签，所以搬迁时要把两个数据目录放在同一父目录下。

## 可迁移运行方式

历史 JSON 保留原实验参数。换电脑时不要改历史文件中的绝对路径，使用环境变量覆盖：

```bash
cd GeoSeg2
export ROGLS_DATASET_MANIFEST=/data/dataset_heightnorm_1024/manifest.json
export ROGLS_OUTPUT_DIR=$PWD/runs/stage1_3090

python -m multimodal.height_train \
  --config multimodal/vaihingen_height_stage1_1024.json \
  --preflight
python -m multimodal.height_train \
  --config multimodal/vaihingen_height_stage1_1024.json
```

第二阶段还需指定第一阶段最佳权重：

```bash
export ROGLS_INITIAL_CHECKPOINT=/path/stage1_best_model.pt
export ROGLS_OUTPUT_DIR=$PWD/runs/stage2_3090
python -m multimodal.height_train \
  --config multimodal/vaihingen_height_stage2_1024.json \
  --preflight
python -m multimodal.height_train \
  --config multimodal/vaihingen_height_stage2_1024.json
```

断点续训在同一组环境变量下增加：

```bash
python -m multimodal.height_train \
  --config multimodal/vaihingen_height_stage1_1024.json \
  --resume /path/resume_checkpoint.pt
```

输入级基线可用 `multimodal/vaihingen_input_fusion_1024.json` 与 `multimodal.train` 复现。RTX 3060 用 batch 1、累积 8；RTX 3090 建议先预检 batch 4、累积 2，保持有效 batch 8。

## DSM 有效性诊断

以下命令只推理，不训练：

```bash
python -m multimodal.diagnose_auxiliary \
  --model input_fusion \
  --config multimodal/vaihingen_input_fusion_1024.json \
  --checkpoint /path/input_fusion_best_model.pt \
  --output runs/input_fusion_diagnostics.json
```

报告比较正常 DSM、全零 DSM 和跨区域打乱 DSM。正常 DSM 的 mF1 至少高于最强对照 `0.002`，才能把提升归因于第二模态。

## 3090 后续方向

不要原样重跑已经失败的四层融合。下一版加载输入级权重中的 `segmenter` 作为光学核心，不把几乎失效的输入门控当作成果；另用区域归一化 DSM 提取坡度、局部高差和边缘，只对 RoAFM value 与最终 logits 做有界残差修正。详细门槛和训练顺序见 [`HANDOFF_20261113.md`](HANDOFF_20261113.md)。
