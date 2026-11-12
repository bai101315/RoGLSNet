# RoGLSNet IRRG+DSM（独立实验）

这里不依赖 AI-Scientist 的运行器。训练代码直接使用 RoGLSNet，当前只复用已经离线切好的
Vaihingen 1024×1024 数据目录，避免再复制 2.4 GB 数据。

## 当前方案

- IRRG：保留论文中的 CDFormer-S18 + RoGLSNet 解码器。
- DSM：使用约 0.21M 参数的深度可分离卷积金字塔。
- 融合：在 1/4、1/8、1/16、1/32 四个尺度上，用通道门控和空间门控把 DSM 作为残差注入 IRRG 特征。
- 初始化：融合残差从 0 开始，因此初始网络严格等价于单模态光学分支。
- 训练：1024 输入、batch 1、梯度累积 8、100 epochs；每 5 轮覆盖续训断点，仅保留一个最佳模型。
- 评测：area30 选最佳模型；最终 17 个测试区只评测一次，并使用论文风格 D4/多尺度 TTA。

设计取舍参考了以下成熟路线：

- [CMX](https://github.com/huaaaliu/RGBX_Semantic_Segmentation)：分阶段的跨模态特征校准和融合。
- [LMFNet](https://arxiv.org/abs/2404.13659)：轻量、多尺度的遥感多模态融合。
- [AMMNet](https://arxiv.org/abs/2507.16158)：重光学分支配合轻量 DSM 分支的非对称设计。
- [FTransUNet](https://doi.org/10.1109/TGRS.2024.3373033)：Vaihingen/Potsdam 上浅层与深层的多级 RGB-DSM 融合。

完整 CMX 或 FTransUNet 需要两个重型编码器，不适合本机 12 GB RTX 3060 和 1024 输入，
所以这里保留其多尺度特征融合思想，但不用第二个 CDFormer。

## 使用

在 `GeoSeg2` 目录运行：

```bash
/home/pc/桌面/paral/AI-Scientist-v3/.venv/bin/python -m multimodal.train --preflight
/home/pc/桌面/paral/AI-Scientist-v3/.venv/bin/python -m multimodal.train
```

断点续训：

```bash
/home/pc/桌面/paral/AI-Scientist-v3/.venv/bin/python -m multimodal.train \
  --resume runs/vaihingen_agf_1024/checkpoints/resume_checkpoint.pt
```

配置文件是 `vaihingen_agf_1024.json`。正式训练默认从与论文一致的 ImageNet 预训练主干开始，
不使用上一次实验权重，以保证与基线比较公平。如果只是快速验证新模块，也可以把
`initial_optical_checkpoint` 指向旧实验的 `best_model.pt`，但这种结果不能当作公平对比。

