# Vaihingen 最终对比

| Method | OA | mF1 | mIoU | DSM effective | DSM mF1 margin |
|---|---:|---:|---:|---|---:|
| RoGLSNet paper single-modal | 93.28 | 91.31 | 84.31 | n/a | n/a |
| Input-level IRRG+DSM | 91.09 | 90.41 | 82.70 | False | 0.001 pp |
| Four-stage gated fusion | 90.16 | 89.34 | 80.96 | False | -0.019 pp |
| HeightSelective stage 1 | 90.71 | 89.78 | 81.68 | True | 0.552 pp |
| HeightSelective stage 2 | 90.71 | 89.85 | 81.79 | True | 0.531 pp |

mF1/mIoU 为前五类平均，OA 包含所有有效类别。
