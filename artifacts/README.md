# artifacts 目录索引

当前目录按“核心运行文件 + 归档 + 对比结果”组织。

## 1. 核心运行文件（默认路径）

- star_history.csv: 当前默认训练/对比使用的数据集
- sci_factor_cache.csv: 当前因子缓存
- star_direction_h5.joblib: 当前 5 日模型文件
- cv_metrics_h5.csv: 当前训练逐折指标
- cv_predictions_h5.csv: 当前训练逐笔预测
- latest_predictions_h5.csv: 当前最新预测输出

## 2. 对比结果

- model_comparison/: 当前主线对比实验（已整理）
- model_comparison/README.md: model_comparison 子目录说明

## 3. 历史归档

- archive/runs/: 历史实验目录归档
- archive/datasets/: 历史数据快照归档

说明：
- 为了不影响现有脚本默认参数，核心运行文件仍保留在 artifacts 根目录。
- 历史实验与非默认数据快照已移动到 archive，避免根目录堆积。
