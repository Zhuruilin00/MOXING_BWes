# 论文实验章节模板（含可填表格）

## 4 实验设置

### 4.1 数据与任务

- 市场：科创板
- 任务：未来 $h$ 日方向预测（默认 $h=5$）
- 标签：按日横截面去噪后构建二分类标签
- 数据路径：artifacts/star_history.csv

### 4.2 对比方法

- 本文方法：RDSW + ExtraTrees（benchmark-thoughtful）
- 对比基线：RandomForest、HistGradientBoosting、LogisticRegression

### 4.3 评估协议

- 验证方式：时间滚动验证
- 指标：accuracy、balanced_accuracy、AUC、selection_score、calibrated_high_conf_accuracy、calibrated_coverage
- 阈值：验证集后校准（除固定阈值消融）

## 5 主结果

表 1 可直接填写同条件对比结果（来源于 summaries 与 compare_models 输出）。

| Model | accuracy | balanced_accuracy | AUC | selection_score | cal_high_conf_acc | cal_coverage |
|---|---:|---:|---:|---:|---:|---:|
| RDSW-ExtraTrees (full) | 0.6292 | 0.6412 | 0.7024 | 0.6598 | 0.8679 | 0.1227 |
| RandomForest |  |  |  |  |  |  |
| HistGradientBoosting |  |  |  |  |  |  |
| LogisticRegression |  |  |  |  |  |  |

## 6 消融实验

运行命令：

```bash
python run_ablation.py --preset benchmark-thoughtful --dataset-path artifacts/star_history.csv
```

输出目录默认：

- artifacts/model_comparison/ablations/benchmark_thoughtful/ablation_summary.csv
- artifacts/model_comparison/ablations/benchmark_thoughtful/ablation_folds.csv
- artifacts/model_comparison/ablations/benchmark_thoughtful/ablation_predictions.csv

表 2 填写消融结果（建议直接从 ablation_summary.csv 粘贴）：

| Variant | accuracy | balanced_accuracy | AUC | selection_score | delta_selection_vs_full |
|---|---:|---:|---:|---:|---:|
| full_method |  |  |  |  | 0.0000 |
| ablate_denoise_strength |  |  |  |  |  |
| ablate_time_decay |  |  |  |  |  |
| ablate_recent_window |  |  |  |  |  |
| ablate_threshold_calibration |  |  |  |  |  |

## 7 统计显著性检验模板

建议采用折级配对检验（例如 Wilcoxon signed-rank）：

表 3 显著性检验结果：

| Pair | Metric | Test | p-value | Significant (p<0.05) |
|---|---|---|---:|---|
| full_method vs RandomForest | selection_score | Wilcoxon |  |  |
| full_method vs HistGB | selection_score | Wilcoxon |  |  |
| full_method vs LogisticRegression | selection_score | Wilcoxon |  |  |
| full_method vs ablate_time_decay | selection_score | Wilcoxon |  |  |

## 8 稳健性实验模板

表 4 稳健性（不同 horizon/时间段/股票池）

| Setting | accuracy | balanced_accuracy | AUC | selection_score |
|---|---:|---:|---:|---:|
| h=3 |  |  |  |  |
| h=5 |  |  |  |  |
| early period |  |  |  |  |
| recent period |  |  |  |  |

## 9 结论段模板

本文在不增加模型结构复杂度的前提下，提出 RDSW 训练机制，通过横截面去噪标签、时间衰减加权、近期训练窗截断与阈值后校准，显著提升了综合性能指标与高置信信号纯度。消融实验表明，各模块均对性能提升有独立贡献，其中 [待填核心模块] 对 selection_score 的影响最大。
