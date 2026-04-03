# Benchmark Thoughtful 图表说明

## 总览

当前第一名: ExtraTrees (current preset) | selection_score=0.6598, accuracy=0.6292, balanced_accuracy=0.6412, auc=0.7024

建议阅读顺序: 先看主指标图，再看阈值与覆盖率，最后看逐折趋势。

## comparison_main_metrics.png

图意: 展示各模型在 accuracy、balanced_accuracy、auc、selection_score 四项核心指标上的整体对比。
解读重点: selection_score 用于综合排名；accuracy 代表方向命中；balanced_accuracy 反映涨跌两类识别均衡性；auc 反映概率排序质量。
决策建议: 若 selection_score 差距在 0.01 以内，优先选 balanced_accuracy 和 auc 同时更高者。

## comparison_thresholds_and_coverage.png

图意: 对比各模型的 decision_threshold、confidence_threshold 与高置信 coverage。
解读重点: confidence_threshold 越高，coverage 常会下降；coverage 过低会导致可交易信号不足。
决策建议: 在 coverage 可接受的前提下，优先能保持更高 calibrated_high_conf_accuracy 的模型。

## comparison_fold_accuracy_trend.png

图意: 展示每个模型在各时间折上的 accuracy 变化轨迹，用于观察时序稳定性。
解读重点: 均值高不代表稳健，需同时观察波动幅度与最近折是否失速。
各模型逐折统计(按均值降序、波动升序参考):

| label | fold_accuracy_mean | fold_accuracy_std |
|---|---:|---:|
| RandomForestClassifier | 0.6435 | 0.1240 |
| ExtraTrees (current preset) | 0.6292 | 0.1209 |
| HistGradientBoostingClassifier | 0.6285 | 0.0870 |
| LogisticRegression | 0.5747 | 0.1148 |

---

# Benchmark Thoughtful Chart Notes

## Overview

Top model: ExtraTrees (current preset) | selection_score=0.6598, accuracy=0.6292, balanced_accuracy=0.6412, auc=0.7024

Suggested reading order: main metrics first, then thresholds/coverage, then fold trend.

## comparison_main_metrics.png

Purpose: Compare overall performance across accuracy, balanced_accuracy, auc, and selection_score.
Interpretation: selection_score is the integrated rank; accuracy measures directional hit-rate; balanced_accuracy reflects class balance; auc captures probability ranking quality.
Decision rule: if selection_score gap is within 0.01, prefer the model with both higher balanced_accuracy and auc.

## comparison_thresholds_and_coverage.png

Purpose: Compare decision_threshold, confidence_threshold, and high-confidence coverage.
Interpretation: higher confidence_threshold often lowers coverage; too low coverage may lead to insufficient tradable signals.
Decision rule: under acceptable coverage, prioritize higher calibrated_high_conf_accuracy.

## comparison_fold_accuracy_trend.png

Purpose: Show per-fold accuracy trajectories to evaluate temporal stability.
Interpretation: higher mean alone is insufficient; inspect volatility and late-fold deterioration.
Per-model fold statistics (sorted by higher mean and lower std):

| label | fold_accuracy_mean | fold_accuracy_std |
|---|---:|---:|
| RandomForestClassifier | 0.6435 | 0.1240 |
| ExtraTrees (current preset) | 0.6292 | 0.1209 |
| HistGradientBoostingClassifier | 0.6285 | 0.0870 |
| LogisticRegression | 0.5747 | 0.1148 |

## Generated Files

- comparison_main_metrics.png
- comparison_thresholds_and_coverage.png
- comparison_fold_accuracy_trend.png