# Benchmark Thoughtful 论文图注模板

图1（comparison_main_metrics.png）图注建议:
在相同滚动验证与阈值校准条件下，各模型在 accuracy、balanced_accuracy、auc 与 selection_score 上的对比结果。当前排名第一模型为 ExtraTrees (current preset) (selection_score=0.6598)。

图2（comparison_thresholds_and_coverage.png）图注建议:
各模型的决策阈值、置信阈值与高置信覆盖率对比。该图用于分析信号纯度与可交易样本规模之间的权衡关系。

图3（comparison_fold_accuracy_trend.png）图注建议:
各模型在时间滚动折上的 accuracy 轨迹。曲线波动用于刻画时序稳定性与阶段适配能力。

---

# Benchmark Thoughtful Paper Caption Templates

Figure 1 (comparison_main_metrics.png) caption draft:
Comparison across models under identical walk-forward validation and threshold calibration, including accuracy, balanced_accuracy, auc, and selection_score. The top-ranked model is ExtraTrees (current preset) (selection_score=0.6598).

Figure 2 (comparison_thresholds_and_coverage.png) caption draft:
Comparison of decision threshold, confidence threshold, and high-confidence coverage. This figure highlights the trade-off between signal purity and tradable sample size.

Figure 3 (comparison_fold_accuracy_trend.png) caption draft:
Per-fold accuracy trajectories over rolling time splits. The volatility of curves reflects temporal stability and regime adaptability.
