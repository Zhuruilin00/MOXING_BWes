# 投稿就绪评估报告

## 1. 主实验核心结果

- selection_score: 0.6598
- accuracy: 0.6292
- balanced_accuracy: 0.6412
- auc: 0.7024
- calibrated_high_conf_accuracy: 0.8679
- calibrated_coverage: 0.1227

## 2. 稳健性摘要

- horizon=3 selection_score: 0.6154
- recent period start_date: 2023-01-01
- recent period selection_score: 0.6499

## 3. 证据检查清单

- main_selection_score_ge_0p65: PASS
- main_auc_ge_0p70: PASS
- main_high_conf_acc_ge_0p80: PASS
- ablation_supports_method: PASS
- robust_h3_selection_score_ge_0p60: PASS
- robust_recent_selection_score_ge_0p60: PASS
- significant_edge_exists: FAIL

## 4. 评估结论

- grade: ready_for_application_or_empirical_journal

## 5. 关键输出路径

- ablation main: artifacts\model_comparison\publication_pack\ablation_main\ablation_summary.csv
- robustness h3: artifacts\model_comparison\publication_pack\robustness_h3\ablation_summary.csv
- robustness recent: artifacts\model_comparison\publication_pack\robustness_recent\ablation_summary.csv
- significance: artifacts\model_comparison\publication_pack\significance_table.csv