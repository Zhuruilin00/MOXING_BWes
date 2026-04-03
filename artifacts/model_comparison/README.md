# model_comparison 目录说明

本目录仅保留当前主线可复现实验输出，按用途分为三层：

- summaries/
- charts/
- reports/
- publication_pack/

## 1. summaries

- summaries/benchmark_thoughtful/summary_h5.csv: 当前推荐主结论的汇总指标
- summaries/benchmark_thoughtful/folds_h5.csv: 当前推荐主结论的逐折指标
- summaries/benchmark_thoughtful/predictions_h5.csv: 当前推荐主结论的逐笔预测
- summaries/benchmark_best/summary_h5.csv: 上一版基准配置的汇总指标
- summaries/benchmark_best/folds_h5.csv: 上一版基准配置的逐折指标
- summaries/benchmark_best/predictions_h5.csv: 上一版基准配置的逐笔预测

## 2. charts

- charts/benchmark_thoughtful/comparison_main_metrics.png
- charts/benchmark_thoughtful/comparison_thresholds_and_coverage.png
- charts/benchmark_thoughtful/comparison_fold_accuracy_trend.png
- charts/benchmark_thoughtful/comparison_chart_notes.md
- charts/benchmark_thoughtful/paper_figure_captions.md

## 3. reports

- reports/academic_innovation_blueprint.md
- reports/paper_method_section_draft.md
- reports/paper_experiment_template.md

## 4. 重生成命令

```bash
python compare_models.py --preset benchmark-thoughtful --dataset-path artifacts/star_history.csv
python visualize_comparison.py \
  --summary-csv artifacts/model_comparison/summaries/benchmark_thoughtful/summary_h5.csv \
  --folds-csv artifacts/model_comparison/summaries/benchmark_thoughtful/folds_h5.csv \
  --output-dir artifacts/model_comparison/charts/benchmark_thoughtful \
  --notes-language bilingual \
  --title "Benchmark Thoughtful"
```

投稿证据包一键生成：

```bash
python run_publication_pack.py --dataset-path artifacts/star_history.csv --root-output artifacts/model_comparison/publication_pack
```
