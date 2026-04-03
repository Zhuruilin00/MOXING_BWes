from __future__ import annotations

import argparse

from src.star_predictor.benchmark import compare_models
from src.star_predictor.pipeline import get_training_preset_names


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="对比当前模型与其他热门模型在同一滚动验证下的预测效果")
    parser.add_argument("--preset", choices=get_training_preset_names(), default="accuracy-balanced")
    parser.add_argument("--dataset-path", default="artifacts/star_history.csv")
    parser.add_argument("--output-dir", default="artifacts/model_comparison")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = compare_models(
        preset_name=args.preset,
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
    )
    summary = result["summary"]
    print("模型对比完成")
    print(f"汇总文件: {result['summary_path']}")
    print(f"逐折文件: {result['fold_metrics_path']}")
    print(f"逐笔预测文件: {result['predictions_path']}")
    print("按 selection_score 排序的前几名:")
    print(summary.loc[:, ["model", "label", "selection_score", "accuracy", "balanced_accuracy", "auc"]].head(10).round(4).to_string(index=False))


if __name__ == "__main__":
    main()