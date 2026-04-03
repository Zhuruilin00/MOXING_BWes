from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd

from src.star_predictor.benchmark import _resolve_decision_metric, _run_walk_forward_validation
from src.star_predictor.data import load_dataset
from src.star_predictor.features import make_feature_frame
from src.star_predictor.model import (
    EstimatorConfig,
    build_estimator,
    calibrate_thresholds,
    summarize_cv_metrics,
    summarize_validation_predictions,
)
from src.star_predictor.pipeline import TrainConfig, get_training_preset, get_training_preset_names
from src.star_predictor.sci_factors import SciFactorConfig, build_sci_factor_frame


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="运行 benchmark-thoughtful 学术消融实验")
    parser.add_argument("--preset", choices=get_training_preset_names(), default="benchmark-thoughtful")
    parser.add_argument("--dataset-path", default="artifacts/star_history.csv")
    parser.add_argument("--output-dir", default="artifacts/model_comparison/ablations/benchmark_thoughtful")
    parser.add_argument("--start-date", default=None, help="可选：仅使用该日期及之后的数据，例如 2023-01-01")
    parser.add_argument("--end-date", default=None, help="可选：仅使用该日期及之前的数据，例如 2025-12-31")
    parser.add_argument("--horizon", type=int, default=None, help="可选：覆盖 preset 中的 horizon 用于稳健性实验")
    parser.add_argument("--include-benchmark-models", action="store_true", help="同时输出 compare_models 全模型基线")
    parser.add_argument("--benchmark-output-dir", default="artifacts/model_comparison/ablations/benchmark_models")
    return parser


def _run_single_variant(
    name: str,
    label: str,
    train_config: TrainConfig,
    estimator_config: EstimatorConfig,
    dataset: pd.DataFrame,
    force_thresholds: tuple[float, float] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    sci_factor_frame = (
        build_sci_factor_frame(
            dataset,
            SciFactorConfig(refresh_incomplete_cache=train_config.refresh_incomplete_sci_cache),
        )
        if train_config.enable_sci_factors
        else None
    )

    frame, feature_names = make_feature_frame(
        dataset,
        horizon=train_config.horizon,
        neutral_quantile=train_config.neutral_quantile,
        sci_factor_frame=sci_factor_frame,
        label_denoise_mode=train_config.label_denoise_mode,
        adaptive_neutral_strength=train_config.adaptive_neutral_strength,
        adaptive_neutral_min_quantile=train_config.adaptive_neutral_min_quantile,
        adaptive_neutral_max_quantile=train_config.adaptive_neutral_max_quantile,
    )

    metrics, predictions = _run_walk_forward_validation(
        frame=frame,
        feature_names=list(feature_names),
        min_train_days=train_config.min_train_days,
        valid_days=train_config.valid_days,
        gap_days=train_config.gap_days,
        estimator_factory=lambda: build_estimator(estimator_config),
        estimator_config=estimator_config,
        enable_layered_model=train_config.enable_layered_model,
        newborn_days=train_config.newborn_days,
    )

    decision_metric = _resolve_decision_metric(train_config.selection_objective)
    if force_thresholds is None:
        threshold_config = calibrate_thresholds(
            predictions,
            min_signal_coverage=train_config.min_signal_coverage,
            target_signal_accuracy=train_config.target_signal_accuracy,
            decision_metric=decision_metric,
            decision_threshold_min=train_config.decision_threshold_min,
            decision_threshold_max=train_config.decision_threshold_max,
            decision_threshold_step=train_config.decision_threshold_step,
            confidence_threshold_min=train_config.confidence_threshold_min,
            confidence_threshold_max=train_config.confidence_threshold_max,
            confidence_threshold_step=train_config.confidence_threshold_step,
        )
        decision_threshold = threshold_config.decision_threshold
        confidence_threshold = threshold_config.confidence_threshold
        used_decision_metric = threshold_config.decision_metric
        threshold_mode = "calibrated"
    else:
        decision_threshold, confidence_threshold = force_thresholds
        used_decision_metric = "fixed"
        threshold_mode = "fixed"

    calibrated_summary = summarize_validation_predictions(
        predictions,
        decision_threshold=decision_threshold,
        confidence_threshold=confidence_threshold,
    )
    summary = summarize_cv_metrics(
        metrics,
        selection_objective=train_config.selection_objective,
        calibrated_summary=calibrated_summary,
    )

    row: dict[str, Any] = {
        "variant": name,
        "label": label,
        "threshold_mode": threshold_mode,
        "decision_metric": used_decision_metric,
        "decision_threshold": decision_threshold,
        "confidence_threshold": confidence_threshold,
        "neutral_quantile": train_config.neutral_quantile,
        "label_denoise_mode": train_config.label_denoise_mode,
        "sample_weight_halflife_days": estimator_config.sample_weight_halflife_days,
        "max_train_days": estimator_config.max_train_days,
        "enable_layered_model": train_config.enable_layered_model,
        **summary,
    }

    fold_out = metrics.copy()
    fold_out.insert(0, "variant", name)
    fold_out.insert(1, "label", label)

    pred_out = predictions.copy()
    pred_out.insert(0, "variant", name)
    pred_out.insert(1, "label", label)

    return fold_out, pred_out, row


def main() -> None:
    args = build_parser().parse_args()
    preset = get_training_preset(args.preset)
    base_train = TrainConfig(training_preset=args.preset, **preset["train_config"])
    if args.horizon is not None:
        base_train = replace(base_train, horizon=args.horizon)
    base_est: EstimatorConfig = preset["estimator_config"]

    dataset = load_dataset(args.dataset_path)
    if args.start_date is not None:
        start_ts = pd.Timestamp(args.start_date)
        dataset = dataset.loc[pd.to_datetime(dataset["date"]) >= start_ts].copy()
    if args.end_date is not None:
        end_ts = pd.Timestamp(args.end_date)
        dataset = dataset.loc[pd.to_datetime(dataset["date"]) <= end_ts].copy()
    if dataset.empty:
        raise RuntimeError("筛选后数据为空，请检查 --start-date/--end-date 参数")

    variants: list[dict[str, Any]] = [
        {
            "name": "full_method",
            "label": "RDSW full (benchmark-thoughtful)",
            "train": base_train,
            "est": base_est,
            "force_thresholds": None,
        },
        {
            "name": "ablate_denoise_strength",
            "label": "Ablation: weaker denoising",
            "train": replace(base_train, neutral_quantile=0.95),
            "est": base_est,
            "force_thresholds": None,
        },
        {
            "name": "ablate_time_decay",
            "label": "Ablation: remove time decay",
            "train": base_train,
            "est": replace(base_est, sample_weight_halflife_days=100000),
            "force_thresholds": None,
        },
        {
            "name": "ablate_recent_window",
            "label": "Ablation: full history window",
            "train": base_train,
            "est": replace(base_est, max_train_days=None),
            "force_thresholds": None,
        },
        {
            "name": "ablate_threshold_calibration",
            "label": "Ablation: fixed thresholds 0.50/0.60",
            "train": base_train,
            "est": base_est,
            "force_thresholds": (0.50, 0.60),
        },
    ]

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    all_folds: list[pd.DataFrame] = []
    all_predictions: list[pd.DataFrame] = []
    rows: list[dict[str, Any]] = []

    for variant in variants:
        fold_frame, pred_frame, row = _run_single_variant(
            name=variant["name"],
            label=variant["label"],
            train_config=variant["train"],
            estimator_config=variant["est"],
            dataset=dataset,
            force_thresholds=variant["force_thresholds"],
        )
        all_folds.append(fold_frame)
        all_predictions.append(pred_frame)
        rows.append(row)

    summary = pd.DataFrame(rows).sort_values("selection_score", ascending=False).reset_index(drop=True)
    baseline = summary.loc[summary["variant"] == "full_method"].iloc[0]
    summary["delta_selection_score_vs_full"] = summary["selection_score"] - float(baseline["selection_score"])
    summary["delta_accuracy_vs_full"] = summary["accuracy"] - float(baseline["accuracy"])
    summary["delta_balanced_accuracy_vs_full"] = summary["balanced_accuracy"] - float(baseline["balanced_accuracy"])
    summary["delta_auc_vs_full"] = summary["auc"] - float(baseline["auc"])

    folds = pd.concat(all_folds, ignore_index=True)
    predictions = pd.concat(all_predictions, ignore_index=True)

    summary_path = output_root / "ablation_summary.csv"
    folds_path = output_root / "ablation_folds.csv"
    predictions_path = output_root / "ablation_predictions.csv"

    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    folds.to_csv(folds_path, index=False, encoding="utf-8-sig")
    predictions.to_csv(predictions_path, index=False, encoding="utf-8-sig")

    print("消融实验完成")
    print(f"汇总文件: {summary_path}")
    print(f"逐折文件: {folds_path}")
    print(f"逐笔预测文件: {predictions_path}")
    print("按 selection_score 排序:")
    print(
        summary.loc[
            :,
            [
                "variant",
                "selection_score",
                "accuracy",
                "balanced_accuracy",
                "auc",
                "calibrated_high_conf_accuracy",
                "calibrated_coverage",
                "delta_selection_score_vs_full",
            ],
        ]
        .round(4)
        .to_string(index=False)
    )

    if args.include_benchmark_models:
        from src.star_predictor.benchmark import compare_models

        benchmark_result = compare_models(
            preset_name=args.preset,
            dataset_path=args.dataset_path,
            output_dir=args.benchmark_output_dir,
        )
        print("\n已额外输出全模型对比基线:")
        print(f"汇总文件: {benchmark_result['summary_path']}")


if __name__ == "__main__":
    main()
