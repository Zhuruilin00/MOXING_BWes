from __future__ import annotations

import argparse

from src.star_predictor.pipeline import (
    StarMarketDirectionPredictor,
    TrainConfig,
    get_training_preset,
    get_training_preset_names,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="训练科创板未来 3-5 天涨跌方向预测模型")
    parser.add_argument("--preset", choices=get_training_preset_names(), default=None)
    parser.add_argument("--start-date", default="2020-01-01")
    parser.add_argument("--end-date", default="2026-04-01")
    parser.add_argument("--horizon", type=int, choices=[3, 5], default=5)
    parser.add_argument("--neutral-quantile", type=float, default=0.35)
    parser.add_argument("--label-denoise-mode", choices=["fixed", "adaptive"], default="fixed")
    parser.add_argument("--adaptive-neutral-strength", type=float, default=0.3)
    parser.add_argument("--adaptive-neutral-min-quantile", type=float, default=0.25)
    parser.add_argument("--adaptive-neutral-max-quantile", type=float, default=0.995)
    parser.add_argument("--min-history-days", type=int, default=80)
    parser.add_argument("--min-train-days", type=int, default=240)
    parser.add_argument("--valid-days", type=int, default=40)
    parser.add_argument("--gap-days", type=int, default=5)
    parser.add_argument("--universe-limit", type=int, default=None)
    parser.add_argument("--dataset-path", default=None)
    parser.add_argument("--disable-auto-tune", action="store_true")
    parser.add_argument("--disable-sci-factors", action="store_true")
    parser.add_argument("--reuse-sci-cache", action="store_true")
    parser.add_argument("--disable-layered-model", action="store_true")
    parser.add_argument(
        "--selection-objective",
        choices=["signal_quality", "accuracy", "balanced_accuracy", "joint"],
        default="accuracy",
    )
    parser.add_argument("--newborn-days", type=int, default=720)
    parser.add_argument("--min-signal-coverage", type=float, default=0.12)
    parser.add_argument("--target-signal-accuracy", type=float, default=None)
    parser.add_argument("--decision-threshold-min", type=float, default=0.45)
    parser.add_argument("--decision-threshold-max", type=float, default=0.56)
    parser.add_argument("--decision-threshold-step", type=float, default=0.01)
    parser.add_argument("--confidence-threshold-min", type=float, default=0.52)
    parser.add_argument("--confidence-threshold-max", type=float, default=0.90)
    parser.add_argument("--confidence-threshold-step", type=float, default=0.01)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config_kwargs = {
        "start_date": args.start_date,
        "end_date": args.end_date,
        "horizon": args.horizon,
        "neutral_quantile": args.neutral_quantile,
        "label_denoise_mode": args.label_denoise_mode,
        "adaptive_neutral_strength": args.adaptive_neutral_strength,
        "adaptive_neutral_min_quantile": args.adaptive_neutral_min_quantile,
        "adaptive_neutral_max_quantile": args.adaptive_neutral_max_quantile,
        "min_history_days": args.min_history_days,
        "min_train_days": args.min_train_days,
        "valid_days": args.valid_days,
        "gap_days": args.gap_days,
        "universe_limit": args.universe_limit,
        "auto_tune": not args.disable_auto_tune,
        "enable_sci_factors": not args.disable_sci_factors,
        "refresh_incomplete_sci_cache": not args.reuse_sci_cache,
        "enable_layered_model": not args.disable_layered_model,
        "selection_objective": args.selection_objective,
        "training_preset": args.preset,
        "newborn_days": args.newborn_days,
        "min_signal_coverage": args.min_signal_coverage,
        "target_signal_accuracy": args.target_signal_accuracy,
        "decision_threshold_min": args.decision_threshold_min,
        "decision_threshold_max": args.decision_threshold_max,
        "decision_threshold_step": args.decision_threshold_step,
        "confidence_threshold_min": args.confidence_threshold_min,
        "confidence_threshold_max": args.confidence_threshold_max,
        "confidence_threshold_step": args.confidence_threshold_step,
    }
    if args.preset is not None:
        preset = get_training_preset(args.preset)
        config_kwargs.update(preset["train_config"])
        config_kwargs["training_preset"] = args.preset

    config = TrainConfig(
        **config_kwargs,
    )

    predictor = StarMarketDirectionPredictor()
    result = predictor.train(config=config, dataset_path=args.dataset_path)
    metrics = result["metrics"]
    print("训练完成")
    print(f"模型文件: {result['model_path']}")
    print(f"滚动验证文件: {result['metrics_path']}")
    print(f"训练预设: {config.training_preset or '无'}")
    print(f"分层模型: {'启用' if config.enable_layered_model else '关闭'} (newborn_days={config.newborn_days})")
    print(f"科创因子缓存: {'复用现有缓存' if not config.refresh_incomplete_sci_cache else '允许补刷不完整缓存'}")
    print(f"选模目标: {config.selection_objective}")
    print(
        "标签去噪: "
        f"mode={config.label_denoise_mode}, "
        f"base_q={config.neutral_quantile}, "
        f"adaptive_strength={config.adaptive_neutral_strength}, "
        f"adaptive_q_min={config.adaptive_neutral_min_quantile}, "
        f"adaptive_q_max={config.adaptive_neutral_max_quantile}"
    )
    print(f"选择的 neutral_quantile: {result['selected_neutral_quantile']}")
    print(
        "选择的模型参数: "
        f"n_estimators={result['selected_estimator_config'].n_estimators}, "
        f"max_depth={result['selected_estimator_config'].max_depth}, "
        f"min_samples_split={result['selected_estimator_config'].min_samples_split}, "
        f"min_samples_leaf={result['selected_estimator_config'].min_samples_leaf}, "
        f"max_features={result['selected_estimator_config'].max_features}, "
        f"criterion={result['selected_estimator_config'].criterion}, "
        f"class_weight={result['selected_estimator_config'].class_weight}, "
        f"use_rf_blend={result['selected_estimator_config'].use_rf_blend}, "
        f"rf_blend_weight={result['selected_estimator_config'].rf_blend_weight}, "
        f"halflife={result['selected_estimator_config'].sample_weight_halflife_days}, "
        f"max_train_days={result['selected_estimator_config'].max_train_days}"
    )
    print(
        "校准阈值: "
        f"decision_threshold={result['threshold_config'].decision_threshold}, "
        f"decision_metric={result['threshold_config'].decision_metric}, "
        f"confidence_threshold={result['threshold_config'].confidence_threshold}, "
        f"signal_accuracy={result['threshold_config'].signal_accuracy:.4f}, "
        f"signal_balanced_accuracy={result['threshold_config'].signal_balanced_accuracy:.4f}"
    )
    print("验证均值:")
    print(metrics[["auc", "accuracy", "balanced_accuracy", "high_conf_accuracy", "coverage"]].mean(numeric_only=True).round(4).to_string())
    summary = result["cv_summary"]
    if "calibrated_accuracy" in summary:
        print(
            "校准后验证均值: "
            f"accuracy={summary['calibrated_accuracy']:.4f}, "
            f"balanced_accuracy={summary['calibrated_balanced_accuracy']:.4f}, "
            f"high_conf_accuracy={summary['calibrated_high_conf_accuracy']:.4f}, "
            f"coverage={summary['calibrated_coverage']:.4f}"
        )


if __name__ == "__main__":
    main()
