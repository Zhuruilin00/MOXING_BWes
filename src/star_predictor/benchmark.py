from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Sequence

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data import load_dataset
from .features import make_feature_frame
from .model import (
    FoldResult,
    EstimatorConfig,
    _select_recent_training_data,
    build_estimator,
    build_training_sample_weights,
    calibrate_thresholds,
    summarize_cv_metrics,
    summarize_validation_predictions,
)
from .pipeline import TrainConfig, get_training_preset
from .sci_factors import SciFactorConfig, build_sci_factor_frame


@dataclass(frozen=True)
class BenchmarkModelSpec:
    name: str
    label: str
    estimator_factory: Callable[[], Pipeline]


def get_benchmark_model_specs(current_config: EstimatorConfig) -> list[BenchmarkModelSpec]:
    current_label = "ExtraTrees (current preset)"
    if current_config.use_rf_blend:
        current_label = "ExtraTrees+RandomForest Blend (current preset)"

    return [
        BenchmarkModelSpec(
            name="current_extra_trees",
            label=current_label,
            estimator_factory=lambda: build_estimator(current_config),
        ),
        BenchmarkModelSpec(
            name="random_forest",
            label="RandomForestClassifier",
            estimator_factory=lambda: Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "classifier",
                        RandomForestClassifier(
                            n_estimators=400,
                            max_depth=10,
                            min_samples_split=45,
                            min_samples_leaf=16,
                            max_features=0.55,
                            class_weight="balanced_subsample",
                            n_jobs=-1,
                            random_state=42,
                        ),
                    ),
                ]
            ),
        ),
        BenchmarkModelSpec(
            name="hist_gradient_boosting",
            label="HistGradientBoostingClassifier",
            estimator_factory=lambda: Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "classifier",
                        HistGradientBoostingClassifier(
                            learning_rate=0.05,
                            max_depth=6,
                            max_iter=300,
                            min_samples_leaf=20,
                            l2_regularization=0.05,
                            random_state=42,
                        ),
                    ),
                ]
            ),
        ),
        BenchmarkModelSpec(
            name="logistic_regression",
            label="LogisticRegression",
            estimator_factory=lambda: Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    (
                        "classifier",
                        LogisticRegression(
                            C=0.7,
                            max_iter=2000,
                            class_weight="balanced",
                            random_state=42,
                        ),
                    ),
                ]
            ),
        ),
    ]


def get_benchmark_model_names(current_config: EstimatorConfig | None = None) -> list[str]:
    config = current_config or EstimatorConfig()
    return [spec.name for spec in get_benchmark_model_specs(config)]


def _fit_pipeline(estimator: Pipeline, x_train: pd.DataFrame, y_train: pd.Series, sample_weights) -> None:
    last_step_name = estimator.steps[-1][0]
    estimator.fit(x_train, y_train, **{f"{last_step_name}__sample_weight": sample_weights})


def _run_walk_forward_validation(
    frame: pd.DataFrame,
    feature_names: Sequence[str],
    min_train_days: int,
    valid_days: int,
    gap_days: int,
    estimator_factory: Callable[[], Pipeline],
    estimator_config: EstimatorConfig,
    enable_layered_model: bool,
    newborn_days: int = 720,
    listing_age_col: str = "listing_age_days",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    unique_dates = pd.Index(sorted(frame["date"].unique()))
    fold_results: list[FoldResult] = []
    predictions: list[pd.DataFrame] = []

    split_start = min_train_days
    while split_start + gap_days + valid_days <= len(unique_dates):
        train_dates = unique_dates[:split_start]
        valid_dates = unique_dates[split_start + gap_days : split_start + gap_days + valid_days]

        train_data = frame.loc[frame["date"].isin(train_dates)].copy()
        valid_data = frame.loc[frame["date"].isin(valid_dates)].copy()
        train_data = _select_recent_training_data(train_data, estimator_config.max_train_days)
        if train_data.empty or valid_data.empty or train_data["target"].nunique() < 2:
            split_start += valid_days
            continue

        if listing_age_col not in train_data.columns:
            train_data[listing_age_col] = pd.NA
            valid_data[listing_age_col] = pd.NA

        sample_weights = build_training_sample_weights(
            train_data["date"],
            train_data.get("future_return"),
            halflife_days=estimator_config.sample_weight_halflife_days,
            amplitude_scale=estimator_config.sample_weight_amplitude_scale,
        )

        if not enable_layered_model:
            estimator = estimator_factory()
            _fit_pipeline(estimator, train_data[list(feature_names)], train_data["target"], sample_weights)
            valid_probability = estimator.predict_proba(valid_data[list(feature_names)])[:, 1]
        else:
            train_newborn = train_data[listing_age_col].le(newborn_days)
            valid_newborn = valid_data[listing_age_col].le(newborn_days)

            fallback_estimator = estimator_factory()
            _fit_pipeline(fallback_estimator, train_data[list(feature_names)], train_data["target"], sample_weights)

            newborn_estimator = None
            if train_newborn.sum() >= 120 and train_data.loc[train_newborn, "target"].nunique() >= 2:
                newborn_estimator = estimator_factory()
                newborn_weights = build_training_sample_weights(
                    train_data.loc[train_newborn, "date"],
                    train_data.loc[train_newborn, "future_return"],
                    halflife_days=estimator_config.sample_weight_halflife_days,
                    amplitude_scale=estimator_config.sample_weight_amplitude_scale,
                )
                _fit_pipeline(
                    newborn_estimator,
                    train_data.loc[train_newborn, list(feature_names)],
                    train_data.loc[train_newborn, "target"],
                    newborn_weights,
                )

            mature_estimator = None
            mature_mask = ~train_newborn
            if mature_mask.sum() >= 120 and train_data.loc[mature_mask, "target"].nunique() >= 2:
                mature_estimator = estimator_factory()
                mature_weights = build_training_sample_weights(
                    train_data.loc[mature_mask, "date"],
                    train_data.loc[mature_mask, "future_return"],
                    halflife_days=estimator_config.sample_weight_halflife_days,
                    amplitude_scale=estimator_config.sample_weight_amplitude_scale,
                )
                _fit_pipeline(
                    mature_estimator,
                    train_data.loc[mature_mask, list(feature_names)],
                    train_data.loc[mature_mask, "target"],
                    mature_weights,
                )

            valid_probability = fallback_estimator.predict_proba(valid_data[list(feature_names)])[:, 1]
            if newborn_estimator is not None and valid_newborn.any():
                valid_probability[valid_newborn.to_numpy()] = newborn_estimator.predict_proba(valid_data.loc[valid_newborn, list(feature_names)])[:, 1]
            if mature_estimator is not None and (~valid_newborn).any():
                valid_probability[(~valid_newborn).to_numpy()] = mature_estimator.predict_proba(valid_data.loc[~valid_newborn, list(feature_names)])[:, 1]

        valid_pred = (valid_probability >= 0.5).astype(int)
        confidence = pd.Series(valid_probability, index=valid_data.index).map(lambda value: max(value, 1.0 - value))
        high_conf_mask = confidence >= 0.6

        auc = pd.NA
        if valid_data["target"].nunique() > 1:
            from sklearn.metrics import roc_auc_score, accuracy_score, balanced_accuracy_score

            auc = roc_auc_score(valid_data["target"], valid_probability)
            high_conf_accuracy = accuracy_score(valid_data.loc[high_conf_mask, "target"], valid_pred[high_conf_mask]) if high_conf_mask.any() else pd.NA
            accuracy = accuracy_score(valid_data["target"], valid_pred)
            balanced_accuracy = balanced_accuracy_score(valid_data["target"], valid_pred)
        else:
            from sklearn.metrics import accuracy_score, balanced_accuracy_score

            high_conf_accuracy = accuracy_score(valid_data.loc[high_conf_mask, "target"], valid_pred[high_conf_mask]) if high_conf_mask.any() else pd.NA
            accuracy = accuracy_score(valid_data["target"], valid_pred)
            balanced_accuracy = balanced_accuracy_score(valid_data["target"], valid_pred)

        fold_results.append(
            FoldResult(
                train_end=pd.Timestamp(train_dates.max()),
                valid_start=pd.Timestamp(valid_dates.min()),
                valid_end=pd.Timestamp(valid_dates.max()),
                auc=float(auc) if auc is not pd.NA else float("nan"),
                accuracy=float(accuracy),
                balanced_accuracy=float(balanced_accuracy),
                high_conf_accuracy=float(high_conf_accuracy) if high_conf_accuracy is not pd.NA else float("nan"),
                coverage=float(high_conf_mask.mean()),
            )
        )

        fold_prediction = valid_data.loc[:, ["date", "symbol", "name", "target", "future_return"]].copy()
        fold_prediction["prob_up"] = valid_probability
        fold_prediction["pred"] = valid_pred
        fold_prediction["confidence"] = confidence.to_numpy(dtype=float)
        predictions.append(fold_prediction)

        split_start += valid_days

    if not fold_results:
        raise RuntimeError("模型对比未能生成任何有效折，请扩大日期范围或降低最小训练窗口。")

    metrics = pd.DataFrame([asdict(result) for result in fold_results])
    prediction_frame = pd.concat(predictions, ignore_index=True)
    return metrics, prediction_frame


def _resolve_decision_metric(selection_objective: str) -> str:
    if selection_objective == "accuracy":
        return "accuracy"
    if selection_objective == "joint":
        return "joint"
    return "balanced_accuracy"


def compare_models(
    preset_name: str,
    dataset_path: str,
    output_dir: str | Path = "artifacts/model_comparison",
) -> dict[str, Path | pd.DataFrame | dict[str, float]]:
    preset = get_training_preset(preset_name)
    config = TrainConfig(training_preset=preset_name, **preset["train_config"])
    current_estimator_config: EstimatorConfig = preset["estimator_config"]

    dataset = load_dataset(dataset_path)
    sci_factor_frame = build_sci_factor_frame(
        dataset,
        SciFactorConfig(refresh_incomplete_cache=config.refresh_incomplete_sci_cache),
    ) if config.enable_sci_factors else None
    feature_frame, feature_names = make_feature_frame(
        dataset,
        horizon=config.horizon,
        neutral_quantile=config.neutral_quantile,
        sci_factor_frame=sci_factor_frame,
        label_denoise_mode=config.label_denoise_mode,
        adaptive_neutral_strength=config.adaptive_neutral_strength,
        adaptive_neutral_min_quantile=config.adaptive_neutral_min_quantile,
        adaptive_neutral_max_quantile=config.adaptive_neutral_max_quantile,
    )

    summary_rows: list[dict[str, float | str]] = []
    all_fold_metrics: list[pd.DataFrame] = []
    all_predictions: list[pd.DataFrame] = []
    decision_metric = _resolve_decision_metric(config.selection_objective)

    for spec in get_benchmark_model_specs(current_estimator_config):
        metrics, predictions = _run_walk_forward_validation(
            feature_frame,
            feature_names=list(feature_names),
            min_train_days=config.min_train_days,
            valid_days=config.valid_days,
            gap_days=config.gap_days,
            estimator_factory=spec.estimator_factory,
            estimator_config=current_estimator_config,
            enable_layered_model=config.enable_layered_model,
            newborn_days=config.newborn_days,
        )
        threshold_config = calibrate_thresholds(
            predictions,
            min_signal_coverage=config.min_signal_coverage,
            target_signal_accuracy=config.target_signal_accuracy,
            decision_metric=decision_metric,
            decision_threshold_min=config.decision_threshold_min,
            decision_threshold_max=config.decision_threshold_max,
            decision_threshold_step=config.decision_threshold_step,
            confidence_threshold_min=config.confidence_threshold_min,
            confidence_threshold_max=config.confidence_threshold_max,
            confidence_threshold_step=config.confidence_threshold_step,
        )
        calibrated_summary = summarize_validation_predictions(
            predictions,
            decision_threshold=threshold_config.decision_threshold,
            confidence_threshold=threshold_config.confidence_threshold,
        )
        summary = summarize_cv_metrics(
            metrics,
            selection_objective=config.selection_objective,
            calibrated_summary=calibrated_summary,
        )

        summary_rows.append(
            {
                "model": spec.name,
                "label": spec.label,
                "preset": preset_name,
                "selection_objective": config.selection_objective,
                "decision_metric": threshold_config.decision_metric,
                "decision_threshold": threshold_config.decision_threshold,
                "confidence_threshold": threshold_config.confidence_threshold,
                **summary,
            }
        )

        fold_frame = metrics.copy()
        fold_frame.insert(0, "model", spec.name)
        fold_frame.insert(1, "label", spec.label)
        all_fold_metrics.append(fold_frame)

        prediction_frame = predictions.copy()
        prediction_frame.insert(0, "model", spec.name)
        prediction_frame.insert(1, "label", spec.label)
        all_predictions.append(prediction_frame)

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    preset_slug = preset_name.replace("-", "_")

    summary_frame = pd.DataFrame(summary_rows).sort_values("selection_score", ascending=False).reset_index(drop=True)
    fold_metrics_frame = pd.concat(all_fold_metrics, ignore_index=True)
    predictions_frame = pd.concat(all_predictions, ignore_index=True)

    summary_path = output_root / f"model_comparison_{preset_slug}_h{config.horizon}.csv"
    fold_metrics_path = output_root / f"model_comparison_folds_{preset_slug}_h{config.horizon}.csv"
    predictions_path = output_root / f"model_comparison_predictions_{preset_slug}_h{config.horizon}.csv"

    summary_frame.to_csv(summary_path, index=False, encoding="utf-8-sig")
    fold_metrics_frame.to_csv(fold_metrics_path, index=False, encoding="utf-8-sig")
    predictions_frame.to_csv(predictions_path, index=False, encoding="utf-8-sig")

    return {
        "summary": summary_frame,
        "fold_metrics": fold_metrics_frame,
        "predictions": predictions_frame,
        "summary_path": summary_path,
        "fold_metrics_path": fold_metrics_path,
        "predictions_path": predictions_path,
    }