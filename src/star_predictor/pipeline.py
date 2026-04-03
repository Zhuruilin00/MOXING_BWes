from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from .data import DownloadConfig, download_star_history, load_dataset, save_dataset
from .features import make_feature_frame
from .model import (
    EstimatorConfig,
    calibrate_thresholds,
    fit_final_estimator,
    fit_layered_final_estimators,
    predict_proba_with_layered_model,
    rolling_walk_forward_validation,
    rolling_walk_forward_validation_layered,
    summarize_cv_metrics,
    summarize_validation_predictions,
    threshold_config_to_dict,
)
from .sci_factors import SciFactorConfig, build_sci_factor_frame


@dataclass
class TrainConfig:
    start_date: str = "2020-01-01"
    end_date: str = "2026-04-01"
    horizon: int = 5
    min_history_days: int = 80
    neutral_quantile: float = 0.35
    label_denoise_mode: str = "fixed"
    adaptive_neutral_strength: float = 0.3
    adaptive_neutral_min_quantile: float = 0.25
    adaptive_neutral_max_quantile: float = 0.995
    min_train_days: int = 240
    valid_days: int = 40
    gap_days: int = 5
    universe_limit: int | None = None
    auto_tune: bool = True
    enable_sci_factors: bool = True
    refresh_incomplete_sci_cache: bool = True
    enable_layered_model: bool = True
    selection_objective: str = "accuracy"
    training_preset: str | None = None
    newborn_days: int = 720
    min_signal_coverage: float = 0.12
    target_signal_accuracy: float | None = None
    decision_threshold_min: float = 0.45
    decision_threshold_max: float = 0.56
    decision_threshold_step: float = 0.01
    confidence_threshold_min: float = 0.52
    confidence_threshold_max: float = 0.90
    confidence_threshold_step: float = 0.01


TRAINING_PRESETS: dict[str, dict[str, Any]] = {
    "accuracy-over65": {
        "train_config": {
            "horizon": 5,
            "neutral_quantile": 0.98,
            "min_train_days": 160,
            "valid_days": 20,
            "gap_days": 3,
            "auto_tune": False,
            "refresh_incomplete_sci_cache": False,
            "selection_objective": "accuracy",
        },
        "estimator_config": EstimatorConfig(
            n_estimators=220,
            max_depth=5,
            min_samples_split=140,
            min_samples_leaf=45,
            max_features=0.75,
            criterion="gini",
            class_weight=None,
            sample_weight_halflife_days=180,
            max_train_days=None,
        ),
    },
    "accuracy-balanced": {
        "train_config": {
            "horizon": 5,
            "neutral_quantile": 0.97,
            "min_train_days": 180,
            "valid_days": 20,
            "gap_days": 3,
            "auto_tune": False,
            "refresh_incomplete_sci_cache": False,
            "selection_objective": "joint",
        },
        "estimator_config": EstimatorConfig(
            n_estimators=260,
            max_depth=5,
            min_samples_split=130,
            min_samples_leaf=40,
            max_features=0.75,
            criterion="gini",
            class_weight=None,
            sample_weight_halflife_days=180,
            max_train_days=None,
        ),
    },
    "benchmark-best": {
        "train_config": {
            "horizon": 5,
            "neutral_quantile": 0.98,
            "min_train_days": 240,
            "valid_days": 40,
            "gap_days": 3,
            "auto_tune": False,
            "refresh_incomplete_sci_cache": False,
            "selection_objective": "joint",
        },
        "estimator_config": EstimatorConfig(
            n_estimators=460,
            max_depth=10,
            min_samples_split=45,
            min_samples_leaf=16,
            max_features=0.55,
            criterion="log_loss",
            class_weight="balanced",
            sample_weight_halflife_days=180,
            max_train_days=None,
        ),
    },
    "benchmark-blend": {
        "train_config": {
            "horizon": 5,
            "neutral_quantile": 0.98,
            "min_train_days": 240,
            "valid_days": 40,
            "gap_days": 3,
            "auto_tune": False,
            "refresh_incomplete_sci_cache": False,
            "selection_objective": "joint",
        },
        "estimator_config": EstimatorConfig(
            n_estimators=460,
            max_depth=10,
            min_samples_split=45,
            min_samples_leaf=16,
            max_features=0.55,
            criterion="log_loss",
            class_weight="balanced",
            use_rf_blend=True,
            rf_blend_weight=0.20,
            rf_n_estimators=320,
            rf_max_depth=10,
            rf_min_samples_split=45,
            rf_min_samples_leaf=16,
            rf_max_features=0.55,
            sample_weight_halflife_days=180,
            max_train_days=None,
        ),
    },
    "benchmark-thoughtful": {
        "train_config": {
            "horizon": 5,
            "neutral_quantile": 0.985,
            "min_train_days": 240,
            "valid_days": 40,
            "gap_days": 3,
            "auto_tune": False,
            "refresh_incomplete_sci_cache": False,
            "selection_objective": "joint",
            "decision_threshold_min": 0.40,
            "decision_threshold_max": 0.62,
            "decision_threshold_step": 0.01,
            "confidence_threshold_min": 0.58,
            "confidence_threshold_max": 0.95,
            "confidence_threshold_step": 0.01,
        },
        "estimator_config": EstimatorConfig(
            n_estimators=520,
            max_depth=10,
            min_samples_split=50,
            min_samples_leaf=18,
            max_features=0.48,
            criterion="log_loss",
            class_weight="balanced",
            use_rf_blend=False,
            sample_weight_halflife_days=240,
            max_train_days=260,
        ),
    },
    "balanced-over65": {
        "train_config": {
            "horizon": 5,
            "neutral_quantile": 0.98,
            "min_train_days": 240,
            "valid_days": 40,
            "gap_days": 3,
            "auto_tune": False,
            "refresh_incomplete_sci_cache": False,
            "selection_objective": "balanced_accuracy",
        },
        "estimator_config": EstimatorConfig(
            n_estimators=460,
            max_depth=10,
            min_samples_split=45,
            min_samples_leaf=16,
            max_features=0.55,
            criterion="log_loss",
            class_weight="balanced",
            sample_weight_halflife_days=180,
            max_train_days=None,
        ),
    },
    "paper-rald": {
        "train_config": {
            "horizon": 5,
            "neutral_quantile": 0.95,
            "label_denoise_mode": "adaptive",
            "adaptive_neutral_strength": 0.28,
            "adaptive_neutral_min_quantile": 0.80,
            "adaptive_neutral_max_quantile": 0.995,
            "min_train_days": 240,
            "valid_days": 40,
            "gap_days": 3,
            "auto_tune": False,
            "refresh_incomplete_sci_cache": False,
            "selection_objective": "joint",
        },
        "estimator_config": EstimatorConfig(
            n_estimators=460,
            max_depth=10,
            min_samples_split=45,
            min_samples_leaf=16,
            max_features=0.55,
            criterion="log_loss",
            class_weight="balanced",
            sample_weight_halflife_days=180,
            max_train_days=None,
        ),
    },
    "paper-rald-soft": {
        "train_config": {
            "horizon": 5,
            "neutral_quantile": 0.98,
            "label_denoise_mode": "adaptive",
            "adaptive_neutral_strength": 0.08,
            "adaptive_neutral_min_quantile": 0.94,
            "adaptive_neutral_max_quantile": 0.995,
            "min_train_days": 240,
            "valid_days": 40,
            "gap_days": 3,
            "auto_tune": False,
            "refresh_incomplete_sci_cache": False,
            "selection_objective": "joint",
        },
        "estimator_config": EstimatorConfig(
            n_estimators=460,
            max_depth=10,
            min_samples_split=45,
            min_samples_leaf=16,
            max_features=0.55,
            criterion="log_loss",
            class_weight="balanced",
            sample_weight_halflife_days=180,
            max_train_days=None,
        ),
    },
    "paper-rald-lite-a": {
        "train_config": {
            "horizon": 5,
            "neutral_quantile": 0.98,
            "label_denoise_mode": "adaptive",
            "adaptive_neutral_strength": 0.10,
            "adaptive_neutral_min_quantile": 0.90,
            "adaptive_neutral_max_quantile": 0.995,
            "min_train_days": 240,
            "valid_days": 40,
            "gap_days": 3,
            "auto_tune": False,
            "refresh_incomplete_sci_cache": False,
            "selection_objective": "joint",
        },
        "estimator_config": EstimatorConfig(
            n_estimators=460,
            max_depth=10,
            min_samples_split=45,
            min_samples_leaf=16,
            max_features=0.55,
            criterion="log_loss",
            class_weight="balanced",
            sample_weight_halflife_days=180,
            max_train_days=None,
        ),
    },
    "paper-rald-lite-b": {
        "train_config": {
            "horizon": 5,
            "neutral_quantile": 0.98,
            "label_denoise_mode": "adaptive",
            "adaptive_neutral_strength": 0.15,
            "adaptive_neutral_min_quantile": 0.92,
            "adaptive_neutral_max_quantile": 0.995,
            "min_train_days": 240,
            "valid_days": 40,
            "gap_days": 3,
            "auto_tune": False,
            "refresh_incomplete_sci_cache": False,
            "selection_objective": "joint",
        },
        "estimator_config": EstimatorConfig(
            n_estimators=460,
            max_depth=10,
            min_samples_split=45,
            min_samples_leaf=16,
            max_features=0.55,
            criterion="log_loss",
            class_weight="balanced",
            sample_weight_halflife_days=180,
            max_train_days=None,
        ),
    },
    "paper-rald-mid": {
        "train_config": {
            "horizon": 5,
            "neutral_quantile": 0.98,
            "label_denoise_mode": "adaptive",
            "adaptive_neutral_strength": 0.20,
            "adaptive_neutral_min_quantile": 0.90,
            "adaptive_neutral_max_quantile": 0.995,
            "min_train_days": 240,
            "valid_days": 40,
            "gap_days": 3,
            "auto_tune": False,
            "refresh_incomplete_sci_cache": False,
            "selection_objective": "joint",
        },
        "estimator_config": EstimatorConfig(
            n_estimators=460,
            max_depth=10,
            min_samples_split=45,
            min_samples_leaf=16,
            max_features=0.55,
            criterion="log_loss",
            class_weight="balanced",
            sample_weight_halflife_days=180,
            max_train_days=None,
        ),
    },
    "paper-rald-lite-c": {
        "train_config": {
            "horizon": 5,
            "neutral_quantile": 0.98,
            "label_denoise_mode": "adaptive",
            "adaptive_neutral_strength": 0.06,
            "adaptive_neutral_min_quantile": 0.95,
            "adaptive_neutral_max_quantile": 0.995,
            "min_train_days": 240,
            "valid_days": 40,
            "gap_days": 3,
            "auto_tune": False,
            "refresh_incomplete_sci_cache": False,
            "selection_objective": "joint",
        },
        "estimator_config": EstimatorConfig(
            n_estimators=460,
            max_depth=10,
            min_samples_split=45,
            min_samples_leaf=16,
            max_features=0.55,
            criterion="log_loss",
            class_weight="balanced",
            sample_weight_halflife_days=180,
            max_train_days=None,
        ),
    },
    "paper-rald-lite-d": {
        "train_config": {
            "horizon": 5,
            "neutral_quantile": 0.98,
            "label_denoise_mode": "adaptive",
            "adaptive_neutral_strength": 0.04,
            "adaptive_neutral_min_quantile": 0.96,
            "adaptive_neutral_max_quantile": 0.995,
            "min_train_days": 240,
            "valid_days": 40,
            "gap_days": 3,
            "auto_tune": False,
            "refresh_incomplete_sci_cache": False,
            "selection_objective": "joint",
        },
        "estimator_config": EstimatorConfig(
            n_estimators=460,
            max_depth=10,
            min_samples_split=45,
            min_samples_leaf=16,
            max_features=0.55,
            criterion="log_loss",
            class_weight="balanced",
            sample_weight_halflife_days=180,
            max_train_days=None,
        ),
    },
    "paper-rald-lite-e": {
        "train_config": {
            "horizon": 5,
            "neutral_quantile": 0.98,
            "label_denoise_mode": "adaptive",
            "adaptive_neutral_strength": 0.08,
            "adaptive_neutral_min_quantile": 0.96,
            "adaptive_neutral_max_quantile": 0.995,
            "min_train_days": 240,
            "valid_days": 40,
            "gap_days": 3,
            "auto_tune": False,
            "refresh_incomplete_sci_cache": False,
            "selection_objective": "joint",
        },
        "estimator_config": EstimatorConfig(
            n_estimators=460,
            max_depth=10,
            min_samples_split=45,
            min_samples_leaf=16,
            max_features=0.55,
            criterion="log_loss",
            class_weight="balanced",
            sample_weight_halflife_days=180,
            max_train_days=None,
        ),
    },
    "paper-rald-lite-f": {
        "train_config": {
            "horizon": 5,
            "neutral_quantile": 0.98,
            "label_denoise_mode": "adaptive",
            "adaptive_neutral_strength": 0.12,
            "adaptive_neutral_min_quantile": 0.94,
            "adaptive_neutral_max_quantile": 0.995,
            "min_train_days": 240,
            "valid_days": 40,
            "gap_days": 3,
            "auto_tune": False,
            "refresh_incomplete_sci_cache": False,
            "selection_objective": "joint",
        },
        "estimator_config": EstimatorConfig(
            n_estimators=460,
            max_depth=10,
            min_samples_split=45,
            min_samples_leaf=16,
            max_features=0.55,
            criterion="log_loss",
            class_weight="balanced",
            sample_weight_halflife_days=180,
            max_train_days=None,
        ),
    },
    "paper-rald-lite-g": {
        "train_config": {
            "horizon": 5,
            "neutral_quantile": 0.98,
            "label_denoise_mode": "adaptive",
            "adaptive_neutral_strength": 0.02,
            "adaptive_neutral_min_quantile": 0.97,
            "adaptive_neutral_max_quantile": 0.995,
            "min_train_days": 240,
            "valid_days": 40,
            "gap_days": 3,
            "auto_tune": False,
            "refresh_incomplete_sci_cache": False,
            "selection_objective": "joint",
        },
        "estimator_config": EstimatorConfig(
            n_estimators=460,
            max_depth=10,
            min_samples_split=45,
            min_samples_leaf=16,
            max_features=0.55,
            criterion="log_loss",
            class_weight="balanced",
            sample_weight_halflife_days=180,
            max_train_days=None,
        ),
    },
}


def get_training_preset_names() -> list[str]:
    return sorted(TRAINING_PRESETS)


def get_training_preset(name: str) -> dict[str, Any]:
    if name not in TRAINING_PRESETS:
        raise ValueError(f"Unknown training preset: {name}")
    return TRAINING_PRESETS[name]


def _candidate_search_space(config: TrainConfig) -> list[tuple[float, EstimatorConfig]]:
    if config.training_preset is not None:
        preset = get_training_preset(config.training_preset)
        return [(config.neutral_quantile, preset["estimator_config"])]

    if not config.auto_tune:
        return [(config.neutral_quantile, EstimatorConfig())]

    base_estimator_configs = [
        EstimatorConfig(n_estimators=220, max_depth=5, min_samples_split=140, min_samples_leaf=45, max_features=0.75, criterion="gini"),
        EstimatorConfig(n_estimators=260, max_depth=6, min_samples_split=120, min_samples_leaf=40, max_features=0.75, criterion="gini"),
        EstimatorConfig(n_estimators=300, max_depth=7, min_samples_split=100, min_samples_leaf=35, max_features=0.7, criterion="gini"),
        EstimatorConfig(n_estimators=280, max_depth=6, min_samples_split=110, min_samples_leaf=35, max_features=0.75, criterion="entropy"),
        EstimatorConfig(n_estimators=320, max_depth=7, min_samples_split=90, min_samples_leaf=30, max_features=0.7, criterion="log_loss"),
    ]

    if config.selection_objective == "accuracy":
        neutral_values = sorted({config.neutral_quantile, 0.45, 0.55, 0.65, 0.75, 0.85, 0.90, 0.92, 0.95})
        tuning_profiles = [
            {"class_weight": None, "sample_weight_halflife_days": 60, "max_train_days": 120},
            {"class_weight": None, "sample_weight_halflife_days": 120, "max_train_days": 180},
            {"class_weight": "balanced", "sample_weight_halflife_days": 120, "max_train_days": 180},
            {"class_weight": "balanced", "sample_weight_halflife_days": 180, "max_train_days": None},
        ]
    else:
        neutral_values = sorted({config.neutral_quantile, 0.25, 0.35, 0.45, 0.55, 0.75, 0.85, 0.90, 0.96, 0.98})
        tuning_profiles = [
            {"class_weight": "balanced", "sample_weight_halflife_days": 180, "max_train_days": None},
            {"class_weight": "balanced", "sample_weight_halflife_days": 120, "max_train_days": 180},
        ]

    candidates: list[tuple[float, EstimatorConfig]] = []
    for neutral in neutral_values:
        for base_config in base_estimator_configs:
            for profile in tuning_profiles:
                candidates.append(
                    (
                        neutral,
                        EstimatorConfig(
                            n_estimators=base_config.n_estimators,
                            max_depth=base_config.max_depth,
                            min_samples_split=base_config.min_samples_split,
                            min_samples_leaf=base_config.min_samples_leaf,
                            max_features=base_config.max_features,
                            criterion=base_config.criterion,
                            class_weight=profile["class_weight"],
                            sample_weight_halflife_days=profile["sample_weight_halflife_days"],
                            sample_weight_amplitude_scale=base_config.sample_weight_amplitude_scale,
                            max_train_days=profile["max_train_days"],
                        ),
                    )
                )
    return candidates


class StarMarketDirectionPredictor:
    def __init__(self, artifact_dir: str | Path = "artifacts") -> None:
        self.artifact_dir = Path(artifact_dir)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)

    def prepare_dataset(self, config: TrainConfig) -> pd.DataFrame:
        dataset = download_star_history(
            DownloadConfig(
                start_date=config.start_date,
                end_date=config.end_date,
                min_history_days=config.min_history_days,
            ),
            limit=config.universe_limit,
        )
        output_path = self.artifact_dir / "star_history.csv"
        save_dataset(dataset, str(output_path))
        return dataset

    def train(self, config: TrainConfig, dataset_path: str | None = None) -> dict[str, Path | pd.DataFrame]:
        if dataset_path is None:
            dataset = self.prepare_dataset(config)
        else:
            dataset = load_dataset(dataset_path)

        sci_factor_frame = build_sci_factor_frame(
            dataset,
            SciFactorConfig(refresh_incomplete_cache=config.refresh_incomplete_sci_cache),
        ) if config.enable_sci_factors else None

        best_run: dict[str, Any] | None = None
        for neutral_quantile, estimator_config in _candidate_search_space(config):
            feature_frame, feature_names = make_feature_frame(
                dataset,
                horizon=config.horizon,
                neutral_quantile=neutral_quantile,
                sci_factor_frame=sci_factor_frame,
                label_denoise_mode=config.label_denoise_mode,
                adaptive_neutral_strength=config.adaptive_neutral_strength,
                adaptive_neutral_min_quantile=config.adaptive_neutral_min_quantile,
                adaptive_neutral_max_quantile=config.adaptive_neutral_max_quantile,
            )
            if config.enable_layered_model:
                metrics, validation_predictions = rolling_walk_forward_validation_layered(
                    feature_frame,
                    feature_names=list(feature_names),
                    min_train_days=config.min_train_days,
                    valid_days=config.valid_days,
                    gap_days=config.gap_days,
                    estimator_config=estimator_config,
                    newborn_days=config.newborn_days,
                )
            else:
                metrics, validation_predictions = rolling_walk_forward_validation(
                    feature_frame,
                    feature_names=list(feature_names),
                    min_train_days=config.min_train_days,
                    valid_days=config.valid_days,
                    gap_days=config.gap_days,
                    estimator_config=estimator_config,
                )
            if config.selection_objective == "accuracy":
                decision_metric = "accuracy"
            elif config.selection_objective == "joint":
                decision_metric = "joint"
            else:
                decision_metric = "balanced_accuracy"
            threshold_config = calibrate_thresholds(
                validation_predictions,
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
                validation_predictions,
                decision_threshold=threshold_config.decision_threshold,
                confidence_threshold=threshold_config.confidence_threshold,
            )
            summary = summarize_cv_metrics(
                metrics,
                selection_objective=config.selection_objective,
                calibrated_summary=calibrated_summary,
            )
            run = {
                "neutral_quantile": neutral_quantile,
                "estimator_config": estimator_config,
                "feature_frame": feature_frame,
                "feature_names": list(feature_names),
                "metrics": metrics,
                "validation_predictions": validation_predictions,
                "threshold_config": threshold_config,
                "summary": summary,
            }
            if best_run is None or run["summary"]["selection_score"] > best_run["summary"]["selection_score"]:
                best_run = run

        if best_run is None:
            raise RuntimeError("未找到可用的训练配置。")

        feature_frame = best_run["feature_frame"]
        feature_names = best_run["feature_names"]
        metrics = best_run["metrics"]
        validation_predictions = best_run["validation_predictions"]
        selected_estimator_config = best_run["estimator_config"]
        threshold_config = best_run["threshold_config"]

        if config.enable_layered_model:
            layered_fit = fit_layered_final_estimators(
                feature_frame,
                feature_names,
                estimator_config=selected_estimator_config,
                newborn_days=config.newborn_days,
            )
            estimator = layered_fit["fallback_estimator"]
        else:
            estimator = fit_final_estimator(feature_frame, feature_names, selected_estimator_config)
            layered_fit = {
                "use_layered": False,
                "fallback_estimator": estimator,
                "newborn_estimator": None,
                "mature_estimator": None,
                "newborn_days": config.newborn_days,
                "listing_age_col": "listing_age_days",
            }

        model_bundle = {
            "config": {**asdict(config), "neutral_quantile": best_run["neutral_quantile"]},
            "feature_names": list(feature_names),
            "estimator": estimator,
            "use_layered": layered_fit["use_layered"],
            "fallback_estimator": layered_fit["fallback_estimator"],
            "newborn_estimator": layered_fit["newborn_estimator"],
            "mature_estimator": layered_fit["mature_estimator"],
            "newborn_days": layered_fit["newborn_days"],
            "listing_age_col": layered_fit["listing_age_col"],
            "estimator_config": asdict(selected_estimator_config),
            "threshold_config": threshold_config_to_dict(threshold_config),
            "cv_summary": best_run["summary"],
            "train_end_date": str(feature_frame["date"].max().date()),
        }

        model_path = self.artifact_dir / f"star_direction_h{config.horizon}.joblib"
        metrics_path = self.artifact_dir / f"cv_metrics_h{config.horizon}.csv"
        validation_path = self.artifact_dir / f"cv_predictions_h{config.horizon}.csv"

        joblib.dump(model_bundle, model_path)
        metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")
        validation_predictions.to_csv(validation_path, index=False, encoding="utf-8-sig")

        return {
            "dataset": dataset,
            "sci_factor_frame": sci_factor_frame if sci_factor_frame is not None else pd.DataFrame(),
            "feature_frame": feature_frame,
            "metrics": metrics,
            "validation_predictions": validation_predictions,
            "selected_estimator_config": selected_estimator_config,
            "selected_neutral_quantile": best_run["neutral_quantile"],
            "threshold_config": threshold_config,
            "cv_summary": best_run["summary"],
            "model_path": model_path,
            "metrics_path": metrics_path,
            "validation_path": validation_path,
        }

    def predict_latest(self, model_path: str | Path, dataset_path: str | None = None, top_k: int = 20) -> pd.DataFrame:
        bundle = joblib.load(model_path)
        config = TrainConfig(**bundle["config"])
        if dataset_path is None:
            dataset = self.prepare_dataset(config)
        else:
            dataset = load_dataset(dataset_path)

        sci_factor_frame = build_sci_factor_frame(
            dataset,
            SciFactorConfig(refresh_incomplete_cache=config.refresh_incomplete_sci_cache),
        ) if config.enable_sci_factors else None
        feature_frame, _ = make_feature_frame(
            dataset,
            horizon=config.horizon,
            neutral_quantile=config.neutral_quantile,
            sci_factor_frame=sci_factor_frame,
            label_denoise_mode=config.label_denoise_mode,
            adaptive_neutral_strength=config.adaptive_neutral_strength,
            adaptive_neutral_min_quantile=config.adaptive_neutral_min_quantile,
            adaptive_neutral_max_quantile=config.adaptive_neutral_max_quantile,
        )
        latest_date = feature_frame["date"].max()
        latest = feature_frame.loc[feature_frame["date"] == latest_date].copy()

        probabilities = predict_proba_with_layered_model(
            latest,
            bundle["feature_names"],
            bundle,
        )
        threshold_config = bundle.get("threshold_config", {"decision_threshold": 0.5, "confidence_threshold": 0.6})
        decision_threshold = float(threshold_config.get("decision_threshold", 0.5))
        confidence_threshold = float(threshold_config.get("confidence_threshold", 0.6))
        latest["prob_up"] = probabilities
        latest["pred_direction"] = latest["prob_up"].map(lambda value: "上涨" if value >= decision_threshold else "下跌")
        latest["confidence"] = latest["prob_up"].map(lambda value: max(value, 1.0 - value))
        latest = latest.loc[latest["confidence"] >= confidence_threshold].copy()

        if latest.empty:
            latest = feature_frame.loc[feature_frame["date"] == latest_date].copy()
            latest["prob_up"] = probabilities
            latest["pred_direction"] = latest["prob_up"].map(lambda value: "上涨" if value >= decision_threshold else "下跌")
            latest["confidence"] = latest["prob_up"].map(lambda value: max(value, 1.0 - value))

        result = latest.loc[:, ["date", "symbol", "name", "prob_up", "pred_direction", "confidence"]].copy()
        result = result.sort_values(["confidence", "prob_up"], ascending=[False, False]).head(top_k).reset_index(drop=True)
        output_path = self.artifact_dir / f"latest_predictions_h{config.horizon}.csv"
        result.to_csv(output_path, index=False, encoding="utf-8-sig")
        return result
