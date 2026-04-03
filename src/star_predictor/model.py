from __future__ import annotations

from dataclasses import asdict, dataclass
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, ClassifierMixin


warnings.filterwarnings(
    "ignore",
    message="'penalty' was deprecated in version 1.8 and will be removed in 1.10.*",
    category=FutureWarning,
)


@dataclass
class FoldResult:
    train_end: pd.Timestamp
    valid_start: pd.Timestamp
    valid_end: pd.Timestamp
    auc: float
    accuracy: float
    balanced_accuracy: float
    high_conf_accuracy: float
    coverage: float


@dataclass(frozen=True)
class EstimatorConfig:
    n_estimators: int = 260
    max_depth: int = 6
    min_samples_split: int = 120
    min_samples_leaf: int = 40
    max_features: float = 0.75
    criterion: str = "gini"
    class_weight: str | None = "balanced"
    use_rf_blend: bool = False
    rf_blend_weight: float = 0.35
    rf_n_estimators: int = 320
    rf_max_depth: int | None = 10
    rf_min_samples_split: int = 45
    rf_min_samples_leaf: int = 16
    rf_max_features: float = 0.55
    sample_weight_halflife_days: int = 180
    sample_weight_amplitude_scale: float = 12.0
    max_train_days: int | None = None


class TreeBlendClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        extra_trees: ExtraTreesClassifier,
        random_forest: RandomForestClassifier,
        rf_blend_weight: float = 0.35,
    ) -> None:
        if not (0.0 <= rf_blend_weight <= 1.0):
            raise ValueError("rf_blend_weight must be in [0, 1]")
        self.extra_trees = extra_trees
        self.random_forest = random_forest
        self.rf_blend_weight = float(rf_blend_weight)

    def fit(self, x_train: pd.DataFrame, y_train: pd.Series, sample_weight: np.ndarray | None = None):
        self.extra_trees.fit(x_train, y_train, sample_weight=sample_weight)
        self.random_forest.fit(x_train, y_train, sample_weight=sample_weight)
        self.classes_ = np.array(sorted(pd.Series(y_train).dropna().unique()))
        self.n_features_in_ = x_train.shape[1]
        self.is_fitted_ = True
        return self

    def predict_proba(self, x_frame: pd.DataFrame) -> np.ndarray:
        extra_prob = self.extra_trees.predict_proba(x_frame)
        rf_prob = self.random_forest.predict_proba(x_frame)
        return (1.0 - self.rf_blend_weight) * extra_prob + self.rf_blend_weight * rf_prob

    def predict(self, x_frame: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(x_frame)[:, 1] >= 0.5).astype(int)


@dataclass(frozen=True)
class ThresholdConfig:
    decision_threshold: float = 0.5
    confidence_threshold: float = 0.6
    min_signal_coverage: float = 0.12
    signal_accuracy: float = 0.0
    signal_balanced_accuracy: float = 0.0
    decision_metric: str = "balanced_accuracy"


def build_estimator(config: EstimatorConfig | None = None) -> Pipeline:
    config = config or EstimatorConfig()
    extra_trees = ExtraTreesClassifier(
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        min_samples_split=config.min_samples_split,
        min_samples_leaf=config.min_samples_leaf,
        max_features=config.max_features,
        criterion=config.criterion,
        class_weight=config.class_weight,
        n_jobs=-1,
        random_state=42,
    )
    if config.use_rf_blend:
        random_forest = RandomForestClassifier(
            n_estimators=config.rf_n_estimators,
            max_depth=config.rf_max_depth,
            min_samples_split=config.rf_min_samples_split,
            min_samples_leaf=config.rf_min_samples_leaf,
            max_features=config.rf_max_features,
            class_weight="balanced_subsample" if config.class_weight else None,
            n_jobs=-1,
            random_state=42,
        )
        classifier = TreeBlendClassifier(
            extra_trees=extra_trees,
            random_forest=random_forest,
            rf_blend_weight=config.rf_blend_weight,
        )
    else:
        classifier = extra_trees

    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", classifier),
        ]
    )


def _recent_sample_weights(dates: pd.Series, halflife_days: int = 180) -> np.ndarray:
    age_days = (dates.max() - dates).dt.days.clip(lower=0)
    weights = np.power(0.5, age_days / halflife_days)
    return weights.to_numpy(dtype=float)


def build_training_sample_weights(
    dates: pd.Series,
    future_return: pd.Series | None = None,
    halflife_days: int = 180,
    amplitude_scale: float = 12.0,
    max_amplitude_boost: float = 2.5,
) -> np.ndarray:
    weights = _recent_sample_weights(dates, halflife_days=halflife_days)
    if future_return is None:
        return weights

    amplitude = np.clip(np.abs(pd.to_numeric(future_return, errors="coerce").fillna(0.0).to_numpy()) * amplitude_scale, 0.0, max_amplitude_boost)
    return weights * (1.0 + amplitude)


def _select_recent_training_data(train_data: pd.DataFrame, max_train_days: int | None) -> pd.DataFrame:
    if max_train_days is None:
        return train_data

    unique_dates = pd.Index(sorted(train_data["date"].unique()))
    if len(unique_dates) <= max_train_days:
        return train_data

    keep_dates = unique_dates[-max_train_days:]
    return train_data.loc[train_data["date"].isin(keep_dates)].copy()


def rolling_walk_forward_validation(
    frame: pd.DataFrame,
    feature_names: list[str],
    min_train_days: int,
    valid_days: int,
    gap_days: int,
    estimator_config: EstimatorConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    unique_dates = pd.Index(sorted(frame["date"].unique()))
    fold_results: list[FoldResult] = []
    predictions: list[pd.DataFrame] = []

    split_start = min_train_days
    while split_start + gap_days + valid_days <= len(unique_dates):
        train_end_idx = split_start - 1
        valid_start_idx = split_start + gap_days
        valid_end_idx = valid_start_idx + valid_days - 1

        train_dates = unique_dates[: train_end_idx + 1]
        valid_dates = unique_dates[valid_start_idx : valid_end_idx + 1]

        train_mask = frame["date"].isin(train_dates)
        valid_mask = frame["date"].isin(valid_dates)
        train_data = frame.loc[train_mask].copy()
        valid_data = frame.loc[valid_mask].copy()
        train_data = _select_recent_training_data(train_data, estimator_config.max_train_days if estimator_config is not None else None)

        if train_data["target"].nunique() < 2 or valid_data.empty:
            split_start += valid_days
            continue

        estimator = build_estimator(estimator_config)
        sample_weights = build_training_sample_weights(
            train_data["date"],
            train_data.get("future_return"),
            halflife_days=estimator_config.sample_weight_halflife_days if estimator_config is not None else 180,
            amplitude_scale=estimator_config.sample_weight_amplitude_scale if estimator_config is not None else 12.0,
        )
        estimator.fit(train_data[feature_names], train_data["target"], classifier__sample_weight=sample_weights)

        valid_probability = estimator.predict_proba(valid_data[feature_names])[:, 1]
        valid_pred = (valid_probability >= 0.5).astype(int)
        confidence = np.maximum(valid_probability, 1.0 - valid_probability)
        high_conf_mask = confidence >= 0.6

        auc = roc_auc_score(valid_data["target"], valid_probability) if valid_data["target"].nunique() > 1 else np.nan
        high_conf_accuracy = accuracy_score(valid_data.loc[high_conf_mask, "target"], valid_pred[high_conf_mask]) if high_conf_mask.any() else np.nan

        fold_results.append(
            FoldResult(
                train_end=pd.Timestamp(train_dates.max()),
                valid_start=pd.Timestamp(valid_dates.min()),
                valid_end=pd.Timestamp(valid_dates.max()),
                auc=auc,
                accuracy=accuracy_score(valid_data["target"], valid_pred),
                balanced_accuracy=balanced_accuracy_score(valid_data["target"], valid_pred),
                high_conf_accuracy=high_conf_accuracy,
                coverage=float(high_conf_mask.mean()),
            )
        )

        fold_prediction = valid_data.loc[:, ["date", "symbol", "name", "target", "future_return"]].copy()
        fold_prediction["prob_up"] = valid_probability
        fold_prediction["pred"] = valid_pred
        fold_prediction["confidence"] = confidence
        predictions.append(fold_prediction)

        split_start += valid_days

    if not fold_results:
        raise RuntimeError("滚动验证未能生成任何有效折，请扩大日期范围或降低最小训练窗口。")

    metrics = pd.DataFrame([result.__dict__ for result in fold_results])
    prediction_frame = pd.concat(predictions, ignore_index=True)
    return metrics, prediction_frame


def rolling_walk_forward_validation_layered(
    frame: pd.DataFrame,
    feature_names: list[str],
    min_train_days: int,
    valid_days: int,
    gap_days: int,
    estimator_config: EstimatorConfig | None = None,
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
        train_data = _select_recent_training_data(train_data, estimator_config.max_train_days if estimator_config is not None else None)
        if train_data.empty or valid_data.empty:
            split_start += valid_days
            continue

        if listing_age_col not in train_data.columns:
            train_data[listing_age_col] = np.nan
            valid_data[listing_age_col] = np.nan

        train_newborn = train_data[listing_age_col].le(newborn_days)
        valid_newborn = valid_data[listing_age_col].le(newborn_days)

        fallback_estimator = build_estimator(estimator_config)
        fallback_weights = build_training_sample_weights(
            train_data["date"],
            train_data.get("future_return"),
            halflife_days=estimator_config.sample_weight_halflife_days if estimator_config is not None else 180,
            amplitude_scale=estimator_config.sample_weight_amplitude_scale if estimator_config is not None else 12.0,
        )
        fallback_estimator.fit(train_data[feature_names], train_data["target"], classifier__sample_weight=fallback_weights)

        newborn_estimator = None
        if train_newborn.sum() >= 120 and train_data.loc[train_newborn, "target"].nunique() >= 2:
            newborn_estimator = build_estimator(estimator_config)
            newborn_weights = build_training_sample_weights(
                train_data.loc[train_newborn, "date"],
                train_data.loc[train_newborn, "future_return"],
                halflife_days=estimator_config.sample_weight_halflife_days if estimator_config is not None else 180,
                amplitude_scale=estimator_config.sample_weight_amplitude_scale if estimator_config is not None else 12.0,
            )
            newborn_estimator.fit(
                train_data.loc[train_newborn, feature_names],
                train_data.loc[train_newborn, "target"],
                classifier__sample_weight=newborn_weights,
            )

        mature_estimator = None
        mature_mask = ~train_newborn
        if mature_mask.sum() >= 120 and train_data.loc[mature_mask, "target"].nunique() >= 2:
            mature_estimator = build_estimator(estimator_config)
            mature_weights = build_training_sample_weights(
                train_data.loc[mature_mask, "date"],
                train_data.loc[mature_mask, "future_return"],
                halflife_days=estimator_config.sample_weight_halflife_days if estimator_config is not None else 180,
                amplitude_scale=estimator_config.sample_weight_amplitude_scale if estimator_config is not None else 12.0,
            )
            mature_estimator.fit(
                train_data.loc[mature_mask, feature_names],
                train_data.loc[mature_mask, "target"],
                classifier__sample_weight=mature_weights,
            )

        valid_probability = np.zeros(len(valid_data), dtype=float)
        valid_probability[:] = fallback_estimator.predict_proba(valid_data[feature_names])[:, 1]
        if newborn_estimator is not None and valid_newborn.any():
            valid_probability[valid_newborn.to_numpy()] = newborn_estimator.predict_proba(valid_data.loc[valid_newborn, feature_names])[:, 1]
        if mature_estimator is not None and (~valid_newborn).any():
            valid_probability[(~valid_newborn).to_numpy()] = mature_estimator.predict_proba(valid_data.loc[~valid_newborn, feature_names])[:, 1]

        valid_pred = (valid_probability >= 0.5).astype(int)
        confidence = np.maximum(valid_probability, 1.0 - valid_probability)
        high_conf_mask = confidence >= 0.6

        auc = roc_auc_score(valid_data["target"], valid_probability) if valid_data["target"].nunique() > 1 else np.nan
        high_conf_accuracy = accuracy_score(valid_data.loc[high_conf_mask, "target"], valid_pred[high_conf_mask]) if high_conf_mask.any() else np.nan
        fold_results.append(
            FoldResult(
                train_end=pd.Timestamp(train_dates.max()),
                valid_start=pd.Timestamp(valid_dates.min()),
                valid_end=pd.Timestamp(valid_dates.max()),
                auc=auc,
                accuracy=accuracy_score(valid_data["target"], valid_pred),
                balanced_accuracy=balanced_accuracy_score(valid_data["target"], valid_pred),
                high_conf_accuracy=high_conf_accuracy,
                coverage=float(high_conf_mask.mean()),
            )
        )

        fold_prediction = valid_data.loc[:, ["date", "symbol", "name", "target", "future_return"]].copy()
        fold_prediction["prob_up"] = valid_probability
        fold_prediction["pred"] = valid_pred
        fold_prediction["confidence"] = confidence
        predictions.append(fold_prediction)

        split_start += valid_days

    if not fold_results:
        raise RuntimeError("分层滚动验证未能生成有效折。")

    metrics = pd.DataFrame([result.__dict__ for result in fold_results])
    prediction_frame = pd.concat(predictions, ignore_index=True)
    return metrics, prediction_frame


def summarize_validation_predictions(
    validation_predictions: pd.DataFrame,
    decision_threshold: float,
    confidence_threshold: float,
) -> dict[str, float]:
    pred = (validation_predictions["prob_up"] >= decision_threshold).astype(int)
    confidence_mask = validation_predictions["confidence"] >= confidence_threshold
    calibrated_accuracy = accuracy_score(validation_predictions["target"], pred)
    calibrated_balanced_accuracy = balanced_accuracy_score(validation_predictions["target"], pred)
    signal_accuracy = accuracy_score(validation_predictions.loc[confidence_mask, "target"], pred.loc[confidence_mask]) if confidence_mask.any() else np.nan
    signal_balanced_accuracy = balanced_accuracy_score(validation_predictions.loc[confidence_mask, "target"], pred.loc[confidence_mask]) if confidence_mask.any() else np.nan
    return {
        "calibrated_accuracy": float(calibrated_accuracy),
        "calibrated_balanced_accuracy": float(calibrated_balanced_accuracy),
        "calibrated_high_conf_accuracy": float(signal_accuracy) if not np.isnan(signal_accuracy) else np.nan,
        "calibrated_high_conf_balanced_accuracy": float(signal_balanced_accuracy) if not np.isnan(signal_balanced_accuracy) else np.nan,
        "calibrated_coverage": float(confidence_mask.mean()),
    }


def summarize_cv_metrics(
    metrics: pd.DataFrame,
    selection_objective: str = "accuracy",
    calibrated_summary: dict[str, float] | None = None,
) -> dict[str, float]:
    summary = metrics[["auc", "accuracy", "balanced_accuracy", "high_conf_accuracy", "coverage"]].mean(numeric_only=True).to_dict()
    if calibrated_summary:
        summary.update(calibrated_summary)
    high_conf = summary.get("high_conf_accuracy")
    if np.isnan(high_conf):
        high_conf = summary.get("balanced_accuracy", 0.0)
    signal_quality = high_conf * max(summary.get("coverage", 0.0), 0.08)
    calibrated_accuracy = summary.get("calibrated_accuracy", summary.get("accuracy", 0.0))
    calibrated_balanced_accuracy = summary.get("calibrated_balanced_accuracy", summary.get("balanced_accuracy", 0.0))
    calibrated_high_conf = summary.get("calibrated_high_conf_accuracy", high_conf)
    accuracy_priority = (
        0.6 * calibrated_accuracy
        + 0.15 * calibrated_balanced_accuracy
        + 0.15 * summary.get("auc", 0.0)
        + 0.1 * calibrated_high_conf
    )
    balanced_priority = (
        0.45 * calibrated_balanced_accuracy
        + 0.25 * calibrated_accuracy
        + 0.2 * summary.get("auc", 0.0)
        + 0.1 * signal_quality
    )
    joint_priority = (
        0.45 * calibrated_accuracy
        + 0.45 * calibrated_balanced_accuracy
        + 0.1 * summary.get("auc", 0.0)
    )
    signal_priority = 0.5 * signal_quality + 0.3 * summary.get("balanced_accuracy", 0.0) + 0.2 * summary.get("auc", 0.0)
    objective_scores = {
        "signal_quality": signal_priority,
        "accuracy": accuracy_priority,
        "balanced_accuracy": balanced_priority,
        "joint": joint_priority,
    }
    if selection_objective not in objective_scores:
        raise ValueError(f"Unsupported selection objective: {selection_objective}")

    summary["selection_score"] = objective_scores[selection_objective]
    summary["signal_priority_score"] = signal_priority
    summary["accuracy_priority_score"] = accuracy_priority
    summary["balanced_priority_score"] = balanced_priority
    summary["joint_priority_score"] = joint_priority
    summary["signal_quality"] = signal_quality
    return {key: float(value) for key, value in summary.items()}


def calibrate_thresholds(
    validation_predictions: pd.DataFrame,
    min_signal_coverage: float = 0.12,
    target_signal_accuracy: float | None = None,
    decision_metric: str = "balanced_accuracy",
    decision_threshold_min: float = 0.45,
    decision_threshold_max: float = 0.56,
    decision_threshold_step: float = 0.01,
    confidence_threshold_min: float = 0.52,
    confidence_threshold_max: float = 0.90,
    confidence_threshold_step: float = 0.01,
) -> ThresholdConfig:
    if not (0.0 < decision_threshold_min < 1.0 and 0.0 < decision_threshold_max < 1.0):
        raise ValueError("decision threshold bounds must be in (0,1)")
    if decision_threshold_min > decision_threshold_max:
        raise ValueError("decision_threshold_min cannot be greater than decision_threshold_max")
    if decision_threshold_step <= 0:
        raise ValueError("decision_threshold_step must be positive")
    if not (0.0 < confidence_threshold_min < 1.0 and 0.0 < confidence_threshold_max < 1.0):
        raise ValueError("confidence threshold bounds must be in (0,1)")
    if confidence_threshold_min > confidence_threshold_max:
        raise ValueError("confidence_threshold_min cannot be greater than confidence_threshold_max")
    if confidence_threshold_step <= 0:
        raise ValueError("confidence_threshold_step must be positive")

    decision_candidates = np.arange(
        decision_threshold_min,
        decision_threshold_max + decision_threshold_step / 2.0,
        decision_threshold_step,
    )
    best_decision_threshold = 0.5
    best_decision_score = -np.inf

    if decision_metric not in {"accuracy", "balanced_accuracy", "joint"}:
        raise ValueError(f"Unsupported decision metric: {decision_metric}")

    for threshold in decision_candidates:
        pred = (validation_predictions["prob_up"] >= threshold).astype(int)
        if decision_metric == "accuracy":
            score = accuracy_score(validation_predictions["target"], pred)
        elif decision_metric == "joint":
            score = 0.5 * (
                accuracy_score(validation_predictions["target"], pred)
                + balanced_accuracy_score(validation_predictions["target"], pred)
            )
        else:
            score = balanced_accuracy_score(validation_predictions["target"], pred)
        if score > best_decision_score:
            best_decision_score = score
            best_decision_threshold = float(np.round(threshold, 2))

    confidence_candidates = np.arange(
        confidence_threshold_min,
        confidence_threshold_max + confidence_threshold_step / 2.0,
        confidence_threshold_step,
    )
    candidate_rows: list[dict[str, float]] = []

    for threshold in confidence_candidates:
        mask = validation_predictions["confidence"] >= threshold
        if not mask.any():
            continue

        coverage = float(mask.mean())
        pred = (validation_predictions.loc[mask, "prob_up"] >= best_decision_threshold).astype(int)
        accuracy = accuracy_score(validation_predictions.loc[mask, "target"], pred)
        balanced = balanced_accuracy_score(validation_predictions.loc[mask, "target"], pred)

        candidate_rows.append(
            {
                "threshold": float(np.round(threshold, 2)),
                "coverage": coverage,
                "accuracy": float(accuracy),
                "balanced_accuracy": float(balanced),
                "joint_score": float(0.5 * (accuracy + balanced)),
            }
        )

    if not candidate_rows:
        best_confidence_threshold = 0.6
        best_accuracy = 0.0
        best_balanced = 0.0
    else:
        candidates = pd.DataFrame(candidate_rows)
        feasible = candidates.loc[candidates["coverage"] >= min_signal_coverage].copy()
        if feasible.empty:
            if decision_metric == "joint":
                chosen = candidates.sort_values(["joint_score", "balanced_accuracy", "accuracy", "coverage"], ascending=[False, False, False, False]).iloc[0]
            else:
                chosen = candidates.sort_values(["coverage", "accuracy", "balanced_accuracy"], ascending=[False, False, False]).iloc[0]
        else:
            if decision_metric == "joint":
                chosen = feasible.sort_values(["joint_score", "balanced_accuracy", "accuracy", "coverage"], ascending=[False, False, False, False]).iloc[0]
            elif target_signal_accuracy is not None:
                hit = feasible.loc[feasible["accuracy"] >= target_signal_accuracy].copy()
                if not hit.empty:
                    chosen = hit.sort_values(["coverage", "accuracy", "balanced_accuracy"], ascending=[False, False, False]).iloc[0]
                else:
                    chosen = feasible.sort_values(["accuracy", "balanced_accuracy", "coverage"], ascending=[False, False, False]).iloc[0]
            else:
                chosen = feasible.sort_values(["accuracy", "balanced_accuracy", "coverage"], ascending=[False, False, False]).iloc[0]

            
        best_confidence_threshold = float(chosen["threshold"])
        best_accuracy = float(chosen["accuracy"])
        best_balanced = float(chosen["balanced_accuracy"])

    return ThresholdConfig(
        decision_threshold=best_decision_threshold,
        confidence_threshold=best_confidence_threshold,
        min_signal_coverage=min_signal_coverage,
        signal_accuracy=best_accuracy,
        signal_balanced_accuracy=best_balanced,
        decision_metric=decision_metric,
    )


def threshold_config_to_dict(config: ThresholdConfig) -> dict[str, float]:
    return asdict(config)


def fit_final_estimator(
    frame: pd.DataFrame,
    feature_names: list[str],
    estimator_config: EstimatorConfig | None = None,
) -> Pipeline:
    config = estimator_config or EstimatorConfig()
    train_data = _select_recent_training_data(frame, config.max_train_days)
    estimator = build_estimator(config)
    sample_weights = build_training_sample_weights(
        train_data["date"],
        train_data.get("future_return"),
        halflife_days=config.sample_weight_halflife_days,
        amplitude_scale=config.sample_weight_amplitude_scale,
    )
    estimator.fit(train_data[feature_names], train_data["target"], classifier__sample_weight=sample_weights)
    return estimator


def fit_layered_final_estimators(
    frame: pd.DataFrame,
    feature_names: list[str],
    estimator_config: EstimatorConfig | None = None,
    newborn_days: int = 720,
    listing_age_col: str = "listing_age_days",
) -> dict[str, object]:
    if listing_age_col not in frame.columns:
        return {
            "use_layered": False,
            "fallback_estimator": fit_final_estimator(frame, feature_names, estimator_config),
            "newborn_estimator": None,
            "mature_estimator": None,
            "newborn_days": newborn_days,
            "listing_age_col": listing_age_col,
        }

    frame_local = frame.copy()
    frame_local[listing_age_col] = pd.to_numeric(frame_local[listing_age_col], errors="coerce")
    newborn_mask = frame_local[listing_age_col].le(newborn_days)

    fallback_estimator = fit_final_estimator(frame_local, feature_names, estimator_config)

    newborn_estimator = None
    if newborn_mask.sum() >= 120 and frame_local.loc[newborn_mask, "target"].nunique() >= 2:
        newborn_estimator = fit_final_estimator(frame_local.loc[newborn_mask], feature_names, estimator_config)

    mature_estimator = None
    mature_mask = ~newborn_mask
    if mature_mask.sum() >= 120 and frame_local.loc[mature_mask, "target"].nunique() >= 2:
        mature_estimator = fit_final_estimator(frame_local.loc[mature_mask], feature_names, estimator_config)

    return {
        "use_layered": True,
        "fallback_estimator": fallback_estimator,
        "newborn_estimator": newborn_estimator,
        "mature_estimator": mature_estimator,
        "newborn_days": newborn_days,
        "listing_age_col": listing_age_col,
    }


def predict_proba_with_layered_model(
    frame: pd.DataFrame,
    feature_names: list[str],
    model_bundle: dict[str, object],
) -> np.ndarray:
    if not bool(model_bundle.get("use_layered", False)):
        estimator = model_bundle.get("estimator") or model_bundle["fallback_estimator"]
        return estimator.predict_proba(frame[feature_names])[:, 1]

    listing_age_col = str(model_bundle.get("listing_age_col", "listing_age_days"))
    newborn_days = int(model_bundle.get("newborn_days", 720))
    fallback_estimator = model_bundle["fallback_estimator"]
    newborn_estimator = model_bundle.get("newborn_estimator")
    mature_estimator = model_bundle.get("mature_estimator")

    working = frame.copy()
    if listing_age_col not in working.columns:
        working[listing_age_col] = np.nan
    working[listing_age_col] = pd.to_numeric(working[listing_age_col], errors="coerce")
    newborn_mask = working[listing_age_col].le(newborn_days)

    prob = np.zeros(len(working), dtype=float)
    prob[:] = fallback_estimator.predict_proba(working[feature_names])[:, 1]
    if newborn_estimator is not None and newborn_mask.any():
        prob[newborn_mask.to_numpy()] = newborn_estimator.predict_proba(working.loc[newborn_mask, feature_names])[:, 1]
    mature_mask = ~newborn_mask
    if mature_estimator is not None and mature_mask.any():
        prob[mature_mask.to_numpy()] = mature_estimator.predict_proba(working.loc[mature_mask, feature_names])[:, 1]
    return prob
