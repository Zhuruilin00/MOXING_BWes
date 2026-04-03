from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


BASE_FEATURES = [
    "ret_1",
    "ret_2",
    "ret_3",
    "ret_5",
    "ret_10",
    "ret_15",
    "ret_20",
    "overnight_gap",
    "intraday_ret",
    "range_pct",
    "body_to_range",
    "turnover_ratio_5",
    "turnover_ratio_10",
    "turnover_ratio_20",
    "volume_ratio_5",
    "volume_ratio_10",
    "amount_ratio_5",
    "volatility_5",
    "volatility_10",
    "drawdown_20",
    "distance_to_ma5",
    "distance_to_ma10",
    "distance_to_ma20",
    "trend_strength_5",
    "trend_strength_10",
    "up_days_5",
    "up_days_10",
]

BOARD_FEATURES = [
    "board_ret_median_1",
    "board_ret_mean_5",
    "board_breadth_up",
    "board_dispersion_1",
    "board_turnover_median",
    "board_volatility_median_5",
]

SCI_FEATURES = [
    "listing_age_days",
    "rd_expense",
    "rd_expense_ratio",
    "unlock_days_to_next",
    "unlock_next_ratio",
    "research_report_count_180d",
    "patent_proxy_score",
    "main_fund_net_ratio",
    "super_large_net_ratio",
    "big_order_net_ratio",
    "announcement_count_30d",
    "announcement_risk_score_90d",
    "announcement_positive_score_90d",
    "minute_intraday_volatility",
    "minute_tail_return",
    "minute_range_ratio",
    "minute_tail_volume_ratio",
]

VALID_LABEL_DENOISE_MODES = {"fixed", "adaptive"}


def _safe_rank(series: pd.Series) -> pd.Series:
    if series.notna().sum() <= 1:
        return pd.Series(0.0, index=series.index)
    return series.rank(pct=True).sub(0.5)


def _rolling_share_positive(series: pd.Series, window: int) -> pd.Series:
    return series.gt(0).rolling(window).mean()


def _normalize_series(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    finite = values.replace([np.inf, -np.inf], np.nan)
    if finite.notna().sum() <= 1:
        return pd.Series(0.5, index=series.index)
    min_value = float(finite.min())
    max_value = float(finite.max())
    if np.isclose(max_value, min_value):
        return pd.Series(0.5, index=series.index)
    scaled = (finite - min_value) / (max_value - min_value)
    return scaled.fillna(0.5)


def _build_adaptive_neutral_schedule(
    board_frame: pd.DataFrame,
    base_quantile: float,
    adaptive_strength: float,
    min_quantile: float,
    max_quantile: float,
) -> pd.DataFrame:
    if board_frame.empty:
        return pd.DataFrame(columns=["date", "adaptive_neutral_quantile", "label_noise_score"])

    working = board_frame.loc[:, ["date", "board_volatility_median_5", "board_dispersion_1", "board_breadth_up"]].copy()
    working["board_breadth_up"] = pd.to_numeric(working["board_breadth_up"], errors="coerce")
    working["breadth_indecision"] = 1.0 - (working["board_breadth_up"].sub(0.5).abs() * 2.0)
    working["breadth_indecision"] = working["breadth_indecision"].clip(lower=0.0, upper=1.0)

    vol_component = _normalize_series(working["board_volatility_median_5"])
    dispersion_component = _normalize_series(working["board_dispersion_1"])
    indecision_component = _normalize_series(working["breadth_indecision"])
    working["label_noise_score"] = pd.concat(
        [vol_component, dispersion_component, indecision_component],
        axis=1,
    ).mean(axis=1)

    centered_noise = working["label_noise_score"].sub(0.5)
    working["adaptive_neutral_quantile"] = (
        base_quantile + adaptive_strength * centered_noise
    ).clip(lower=min_quantile, upper=max_quantile)

    return working.loc[:, ["date", "adaptive_neutral_quantile", "label_noise_score"]]


def make_feature_frame(
    dataset: pd.DataFrame,
    horizon: int,
    neutral_quantile: float = 0.35,
    min_feature_coverage: float = 0.2,
    sci_factor_frame: pd.DataFrame | None = None,
    label_denoise_mode: str = "fixed",
    adaptive_neutral_strength: float = 0.3,
    adaptive_neutral_min_quantile: float = 0.25,
    adaptive_neutral_max_quantile: float = 0.995,
) -> tuple[pd.DataFrame, Sequence[str]]:
    if label_denoise_mode not in VALID_LABEL_DENOISE_MODES:
        raise ValueError(
            f"Unsupported label_denoise_mode: {label_denoise_mode}. "
            f"Expected one of {sorted(VALID_LABEL_DENOISE_MODES)}"
        )

    if not (0.0 < adaptive_neutral_min_quantile <= adaptive_neutral_max_quantile <= 0.999):
        raise ValueError("adaptive neutral quantile bounds must satisfy 0 < min <= max <= 0.999")

    frame = dataset.copy()
    frame["symbol"] = frame["symbol"].astype(str).str.zfill(6)
    frame = frame.sort_values(["symbol", "date"]).reset_index(drop=True)

    by_symbol = frame.groupby("symbol", group_keys=False)

    frame["ret_1"] = by_symbol["close"].pct_change(1)
    frame["ret_2"] = by_symbol["close"].pct_change(2)
    frame["ret_3"] = by_symbol["close"].pct_change(3)
    frame["ret_5"] = by_symbol["close"].pct_change(5)
    frame["ret_10"] = by_symbol["close"].pct_change(10)
    frame["ret_15"] = by_symbol["close"].pct_change(15)
    frame["ret_20"] = by_symbol["close"].pct_change(20)
    frame["overnight_gap"] = frame["open"].div(by_symbol["close"].shift(1)).sub(1.0)
    frame["intraday_ret"] = frame["close"].div(frame["open"]).sub(1.0)
    frame["range_pct"] = frame["high"].sub(frame["low"]).div(frame["close"].replace(0, np.nan))
    frame["body_to_range"] = frame["close"].sub(frame["open"]).div(frame["high"].sub(frame["low"]).replace(0, np.nan))
    frame["volume_ratio_5"] = frame["volume"].div(by_symbol["volume"].transform(lambda series: series.rolling(5).mean()))
    frame["volume_ratio_10"] = frame["volume"].div(by_symbol["volume"].transform(lambda series: series.rolling(10).mean()))
    frame["amount_ratio_5"] = frame["amount"].div(by_symbol["amount"].transform(lambda series: series.rolling(5).mean()))
    frame["turnover_ratio_5"] = frame["turnover"].div(by_symbol["turnover"].transform(lambda series: series.rolling(5).mean()))
    frame["turnover_ratio_10"] = frame["turnover"].div(by_symbol["turnover"].transform(lambda series: series.rolling(10).mean()))
    frame["turnover_ratio_20"] = frame["turnover"].div(by_symbol["turnover"].transform(lambda series: series.rolling(20).mean()))
    frame["volatility_5"] = by_symbol["ret_1"].transform(lambda series: series.rolling(5).std())
    frame["volatility_10"] = by_symbol["ret_1"].transform(lambda series: series.rolling(10).std())
    frame["drawdown_20"] = frame["close"].div(by_symbol["close"].transform(lambda series: series.rolling(20).max())).sub(1.0)
    frame["distance_to_ma5"] = frame["close"].div(by_symbol["close"].transform(lambda series: series.rolling(5).mean())).sub(1.0)
    frame["distance_to_ma10"] = frame["close"].div(by_symbol["close"].transform(lambda series: series.rolling(10).mean())).sub(1.0)
    frame["distance_to_ma20"] = frame["close"].div(by_symbol["close"].transform(lambda series: series.rolling(20).mean())).sub(1.0)
    frame["trend_strength_5"] = by_symbol["ret_1"].transform(lambda series: series.rolling(5).sum())
    frame["trend_strength_10"] = by_symbol["ret_1"].transform(lambda series: series.rolling(10).sum())
    frame["up_days_5"] = by_symbol["ret_1"].transform(lambda series: _rolling_share_positive(series, 5))
    frame["up_days_10"] = by_symbol["ret_1"].transform(lambda series: _rolling_share_positive(series, 10))

    frame["future_return"] = by_symbol["close"].shift(-horizon).div(frame["close"]).sub(1.0)

    board = frame.groupby("date").agg(
        board_ret_median_1=("ret_1", "median"),
        board_ret_mean_5=("ret_5", "mean"),
        board_breadth_up=("ret_1", lambda series: float((series > 0).mean())),
        board_dispersion_1=("ret_1", "std"),
        board_turnover_median=("turnover", "median"),
        board_volatility_median_5=("volatility_5", "median"),
    )
    frame = frame.merge(board, on="date", how="left")

    adaptive_schedule = _build_adaptive_neutral_schedule(
        board.reset_index(drop=False),
        base_quantile=neutral_quantile,
        adaptive_strength=adaptive_neutral_strength,
        min_quantile=adaptive_neutral_min_quantile,
        max_quantile=adaptive_neutral_max_quantile,
    )
    frame = frame.merge(adaptive_schedule, on="date", how="left")

    if sci_factor_frame is not None and not sci_factor_frame.empty:
        sf = sci_factor_frame.copy()
        sf["date"] = pd.to_datetime(sf["date"])
        sf["symbol"] = sf["symbol"].astype(str).str.zfill(6)
        frame = frame.merge(sf, on=["date", "symbol"], how="left")

    rank_features = []
    for feature in BASE_FEATURES:
        rank_name = f"{feature}_rank"
        frame[rank_name] = frame.groupby("date", group_keys=False)[feature].transform(_safe_rank)
        rank_features.append(rank_name)

    board_feature_names = BOARD_FEATURES.copy()
    sci_feature_names = [feature for feature in SCI_FEATURES if feature in frame.columns]
    candidate_features = rank_features + board_feature_names + sci_feature_names
    model_features = [feature for feature in candidate_features if frame[feature].notna().mean() >= min_feature_coverage]

    if len(model_features) < 8:
        raise RuntimeError("有效特征数量不足，请检查行情字段覆盖率或调整数据源。")

    frame["applied_neutral_quantile"] = float(neutral_quantile)
    if label_denoise_mode == "adaptive":
        frame["applied_neutral_quantile"] = frame["adaptive_neutral_quantile"].fillna(float(neutral_quantile))
    frame["applied_neutral_quantile"] = frame["applied_neutral_quantile"].clip(
        lower=adaptive_neutral_min_quantile,
        upper=adaptive_neutral_max_quantile,
    )

    def _neutral_barrier_for_date(group: pd.DataFrame) -> pd.Series:
        valid = group["future_return"].dropna()
        if valid.size < 8:
            return pd.Series(0.0, index=group.index)

        quantile = float(group["applied_neutral_quantile"].iloc[0])
        quantile = float(np.clip(quantile, adaptive_neutral_min_quantile, adaptive_neutral_max_quantile))
        barrier = float(valid.abs().quantile(quantile))
        return pd.Series(barrier, index=group.index)

    neutral_barrier = frame.groupby("date", group_keys=False).apply(_neutral_barrier_for_date)
    frame["neutral_barrier"] = neutral_barrier
    frame["target"] = np.where(frame["future_return"] > 0, 1, 0)
    frame["is_informative_label"] = frame["future_return"].abs().ge(frame["neutral_barrier"])

    feature_frame = frame.dropna(subset=model_features + ["future_return"]).copy()
    feature_frame = feature_frame.loc[feature_frame["is_informative_label"]].copy()
    feature_frame = feature_frame.sort_values(["date", "symbol"]).reset_index(drop=True)
    return feature_frame, model_features
