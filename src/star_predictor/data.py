from __future__ import annotations

from dataclasses import dataclass
import time

import akshare as ak
import numpy as np
import pandas as pd


STAR_PREFIXES = ("688", "689")


@dataclass(frozen=True)
class DownloadConfig:
    start_date: str
    end_date: str
    min_history_days: int = 80


def get_star_universe() -> pd.DataFrame:
    codes = ak.stock_info_a_code_name().rename(columns={"code": "代码", "name": "名称"})
    codes["代码"] = codes["代码"].astype(str).str.zfill(6)
    universe = codes.loc[codes["代码"].str.startswith(STAR_PREFIXES), ["代码", "名称"]].copy()
    universe = universe.drop_duplicates(subset=["代码"]).sort_values("代码").reset_index(drop=True)
    return universe


def _normalize_hist_frame(frame: pd.DataFrame, symbol: str, name: str) -> pd.DataFrame:
    renamed = frame.rename(columns={"日期": "date", "日期": "date", "date": "date", "开盘": "open", "open": "open", "收盘": "close", "close": "close", "最高": "high", "high": "high", "最低": "low", "low": "low", "成交量": "volume", "volume": "volume", "成交额": "amount", "amount": "amount", "振幅": "amplitude", "涨跌幅": "pct_change", "涨跌额": "change", "换手率": "turnover"})
    columns = [
        "date",
        "open",
        "close",
        "high",
        "low",
        "volume",
        "amount",
        "amplitude",
        "pct_change",
        "change",
        "turnover",
    ]

    if "amount" in renamed.columns and "volume" not in renamed.columns:
        renamed["volume"] = renamed["amount"]
    if "amount" not in renamed.columns and "volume" in renamed.columns:
        renamed["amount"] = renamed["volume"] * renamed.get("close", np.nan)

    normalized = renamed.reindex(columns=columns).copy()
    normalized["date"] = pd.to_datetime(normalized["date"])
    numeric_columns = [column for column in columns if column != "date"]
    normalized[numeric_columns] = normalized[numeric_columns].apply(pd.to_numeric, errors="coerce")
    normalized["amplitude"] = normalized["amplitude"].fillna(normalized["high"].sub(normalized["low"]).div(normalized["close"].replace(0, np.nan)))
    normalized["change"] = normalized["change"].fillna(normalized["close"].diff())
    normalized["pct_change"] = normalized["pct_change"].fillna(normalized["close"].pct_change())
    normalized["symbol"] = symbol
    normalized["name"] = name
    return normalized.dropna(subset=["open", "close", "high", "low"])


def _symbol_to_tx(symbol: str) -> str:
    return f"sh{symbol}" if symbol.startswith(("600", "601", "603", "605", "688", "689")) else f"sz{symbol}"


def _download_hist_with_retry(symbol: str, start_date: str, end_date: str, retries: int = 3) -> pd.DataFrame:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            return ak.stock_zh_a_hist(
                symbol=symbol,
                period="daily",
                start_date=start_date.replace("-", ""),
                end_date=end_date.replace("-", ""),
                adjust="qfq",
            )
        except Exception as error:
            last_error = error
            time.sleep(1.0 + attempt)

    for attempt in range(retries):
        try:
            return ak.stock_zh_a_hist_tx(
                symbol=_symbol_to_tx(symbol),
                start_date=start_date.replace("-", ""),
                end_date=end_date.replace("-", ""),
                adjust="qfq",
            )
        except Exception as error:
            last_error = error
            time.sleep(1.0 + attempt)

    if last_error is None:
        raise RuntimeError(f"未能下载 {symbol} 的历史行情")
    raise last_error


def download_star_history(config: DownloadConfig, limit: int | None = None) -> pd.DataFrame:
    universe = get_star_universe()
    if limit is not None:
        universe = universe.head(limit).copy()

    frames: list[pd.DataFrame] = []
    for row in universe.itertuples(index=False):
        try:
            history = _download_hist_with_retry(row.代码, config.start_date, config.end_date)
        except Exception:
            continue

        if history.empty or len(history) < config.min_history_days:
            continue

        frames.append(_normalize_hist_frame(history, row.代码, row.名称))

    if not frames:
        raise RuntimeError("未获取到可用的科创板历史数据，请检查网络或日期范围。")

    dataset = pd.concat(frames, ignore_index=True)
    dataset = dataset.sort_values(["date", "symbol"]).reset_index(drop=True)
    return dataset


def save_dataset(dataset: pd.DataFrame, output_path: str) -> None:
    dataset.to_csv(output_path, index=False, encoding="utf-8-sig")


def load_dataset(path: str) -> pd.DataFrame:
    dataset = pd.read_csv(path)
    dataset["date"] = pd.to_datetime(dataset["date"])
    if "symbol" in dataset.columns:
        dataset["symbol"] = dataset["symbol"].astype(str).str.zfill(6)
    return dataset.sort_values(["date", "symbol"]).reset_index(drop=True)
