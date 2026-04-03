from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import akshare as ak
import numpy as np
import pandas as pd


PATENT_KEYWORDS = ("专利", "知识产权", "发明")
RISK_EVENT_KEYWORDS = ("减持", "解禁", "诉讼", "问询", "处罚", "立案")
POS_EVENT_KEYWORDS = ("中标", "订单", "回购", "增持", "业绩快报", "业绩预增")

CACHE_REFRESH_REQUIRED_COLUMNS = [
    "listing_age_days",
    "rd_expense",
    "rd_expense_ratio",
    "research_report_count_180d",
    "patent_proxy_score",
    "main_fund_net_ratio",
    "super_large_net_ratio",
    "big_order_net_ratio",
    "announcement_count_30d",
    "announcement_risk_score_90d",
    "announcement_positive_score_90d",
]


@dataclass(frozen=True)
class SciFactorConfig:
    min_listing_days: int = 0
    cache_path: str | None = "artifacts/sci_factor_cache.csv"
    enable_minute_microstructure: bool = True
    minute_lookback_days: int = 180
    refresh_incomplete_cache: bool = True


def _symbol_to_market_code(symbol: str) -> str:
    if symbol.startswith(("600", "601", "603", "605", "688", "689")):
        return f"sh{symbol}"
    return f"sz{symbol}"


def _to_float(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _symbols_with_incomplete_cache(
    cache_frame: pd.DataFrame,
    required_keys: pd.DataFrame,
) -> set[str]:
    if cache_frame.empty:
        return set()

    missing_columns = [column for column in CACHE_REFRESH_REQUIRED_COLUMNS if column not in cache_frame.columns]
    if missing_columns:
        return set(required_keys["symbol"].astype(str).unique().tolist())

    merged = required_keys.merge(
        cache_frame[["date", "symbol", *CACHE_REFRESH_REQUIRED_COLUMNS]],
        on=["date", "symbol"],
        how="left",
    )
    coverage = (
        merged.groupby("symbol")[CACHE_REFRESH_REQUIRED_COLUMNS]
        .agg(lambda series: float(pd.to_numeric(series, errors="coerce").notna().mean()))
    )

    incomplete = coverage.loc[coverage["listing_age_days"] <= 0.05]
    return set(incomplete.index.astype(str).tolist())


def _extract_listing_date(symbol: str) -> pd.Timestamp | pd.NaT:
    try:
        info = ak.stock_individual_info_em(symbol=symbol)
    except Exception:
        return pd.NaT

    if info.empty:
        return pd.NaT

    listing_rows = info.loc[info["item"].astype(str) == "上市时间", "value"]
    if listing_rows.empty:
        return pd.NaT
    return pd.to_datetime(str(listing_rows.iloc[0]), format="%Y%m%d", errors="coerce")


def _build_rd_daily(symbol: str, trading_dates: pd.Series) -> pd.DataFrame:
    try:
        report = ak.stock_financial_report_sina(stock=_symbol_to_market_code(symbol), symbol="利润表")
    except Exception:
        return pd.DataFrame({"date": trading_dates, "symbol": symbol, "rd_expense": np.nan, "rd_expense_ratio": np.nan})

    if report.empty or "报告日" not in report.columns:
        return pd.DataFrame({"date": trading_dates, "symbol": symbol, "rd_expense": np.nan, "rd_expense_ratio": np.nan})

    work = report.copy()
    work["报告日"] = pd.to_datetime(work["报告日"], errors="coerce")
    work["研发费用"] = _to_float(work.get("研发费用", pd.Series(dtype=float)))
    work["营业收入"] = _to_float(work.get("营业收入", pd.Series(dtype=float)))
    work["rd_expense"] = work["研发费用"]
    work["rd_expense_ratio"] = work["研发费用"].div(work["营业收入"].replace(0, np.nan))
    work = work[["报告日", "rd_expense", "rd_expense_ratio"]].dropna(subset=["报告日"]).sort_values("报告日")

    target = pd.DataFrame({"date": pd.to_datetime(trading_dates).sort_values().unique()})
    merged = pd.merge_asof(target, work.rename(columns={"报告日": "date"}), on="date", direction="backward")
    merged["symbol"] = symbol
    return merged


def _build_unlock_daily(symbol: str, trading_dates: pd.Series) -> pd.DataFrame:
    base = pd.DataFrame({"date": pd.to_datetime(trading_dates).sort_values().unique()})
    base["symbol"] = symbol
    base["unlock_days_to_next"] = np.nan
    base["unlock_next_ratio"] = np.nan

    try:
        unlock = ak.stock_restricted_release_queue_em(symbol=symbol)
    except Exception:
        return base

    if unlock.empty:
        return base

    unlock = unlock.copy()
    unlock["解禁时间"] = pd.to_datetime(unlock["解禁时间"], errors="coerce")
    unlock["占总市值比例"] = _to_float(unlock.get("占总市值比例", pd.Series(dtype=float)))
    unlock = unlock.dropna(subset=["解禁时间"]).sort_values("解禁时间")
    if unlock.empty:
        return base

    event_dates = unlock["解禁时间"].to_numpy()
    event_ratio = unlock["占总市值比例"].to_numpy()
    dates = base["date"].to_numpy()
    idx = np.searchsorted(event_dates, dates, side="left")
    valid = idx < len(event_dates)
    base.loc[valid, "unlock_days_to_next"] = (event_dates[idx[valid]] - dates[valid]).astype("timedelta64[D]").astype(float)
    base.loc[valid, "unlock_next_ratio"] = event_ratio[idx[valid]]
    return base


def _build_research_proxy_daily(symbol: str, trading_dates: pd.Series) -> pd.DataFrame:
    base = pd.DataFrame({"date": pd.to_datetime(trading_dates).sort_values().unique()})
    base["symbol"] = symbol
    base["research_report_count_180d"] = 0.0
    base["patent_proxy_score"] = 0.0

    try:
        reports = ak.stock_research_report_em(symbol=symbol)
    except Exception:
        return base

    if reports.empty or "日期" not in reports.columns:
        return base

    reports = reports.copy()
    reports["日期"] = pd.to_datetime(reports["日期"], errors="coerce")
    reports = reports.dropna(subset=["日期"]).sort_values("日期")
    if reports.empty:
        return base

    reports["is_patent_keyword"] = reports["报告名称"].astype(str).apply(lambda text: float(any(keyword in text for keyword in PATENT_KEYWORDS)))
    count_daily = reports.groupby("日期").size().rename("cnt")
    patent_daily = reports.groupby("日期")["is_patent_keyword"].sum().rename("patent_cnt")
    daily = pd.concat([count_daily, patent_daily], axis=1).reset_index().rename(columns={"日期": "date"})
    daily = daily.sort_values("date")

    merged = base.merge(daily, on="date", how="left").fillna({"cnt": 0.0, "patent_cnt": 0.0})
    merged["research_report_count_180d"] = merged["cnt"].rolling(180, min_periods=1).sum()
    merged["patent_proxy_score"] = merged["patent_cnt"].rolling(360, min_periods=1).sum()
    return merged[["date", "symbol", "research_report_count_180d", "patent_proxy_score"]]


def _build_fund_flow_daily(symbol: str, trading_dates: pd.Series) -> pd.DataFrame:
    base = pd.DataFrame({"date": pd.to_datetime(trading_dates).sort_values().unique()})
    base["symbol"] = symbol

    market = "sh" if symbol.startswith(("600", "601", "603", "605", "688", "689")) else "sz"
    try:
        flow = ak.stock_individual_fund_flow(stock=symbol, market=market)
    except Exception:
        return base

    if flow.empty:
        return base

    flow = flow.copy()
    flow["日期"] = pd.to_datetime(flow["日期"], errors="coerce")
    flow["主力净流入-净占比"] = _to_float(flow.get("主力净流入-净占比", pd.Series(dtype=float)))
    flow["超大单净流入-净占比"] = _to_float(flow.get("超大单净流入-净占比", pd.Series(dtype=float)))
    flow["大单净流入-净占比"] = _to_float(flow.get("大单净流入-净占比", pd.Series(dtype=float)))
    flow = flow.dropna(subset=["日期"]).sort_values("日期")

    merged = base.merge(
        flow[["日期", "主力净流入-净占比", "超大单净流入-净占比", "大单净流入-净占比"]].rename(
            columns={
                "日期": "date",
                "主力净流入-净占比": "main_fund_net_ratio",
                "超大单净流入-净占比": "super_large_net_ratio",
                "大单净流入-净占比": "big_order_net_ratio",
            }
        ),
        on="date",
        how="left",
    )

    for column in ["main_fund_net_ratio", "super_large_net_ratio", "big_order_net_ratio"]:
        if column not in merged.columns:
            merged[column] = np.nan

    merged["main_fund_net_ratio"] = merged["main_fund_net_ratio"].ffill().fillna(0.0)
    merged["super_large_net_ratio"] = merged["super_large_net_ratio"].ffill().fillna(0.0)
    merged["big_order_net_ratio"] = merged["big_order_net_ratio"].ffill().fillna(0.0)
    return merged[["date", "symbol", "main_fund_net_ratio", "super_large_net_ratio", "big_order_net_ratio"]]


def _build_disclosure_event_daily(symbol: str, trading_dates: pd.Series) -> pd.DataFrame:
    base = pd.DataFrame({"date": pd.to_datetime(trading_dates).sort_values().unique()})
    base["symbol"] = symbol

    start_date = pd.to_datetime(trading_dates).min().strftime("%Y%m%d")
    end_date = pd.to_datetime(trading_dates).max().strftime("%Y%m%d")
    try:
        notice = ak.stock_zh_a_disclosure_report_cninfo(
            symbol=symbol,
            market="沪深京",
            start_date=start_date,
            end_date=end_date,
        )
    except Exception:
        return base

    if notice.empty:
        return base

    notice = notice.copy()
    notice["公告时间"] = pd.to_datetime(notice["公告时间"], errors="coerce")
    notice = notice.dropna(subset=["公告时间"]).sort_values("公告时间")
    if notice.empty:
        return base

    titles = notice["公告标题"].astype(str)
    notice["risk_score"] = titles.apply(lambda text: float(sum(keyword in text for keyword in RISK_EVENT_KEYWORDS)))
    notice["positive_score"] = titles.apply(lambda text: float(sum(keyword in text for keyword in POS_EVENT_KEYWORDS)))

    daily = notice.groupby("公告时间").agg(cnt=("公告标题", "size"), risk=("risk_score", "sum"), pos=("positive_score", "sum")).reset_index().rename(columns={"公告时间": "date"})
    merged = base.merge(daily, on="date", how="left").fillna({"cnt": 0.0, "risk": 0.0, "pos": 0.0})

    merged["announcement_count_30d"] = merged["cnt"].rolling(30, min_periods=1).sum()
    merged["announcement_risk_score_90d"] = merged["risk"].rolling(90, min_periods=1).sum()
    merged["announcement_positive_score_90d"] = merged["pos"].rolling(90, min_periods=1).sum()
    return merged[["date", "symbol", "announcement_count_30d", "announcement_risk_score_90d", "announcement_positive_score_90d"]]


def _build_minute_micro_daily(symbol: str, trading_dates: pd.Series, lookback_days: int) -> pd.DataFrame:
    base = pd.DataFrame({"date": pd.to_datetime(trading_dates).sort_values().unique()})
    base["symbol"] = symbol

    if base.empty:
        return base

    end_date = base["date"].max()
    start_date = max(base["date"].min(), end_date - pd.Timedelta(days=lookback_days))

    try:
        minute = ak.stock_zh_a_hist_min_em(
            symbol=symbol,
            start_date=f"{start_date.strftime('%Y-%m-%d')} 09:30:00",
            end_date=f"{end_date.strftime('%Y-%m-%d')} 15:00:00",
            period="1",
            adjust="",
        )
    except Exception:
        minute = pd.DataFrame()

    if minute.empty:
        base["minute_intraday_volatility"] = 0.0
        base["minute_tail_return"] = 0.0
        base["minute_range_ratio"] = 0.0
        base["minute_tail_volume_ratio"] = 0.0
        return base

    m = minute.rename(columns={"时间": "datetime", "日期": "datetime", "开盘": "open", "收盘": "close", "最高": "high", "最低": "low", "成交量": "volume"}).copy()
    if "datetime" not in m.columns:
        m["datetime"] = pd.to_datetime(m.iloc[:, 0], errors="coerce")
    else:
        m["datetime"] = pd.to_datetime(m["datetime"], errors="coerce")

    for col in ["open", "close", "high", "low", "volume"]:
        if col in m.columns:
            m[col] = _to_float(m[col])
        else:
            m[col] = np.nan

    m = m.dropna(subset=["datetime", "close"])
    if m.empty:
        base["minute_intraday_volatility"] = 0.0
        base["minute_tail_return"] = 0.0
        base["minute_range_ratio"] = 0.0
        base["minute_tail_volume_ratio"] = 0.0
        return base

    m["date"] = m["datetime"].dt.floor("D")
    m = m.sort_values(["date", "datetime"]) 
    m["ret_1m"] = m.groupby("date")["close"].pct_change()

    tail_cut = pd.to_datetime("14:30:00").time()
    m["is_tail"] = m["datetime"].dt.time >= tail_cut

    def _daily_agg(group: pd.DataFrame) -> pd.Series:
        last_close = group["close"].iloc[-1]
        high = group["high"].max()
        low = group["low"].min()
        intraday_vol = group["ret_1m"].std()
        tail = group.loc[group["is_tail"]]
        if tail.empty:
            tail_ret = 0.0
            tail_vol_ratio = 0.0
        else:
            tail_ret = tail["close"].iloc[-1] / max(tail["close"].iloc[0], 1e-9) - 1.0
            total_vol = float(group["volume"].sum()) if group["volume"].notna().any() else 0.0
            tail_vol = float(tail["volume"].sum()) if tail["volume"].notna().any() else 0.0
            tail_vol_ratio = tail_vol / total_vol if total_vol > 0 else 0.0
        range_ratio = (high - low) / last_close if pd.notna(last_close) and last_close != 0 else 0.0
        return pd.Series(
            {
                "minute_intraday_volatility": 0.0 if pd.isna(intraday_vol) else float(intraday_vol),
                "minute_tail_return": float(tail_ret),
                "minute_range_ratio": float(range_ratio),
                "minute_tail_volume_ratio": float(tail_vol_ratio),
            }
        )

    day_feat = m.groupby("date", as_index=False).apply(_daily_agg, include_groups=False)
    merged = base.merge(day_feat, on="date", how="left")
    for col in ["minute_intraday_volatility", "minute_tail_return", "minute_range_ratio", "minute_tail_volume_ratio"]:
        merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0.0)
    return merged


def build_sci_factor_frame(dataset: pd.DataFrame, config: SciFactorConfig | None = None) -> pd.DataFrame:
    config = config or SciFactorConfig()
    frame = dataset[["date", "symbol"]].copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame["symbol"] = frame["symbol"].astype(str).str.zfill(6)

    cache_frame = pd.DataFrame()
    if config.cache_path:
        cache_file = Path(config.cache_path)
        if cache_file.exists():
            try:
                cache_frame = pd.read_csv(cache_file)
                cache_frame["date"] = pd.to_datetime(cache_frame["date"])
                cache_frame["symbol"] = cache_frame["symbol"].astype(str).str.zfill(6)
            except Exception:
                cache_frame = pd.DataFrame()

    required_keys = frame[["date", "symbol"]].drop_duplicates()
    cached_keys = cache_frame[["date", "symbol"]].drop_duplicates() if not cache_frame.empty else pd.DataFrame(columns=["date", "symbol"])
    missing_keys = required_keys.merge(cached_keys, on=["date", "symbol"], how="left", indicator=True)
    missing_keys = missing_keys.loc[missing_keys["_merge"] == "left_only", ["date", "symbol"]]
    incomplete_symbols = _symbols_with_incomplete_cache(cache_frame, required_keys) if config.refresh_incomplete_cache else set()
    missing_symbols = sorted(set(missing_keys["symbol"].unique().tolist()) | incomplete_symbols)

    factor_frames: list[pd.DataFrame] = []
    for symbol, sub in frame.groupby("symbol"):
        if symbol not in missing_symbols:
            continue
        trading_dates = sub["date"].sort_values().drop_duplicates()

        listing_date = _extract_listing_date(symbol)
        listing_daily = pd.DataFrame({"date": trading_dates, "symbol": symbol})
        if pd.isna(listing_date):
            listing_daily["listing_age_days"] = np.nan
        else:
            listing_daily["listing_age_days"] = (listing_daily["date"] - listing_date).dt.days.astype(float)
            listing_daily.loc[listing_daily["listing_age_days"] < config.min_listing_days, "listing_age_days"] = np.nan

        rd_daily = _build_rd_daily(symbol, trading_dates)
        unlock_daily = _build_unlock_daily(symbol, trading_dates)
        research_daily = _build_research_proxy_daily(symbol, trading_dates)
        fund_flow_daily = _build_fund_flow_daily(symbol, trading_dates)
        notice_daily = _build_disclosure_event_daily(symbol, trading_dates)
        minute_daily = _build_minute_micro_daily(symbol, trading_dates, config.minute_lookback_days) if config.enable_minute_microstructure else pd.DataFrame({"date": trading_dates, "symbol": symbol})

        merged = listing_daily.merge(rd_daily, on=["date", "symbol"], how="left")
        merged = merged.merge(unlock_daily, on=["date", "symbol"], how="left")
        merged = merged.merge(research_daily, on=["date", "symbol"], how="left")
        merged = merged.merge(fund_flow_daily, on=["date", "symbol"], how="left")
        merged = merged.merge(notice_daily, on=["date", "symbol"], how="left")
        merged = merged.merge(minute_daily, on=["date", "symbol"], how="left")
        factor_frames.append(merged)

    fetched_frame = pd.concat(factor_frames, ignore_index=True) if factor_frames else pd.DataFrame()
    if not fetched_frame.empty:
        if cache_frame.empty:
            cache_frame = fetched_frame.copy()
        else:
            cache_frame = pd.concat([cache_frame, fetched_frame], ignore_index=True)
            cache_frame = cache_frame.drop_duplicates(subset=["date", "symbol"], keep="last")

        if config.cache_path:
            cache_file = Path(config.cache_path)
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_frame.to_csv(cache_file, index=False, encoding="utf-8-sig")

    if cache_frame.empty:
        raise RuntimeError("未能构建科创扩展因子，请检查网络和数据接口。")

    factor_frame = required_keys.merge(cache_frame, on=["date", "symbol"], how="left")
    factor_frame = factor_frame.sort_values(["date", "symbol"]).reset_index(drop=True)
    return factor_frame
