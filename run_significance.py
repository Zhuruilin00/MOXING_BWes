from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="基于逐折指标计算配对显著性检验")
    parser.add_argument("--ablation-folds", required=True, help="run_ablation.py 产出的 ablation_folds.csv")
    parser.add_argument("--benchmark-folds", required=False, help="compare_models 产出的逐折文件")
    parser.add_argument("--output", default="artifacts/model_comparison/ablations/significance_table.csv")
    return parser


def _safe_wilcoxon(x: pd.Series, y: pd.Series) -> tuple[str, float]:
    xx = pd.to_numeric(x, errors="coerce")
    yy = pd.to_numeric(y, errors="coerce")
    paired = pd.DataFrame({"x": xx, "y": yy}).dropna()
    if paired.empty:
        return "none", float("nan")

    d = paired["x"] - paired["y"]
    d = d.loc[d != 0]
    if len(d) < 6:
        return _sign_test(d)

    try:
        from scipy.stats import wilcoxon

        res = wilcoxon(paired["x"], paired["y"], zero_method="wilcox", correction=False, alternative="two-sided", mode="auto")
        return "wilcoxon", float(res.pvalue)
    except Exception:
        return _sign_test(d)


def _sign_test(d: pd.Series) -> tuple[str, float]:
    d = d.dropna()
    n = len(d)
    if n == 0:
        return "sign_test", 1.0
    pos = int((d > 0).sum())
    k = min(pos, n - pos)
    cdf = 0.0
    for i in range(0, k + 1):
        cdf += math.comb(n, i) * (0.5 ** n)
    p = min(1.0, 2.0 * cdf)
    return "sign_test", float(p)


def _paired_metric_table(df: pd.DataFrame, metric_col: str) -> pd.DataFrame:
    cols = ["valid_end", metric_col]
    out = df.loc[:, cols].copy()
    out[metric_col] = pd.to_numeric(out[metric_col], errors="coerce")
    out["valid_end"] = pd.to_datetime(out["valid_end"], errors="coerce")
    return out.dropna(subset=["valid_end", metric_col])


def _pair_test(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    left_name: str,
    right_name: str,
    metric: str,
) -> dict[str, object]:
    l = _paired_metric_table(left_df, metric_col=metric).rename(columns={metric: "left"})
    r = _paired_metric_table(right_df, metric_col=metric).rename(columns={metric: "right"})

    merged = l.merge(r, on=["valid_end"], how="inner")
    if merged.empty:
        # 当两组因去噪筛样导致日期不完全重叠时，退化为按折序位置配对
        l_seq = l.sort_values("valid_end").reset_index(drop=True)
        r_seq = r.sort_values("valid_end").reset_index(drop=True)
        n = min(len(l_seq), len(r_seq))
        if n > 0:
            merged = pd.DataFrame(
                {
                    "valid_end": l_seq.loc[: n - 1, "valid_end"].to_numpy(),
                    "left": l_seq.loc[: n - 1, "left"].to_numpy(),
                    "right": r_seq.loc[: n - 1, "right"].to_numpy(),
                }
            )
    method, p_value = _safe_wilcoxon(merged["left"], merged["right"])

    return {
        "left": left_name,
        "right": right_name,
        "metric": metric,
        "test": method,
        "n_pairs": int(len(merged)),
        "left_mean": float(merged["left"].mean()) if len(merged) else float("nan"),
        "right_mean": float(merged["right"].mean()) if len(merged) else float("nan"),
        "mean_diff": float((merged["left"] - merged["right"]).mean()) if len(merged) else float("nan"),
        "p_value": p_value,
        "significant_0p05": bool(p_value < 0.05) if pd.notna(p_value) else False,
    }


def main() -> None:
    args = build_parser().parse_args()

    ablation = pd.read_csv(args.ablation_folds)
    required = {"variant", "valid_end", "accuracy", "balanced_accuracy", "auc"}
    miss = required - set(ablation.columns)
    if miss:
        raise RuntimeError(f"ablation folds 缺少字段: {sorted(miss)}")

    rows: list[dict[str, object]] = []
    full = ablation.loc[ablation["variant"] == "full_method"].copy()
    if full.empty:
        raise RuntimeError("ablation_folds 中未找到 full_method")

    for ab in ["ablate_denoise_strength", "ablate_time_decay", "ablate_recent_window", "ablate_threshold_calibration"]:
        part = ablation.loc[ablation["variant"] == ab].copy()
        if part.empty:
            continue
        for metric in ["accuracy", "balanced_accuracy", "auc"]:
            rows.append(
                _pair_test(
                    left_df=full,
                    right_df=part,
                    left_name="full_method",
                    right_name=ab,
                    metric=metric,
                )
            )

    if args.benchmark_folds:
        bench = pd.read_csv(args.benchmark_folds)
        b_required = {"model", "valid_end", "accuracy", "balanced_accuracy", "auc"}
        b_miss = b_required - set(bench.columns)
        if b_miss:
            raise RuntimeError(f"benchmark folds 缺少字段: {sorted(b_miss)}")

        current = bench.loc[bench["model"] == "current_extra_trees"].copy()
        for model_name in ["random_forest", "hist_gradient_boosting", "logistic_regression"]:
            part = bench.loc[bench["model"] == model_name].copy()
            if part.empty:
                continue
            for metric in ["accuracy", "balanced_accuracy", "auc"]:
                rows.append(
                    _pair_test(
                        left_df=current,
                        right_df=part,
                        left_name="current_extra_trees",
                        right_name=model_name,
                        metric=metric,
                    )
                )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    table = pd.DataFrame(rows)
    table.to_csv(output, index=False, encoding="utf-8-sig")

    print("显著性检验完成")
    print(f"输出文件: {output}")
    if not table.empty:
        print(table.loc[:, ["left", "right", "metric", "test", "n_pairs", "mean_diff", "p_value", "significant_0p05"]].round(6).to_string(index=False))


if __name__ == "__main__":
    main()
