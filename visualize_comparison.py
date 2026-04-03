from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="将模型对比结果可视化为图表")
    parser.add_argument("--summary-csv", required=True, help="model_comparison_*.csv 路径")
    parser.add_argument("--folds-csv", required=False, help="model_comparison_folds_*.csv 路径")
    parser.add_argument("--output-dir", default="artifacts/model_comparison/charts")
    parser.add_argument("--title", default="Model Comparison")
    parser.add_argument("--no-notes", action="store_true", help="不生成图表说明 markdown")
    parser.add_argument("--notes-language", choices=["zh", "en", "bilingual"], default="bilingual", help="图表说明语言")
    parser.add_argument("--no-paper-captions", action="store_true", help="不生成论文图注模板 markdown")
    return parser


def _sorted_summary(summary: pd.DataFrame) -> pd.DataFrame:
    cols = ["model", "label", "selection_score", "accuracy", "balanced_accuracy", "auc", "coverage", "decision_threshold", "confidence_threshold"]
    missing = [c for c in cols if c not in summary.columns]
    if missing:
        raise ValueError(f"summary csv 缺少字段: {missing}")
    return summary.loc[:, cols].sort_values("selection_score", ascending=False).reset_index(drop=True)


def _top_model_line(summary: pd.DataFrame) -> str:
    top = summary.iloc[0]
    return (
        f"当前第一名: {top['label']} | "
        f"selection_score={float(top['selection_score']):.4f}, "
        f"accuracy={float(top['accuracy']):.4f}, "
        f"balanced_accuracy={float(top['balanced_accuracy']):.4f}, "
        f"auc={float(top['auc']):.4f}"
    )


def _top_model_line_en(summary: pd.DataFrame) -> str:
    top = summary.iloc[0]
    return (
        f"Top model: {top['label']} | "
        f"selection_score={float(top['selection_score']):.4f}, "
        f"accuracy={float(top['accuracy']):.4f}, "
        f"balanced_accuracy={float(top['balanced_accuracy']):.4f}, "
        f"auc={float(top['auc']):.4f}"
    )


def _add_explanation_box(ax: plt.Axes, text: str) -> None:
    ax.text(
        0.01,
        0.02,
        text,
        transform=ax.transAxes,
        fontsize=9,
        va="bottom",
        ha="left",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.85, "edgecolor": "#bbbbbb"},
    )


def plot_main_metrics(summary: pd.DataFrame, output_dir: Path, title: str) -> Path:
    metrics = ["accuracy", "balanced_accuracy", "auc", "selection_score"]
    labels = summary["label"].tolist()
    x = range(len(labels))

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    axes_list = [axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]]

    for ax, metric in zip(axes_list, metrics):
        vals = summary[metric].astype(float).tolist()
        bars = ax.bar(x, vals)
        ax.set_title(metric)
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, rotation=15, ha="right")
        ax.set_ylim(0.0, 1.0)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01, f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    _add_explanation_box(
        axes_list[0],
        "说明: 先看 selection_score，再结合 accuracy 与 balanced_accuracy。\n"
        "若分数接近，优先选择 auc 更高且后续折线更平稳的模型。",
    )

    fig.suptitle(f"{title}: Main Metrics", fontsize=14)
    output_path = output_dir / "comparison_main_metrics.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_thresholds_and_coverage(summary: pd.DataFrame, output_dir: Path, title: str) -> Path:
    labels = summary["label"].tolist()
    x = range(len(labels))

    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    bar = ax.bar(x, summary["coverage"].astype(float), label="coverage", alpha=0.75)
    line1, = ax.plot(x, summary["decision_threshold"].astype(float), marker="o", linewidth=2, label="decision_threshold")
    line2, = ax.plot(x, summary["confidence_threshold"].astype(float), marker="s", linewidth=2, label="confidence_threshold")

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("value")
    ax.set_title(f"{title}: Thresholds & Coverage")
    ax.legend(handles=[bar, line1, line2])
    _add_explanation_box(
        ax,
        "说明: coverage 反映高置信信号占比；\n"
        "confidence_threshold 越高通常覆盖率越低但信号更纯。\n"
        "实盘中建议在可接受覆盖率下优先看高置信准确率。",
    )

    output_path = output_dir / "comparison_thresholds_and_coverage.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_fold_accuracy(folds: pd.DataFrame, output_dir: Path, title: str) -> Path:
    required = ["label", "accuracy", "valid_end"]
    missing = [c for c in required if c not in folds.columns]
    if missing:
        raise ValueError(f"folds csv 缺少字段: {missing}")

    folds = folds.copy()
    folds["valid_end"] = pd.to_datetime(folds["valid_end"])
    folds = folds.sort_values(["label", "valid_end"]) 

    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    for label, part in folds.groupby("label"):
        ax.plot(part["valid_end"], part["accuracy"], marker="o", linewidth=1.5, label=label)

    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("fold accuracy")
    ax.set_title(f"{title}: Fold Accuracy Trend")
    ax.legend()
    ax.grid(alpha=0.2)
    _add_explanation_box(
        ax,
        "说明: 曲线越平滑，时序稳定性越好。\n"
        "若均值接近，优先选择回撤更小且近期折表现不失速的模型。",
    )

    output_path = output_dir / "comparison_fold_accuracy_trend.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def write_chart_notes(
    summary: pd.DataFrame,
    output_dir: Path,
    title: str,
    generated_files: list[Path],
    folds: pd.DataFrame | None = None,
    language: str = "bilingual",
) -> Path:
    notes_path = output_dir / "comparison_chart_notes.md"
    lines: list[str] = []

    by_label = None
    if folds is not None:
        by_label = (
            folds.groupby("label")["accuracy"]
            .agg(["mean", "std"]) 
            .reset_index()
            .rename(columns={"mean": "fold_accuracy_mean", "std": "fold_accuracy_std"})
            .sort_values(["fold_accuracy_mean", "fold_accuracy_std"], ascending=[False, True])
        )

    if language in {"zh", "bilingual"}:
        lines.append(f"# {title} 图表说明")
        lines.append("")
        lines.append("## 总览")
        lines.append("")
        lines.append(_top_model_line(summary))
        lines.append("")
        lines.append("建议阅读顺序: 先看主指标图，再看阈值与覆盖率，最后看逐折趋势。")
        lines.append("")

        lines.append("## comparison_main_metrics.png")
        lines.append("")
        lines.append("图意: 展示各模型在 accuracy、balanced_accuracy、auc、selection_score 四项核心指标上的整体对比。")
        lines.append("解读重点: selection_score 用于综合排名；accuracy 代表方向命中；balanced_accuracy 反映涨跌两类识别均衡性；auc 反映概率排序质量。")
        lines.append("决策建议: 若 selection_score 差距在 0.01 以内，优先选 balanced_accuracy 和 auc 同时更高者。")
        lines.append("")

        lines.append("## comparison_thresholds_and_coverage.png")
        lines.append("")
        lines.append("图意: 对比各模型的 decision_threshold、confidence_threshold 与高置信 coverage。")
        lines.append("解读重点: confidence_threshold 越高，coverage 常会下降；coverage 过低会导致可交易信号不足。")
        lines.append("决策建议: 在 coverage 可接受的前提下，优先能保持更高 calibrated_high_conf_accuracy 的模型。")
        lines.append("")

        if by_label is not None:
            lines.append("## comparison_fold_accuracy_trend.png")
            lines.append("")
            lines.append("图意: 展示每个模型在各时间折上的 accuracy 变化轨迹，用于观察时序稳定性。")
            lines.append("解读重点: 均值高不代表稳健，需同时观察波动幅度与最近折是否失速。")
            lines.append("各模型逐折统计(按均值降序、波动升序参考):")
            lines.append("")
            lines.append("| label | fold_accuracy_mean | fold_accuracy_std |")
            lines.append("|---|---:|---:|")
            for _, row in by_label.iterrows():
                lines.append(
                    f"| {row['label']} | {float(row['fold_accuracy_mean']):.4f} | {float(row['fold_accuracy_std']):.4f} |"
                )
            lines.append("")

    if language == "bilingual":
        lines.append("---")
        lines.append("")

    if language in {"en", "bilingual"}:
        lines.append(f"# {title} Chart Notes")
        lines.append("")
        lines.append("## Overview")
        lines.append("")
        lines.append(_top_model_line_en(summary))
        lines.append("")
        lines.append("Suggested reading order: main metrics first, then thresholds/coverage, then fold trend.")
        lines.append("")

        lines.append("## comparison_main_metrics.png")
        lines.append("")
        lines.append("Purpose: Compare overall performance across accuracy, balanced_accuracy, auc, and selection_score.")
        lines.append("Interpretation: selection_score is the integrated rank; accuracy measures directional hit-rate; balanced_accuracy reflects class balance; auc captures probability ranking quality.")
        lines.append("Decision rule: if selection_score gap is within 0.01, prefer the model with both higher balanced_accuracy and auc.")
        lines.append("")

        lines.append("## comparison_thresholds_and_coverage.png")
        lines.append("")
        lines.append("Purpose: Compare decision_threshold, confidence_threshold, and high-confidence coverage.")
        lines.append("Interpretation: higher confidence_threshold often lowers coverage; too low coverage may lead to insufficient tradable signals.")
        lines.append("Decision rule: under acceptable coverage, prioritize higher calibrated_high_conf_accuracy.")
        lines.append("")

        if by_label is not None:
            lines.append("## comparison_fold_accuracy_trend.png")
            lines.append("")
            lines.append("Purpose: Show per-fold accuracy trajectories to evaluate temporal stability.")
            lines.append("Interpretation: higher mean alone is insufficient; inspect volatility and late-fold deterioration.")
            lines.append("Per-model fold statistics (sorted by higher mean and lower std):")
            lines.append("")
            lines.append("| label | fold_accuracy_mean | fold_accuracy_std |")
            lines.append("|---|---:|---:|")
            for _, row in by_label.iterrows():
                lines.append(
                    f"| {row['label']} | {float(row['fold_accuracy_mean']):.4f} | {float(row['fold_accuracy_std']):.4f} |"
                )
            lines.append("")

    lines.append("## Generated Files")
    lines.append("")
    for path in generated_files:
        lines.append(f"- {path.name}")

    notes_path.write_text("\n".join(lines), encoding="utf-8")
    return notes_path


def write_paper_captions(
    summary: pd.DataFrame,
    output_dir: Path,
    title: str,
    has_folds: bool,
    language: str = "bilingual",
) -> Path:
    path = output_dir / "paper_figure_captions.md"
    top = summary.iloc[0]
    lines: list[str] = []

    if language in {"zh", "bilingual"}:
        lines.append(f"# {title} 论文图注模板")
        lines.append("")
        lines.append("图1（comparison_main_metrics.png）图注建议:")
        lines.append(
            "在相同滚动验证与阈值校准条件下，各模型在 accuracy、balanced_accuracy、auc 与 selection_score 上的对比结果。"
            f"当前排名第一模型为 {top['label']} (selection_score={float(top['selection_score']):.4f})。"
        )
        lines.append("")
        lines.append("图2（comparison_thresholds_and_coverage.png）图注建议:")
        lines.append(
            "各模型的决策阈值、置信阈值与高置信覆盖率对比。该图用于分析信号纯度与可交易样本规模之间的权衡关系。"
        )
        lines.append("")
        if has_folds:
            lines.append("图3（comparison_fold_accuracy_trend.png）图注建议:")
            lines.append("各模型在时间滚动折上的 accuracy 轨迹。曲线波动用于刻画时序稳定性与阶段适配能力。")
            lines.append("")

    if language == "bilingual":
        lines.append("---")
        lines.append("")

    if language in {"en", "bilingual"}:
        lines.append(f"# {title} Paper Caption Templates")
        lines.append("")
        lines.append("Figure 1 (comparison_main_metrics.png) caption draft:")
        lines.append(
            "Comparison across models under identical walk-forward validation and threshold calibration, including accuracy, "
            "balanced_accuracy, auc, and selection_score. "
            f"The top-ranked model is {top['label']} (selection_score={float(top['selection_score']):.4f})."
        )
        lines.append("")
        lines.append("Figure 2 (comparison_thresholds_and_coverage.png) caption draft:")
        lines.append(
            "Comparison of decision threshold, confidence threshold, and high-confidence coverage. "
            "This figure highlights the trade-off between signal purity and tradable sample size."
        )
        lines.append("")
        if has_folds:
            lines.append("Figure 3 (comparison_fold_accuracy_trend.png) caption draft:")
            lines.append(
                "Per-fold accuracy trajectories over rolling time splits. The volatility of curves reflects temporal stability and regime adaptability."
            )
            lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = pd.read_csv(args.summary_csv)
    summary = _sorted_summary(summary)

    outputs = [
        plot_main_metrics(summary, output_dir, args.title),
        plot_thresholds_and_coverage(summary, output_dir, args.title),
    ]

    folds = None
    if args.folds_csv:
        folds = pd.read_csv(args.folds_csv)
        outputs.append(plot_fold_accuracy(folds, output_dir, args.title))

    notes_path = None
    if not args.no_notes:
        notes_path = write_chart_notes(
            summary,
            output_dir,
            args.title,
            outputs,
            folds=folds,
            language=args.notes_language,
        )

    captions_path = None
    if not args.no_paper_captions:
        captions_path = write_paper_captions(
            summary,
            output_dir,
            args.title,
            has_folds=folds is not None,
            language=args.notes_language,
        )

    print("图表已生成:")
    for path in outputs:
        print(path)
    if notes_path is not None:
        print(f"图表说明: {notes_path}")
    if captions_path is not None:
        print(f"论文图注模板: {captions_path}")


if __name__ == "__main__":
    main()
