from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import pandas as pd


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="一键运行投稿证据包：消融 + 稳健性 + 显著性 + 评估报告")
    parser.add_argument("--python", default="d:/FISM/.venv/Scripts/python.exe")
    parser.add_argument("--dataset-path", default="artifacts/star_history.csv")
    parser.add_argument("--root-output", default="artifacts/model_comparison/publication_pack")
    parser.add_argument("--recent-start-date", default="2024-01-01")
    return parser


def _run(cmd: list[str]) -> None:
    print("RUN:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _try_run(cmd: list[str]) -> bool:
    print("RUN:", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError:
        return False


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise RuntimeError(f"缺少文件: {path}")
    return pd.read_csv(path)


def _grade(readiness: dict[str, bool]) -> str:
    score = sum(1 for v in readiness.values() if v)
    if score >= 5:
        return "ready_for_application_or_empirical_journal"
    if score >= 3:
        return "close_but_need_more_evidence"
    return "not_ready_for_method_innovation_submission"


def main() -> None:
    args = build_parser().parse_args()
    root = Path(args.root_output)
    root.mkdir(parents=True, exist_ok=True)

    ablation_dir = root / "ablation_main"
    robust_h3_dir = root / "robustness_h3"
    robust_recent_dir = root / "robustness_recent"
    benchmark_dir = root / "benchmark_models"
    sig_path = root / "significance_table.csv"
    report_path = root / "publication_readiness_report.md"

    _run(
        [
            args.python,
            "run_ablation.py",
            "--preset",
            "benchmark-thoughtful",
            "--dataset-path",
            args.dataset_path,
            "--output-dir",
            str(ablation_dir),
            "--include-benchmark-models",
            "--benchmark-output-dir",
            str(benchmark_dir),
        ]
    )

    _run(
        [
            args.python,
            "run_ablation.py",
            "--preset",
            "benchmark-thoughtful",
            "--dataset-path",
            args.dataset_path,
            "--horizon",
            "3",
            "--output-dir",
            str(robust_h3_dir),
        ]
    )

    recent_candidates = [args.recent_start_date, "2023-01-01", "2022-01-01"]
    recent_ok = False
    recent_used = ""
    for start_date in recent_candidates:
        recent_ok = _try_run(
            [
                args.python,
                "run_ablation.py",
                "--preset",
                "benchmark-thoughtful",
                "--dataset-path",
                args.dataset_path,
                "--start-date",
                start_date,
                "--output-dir",
                str(robust_recent_dir),
            ]
        )
        if recent_ok:
            recent_used = start_date
            break
    if not recent_ok:
        raise RuntimeError("recent-period 稳健性实验在候选起始日期下均未生成有效折")

    benchmark_folds = benchmark_dir / "model_comparison_folds_benchmark_thoughtful_h5.csv"
    _run(
        [
            args.python,
            "run_significance.py",
            "--ablation-folds",
            str(ablation_dir / "ablation_folds.csv"),
            "--benchmark-folds",
            str(benchmark_folds),
            "--output",
            str(sig_path),
        ]
    )

    main_ab = _read_csv(ablation_dir / "ablation_summary.csv")
    h3_ab = _read_csv(robust_h3_dir / "ablation_summary.csv")
    recent_ab = _read_csv(robust_recent_dir / "ablation_summary.csv")
    sig = _read_csv(sig_path)

    full_main = main_ab.loc[main_ab["variant"] == "full_method"].iloc[0]
    full_h3 = h3_ab.loc[h3_ab["variant"] == "full_method"].iloc[0]
    full_recent = recent_ab.loc[recent_ab["variant"] == "full_method"].iloc[0]

    ablate_rows = main_ab.loc[main_ab["variant"].str.startswith("ablate_", na=False)]
    ablation_support = bool((ablate_rows["delta_selection_score_vs_full"] < 0).all()) if not ablate_rows.empty else False

    current_vs_baselines = sig.loc[
        (sig["left"] == "current_extra_trees") & (sig["metric"].isin(["balanced_accuracy", "auc"]))
    ]
    significant_edges = bool((current_vs_baselines["p_value"] < 0.05).any()) if not current_vs_baselines.empty else False

    readiness = {
        "main_selection_score_ge_0p65": float(full_main["selection_score"]) >= 0.65,
        "main_auc_ge_0p70": float(full_main["auc"]) >= 0.70,
        "main_high_conf_acc_ge_0p80": float(full_main["calibrated_high_conf_accuracy"]) >= 0.80,
        "ablation_supports_method": ablation_support,
        "robust_h3_selection_score_ge_0p60": float(full_h3["selection_score"]) >= 0.60,
        "robust_recent_selection_score_ge_0p60": float(full_recent["selection_score"]) >= 0.60,
        "significant_edge_exists": significant_edges,
    }

    grade = _grade(readiness)

    lines: list[str] = []
    lines.append("# 投稿就绪评估报告")
    lines.append("")
    lines.append("## 1. 主实验核心结果")
    lines.append("")
    lines.append(f"- selection_score: {float(full_main['selection_score']):.4f}")
    lines.append(f"- accuracy: {float(full_main['accuracy']):.4f}")
    lines.append(f"- balanced_accuracy: {float(full_main['balanced_accuracy']):.4f}")
    lines.append(f"- auc: {float(full_main['auc']):.4f}")
    lines.append(f"- calibrated_high_conf_accuracy: {float(full_main['calibrated_high_conf_accuracy']):.4f}")
    lines.append(f"- calibrated_coverage: {float(full_main['calibrated_coverage']):.4f}")
    lines.append("")
    lines.append("## 2. 稳健性摘要")
    lines.append("")
    lines.append(f"- horizon=3 selection_score: {float(full_h3['selection_score']):.4f}")
    lines.append(f"- recent period start_date: {recent_used}")
    lines.append(f"- recent period selection_score: {float(full_recent['selection_score']):.4f}")
    lines.append("")
    lines.append("## 3. 证据检查清单")
    lines.append("")
    for k, v in readiness.items():
        lines.append(f"- {k}: {'PASS' if v else 'FAIL'}")
    lines.append("")
    lines.append("## 4. 评估结论")
    lines.append("")
    lines.append(f"- grade: {grade}")
    lines.append("")
    lines.append("## 5. 关键输出路径")
    lines.append("")
    lines.append(f"- ablation main: {ablation_dir / 'ablation_summary.csv'}")
    lines.append(f"- robustness h3: {robust_h3_dir / 'ablation_summary.csv'}")
    lines.append(f"- robustness recent: {robust_recent_dir / 'ablation_summary.csv'}")
    lines.append(f"- significance: {sig_path}")

    report_path.write_text("\n".join(lines), encoding="utf-8")

    print("\n投稿证据包已完成")
    print(f"评估报告: {report_path}")
    print(f"评估等级: {grade}")


if __name__ == "__main__":
    main()
