"""Rebuild audit tables, metrics, and the README figures."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kbo_fatigue import (
    build_audit_metrics,
    decile_summary,
    forward_decile_summary,
    forward_validation_summary,
    load_dataset,
)


DATA = ROOT / "data" / "final" / "fatigue_with_index.csv"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"

NAVY = "#14213D"
BLUE = "#2D6CDF"
ORANGE = "#F59E0B"
RED = "#D1495B"
GRAY = "#64748B"
GRID = "#DDE3EA"


def style_axis(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)


def save_score_definition(frame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    raw_cap = frame["피로도지수"].quantile(0.99)
    raw = frame.loc[frame["피로도지수"] <= raw_cap, "피로도지수"]
    axes[0].hist(raw, bins=45, color=BLUE, alpha=0.9)
    axes[0].axvline(
        frame["피로도지수"].quantile(0.8), color=ORANGE, linestyle="--",
        linewidth=2, label="80th percentile",
    )
    axes[0].set_title("Original index (values above P99 clipped)", loc="left")
    axes[0].set_xlabel("Original fatigue index")
    axes[0].set_ylabel("Appearances")
    axes[0].legend(frameon=False)
    style_axis(axes[0])

    axes[1].hist(
        frame["피로도지수_점수"], bins=20, color=NAVY, alpha=0.9,
        edgecolor="white", linewidth=1.2,
    )
    axes[1].set_title("Published 0–100 score", loc="left")
    axes[1].set_xlabel("Empirical percentile rank")
    axes[1].set_ylabel("Appearances")
    style_axis(axes[1])
    fig.suptitle(
        "The 0–100 score is a percentile transformation, not a probability",
        x=0.06, ha="left", fontsize=15, fontweight="bold", color=NAVY,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(FIGURES / "01_score_definition.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_outcome_relationship(summary) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    colors = {"WHIP": BLUE, "ERA": ORANGE, "FIP": RED, "GS": NAVY}
    for ax, metric in zip(axes.flat, ("WHIP", "ERA", "FIP", "GS")):
        ax.plot(
            summary["score_decile"], summary[metric], marker="o",
            linewidth=2.4, color=colors[metric],
        )
        ax.set_title(f"Median {metric}", loc="left", fontweight="bold")
        ax.set_ylabel(metric)
        style_axis(ax)
    for ax in axes[-1]:
        ax.set_xlabel("Score decile (1 = lowest, 10 = highest)")
        ax.set_xticks(range(1, 11))
    fig.suptitle(
        "Same-game outcomes move with score deciles",
        x=0.06, ha="left", fontsize=15, fontweight="bold", color=NAVY,
    )
    fig.text(
        0.06, 0.01,
        "Descriptive association only: the stored score was developed from performance-related data.",
        color=GRAY, fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.93))
    fig.savefig(FIGURES / "02_outcomes_by_decile.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_validation_diagnostic(metrics) -> None:
    labels = ["Recovery-failure label", "Injury-risk proxy"]
    values = [
        metrics["classification_diagnostics"]["recovery_failure_auc"],
        metrics["classification_diagnostics"]["injury_proxy_auc"],
    ]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    bars = ax.barh(labels, values, color=[BLUE, RED], height=0.55)
    ax.axvline(0.5, color=GRAY, linestyle="--", linewidth=1.5, label="Random = 0.50")
    ax.set_xlim(0.45, 0.60)
    ax.set_xlabel("ROC AUC")
    ax.set_title(
        "Recovery and injury labels require additional predictors",
        loc="left", fontsize=14, fontweight="bold", color=NAVY,
    )
    for bar, value in zip(bars, values):
        ax.text(value + 0.003, bar.get_y() + bar.get_height() / 2, f"{value:.3f}", va="center")
    ax.legend(frameon=False, loc="lower right")
    style_axis(ax)
    fig.tight_layout()
    fig.savefig(FIGURES / "03_threshold_validation.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_role_temporal_validation(summary) -> None:
    role_data = summary.loc[summary["role"].isin(["SP", "RP"])].set_index("role")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    positions = range(len(role_data))
    width = 0.34
    axes[0].bar(
        [value - width / 2 for value in positions], role_data["same_game_gs_r"],
        width=width, label="Same game", color=NAVY,
    )
    axes[0].bar(
        [value + width / 2 for value in positions], role_data["next_appearance_gs_r"],
        width=width, label="Next same-role appearance", color=ORANGE,
    )
    axes[0].set_xticks(list(positions), ["Starter", "Reliever"])
    axes[0].axhline(0, color=GRAY, linewidth=1)
    axes[0].set_ylabel("Correlation with GS")
    axes[0].set_title("Time horizon changes the signal", loc="left", fontweight="bold")
    axes[0].legend(frameon=False)
    style_axis(axes[0])

    axes[1].bar(
        [value - width / 2 for value in positions], role_data["below_72_6_next_gs_median"],
        width=width, label="Below 72.6", color=BLUE,
    )
    axes[1].bar(
        [value + width / 2 for value in positions], role_data["above_72_6_next_gs_median"],
        width=width, label="At or above 72.6", color=RED,
    )
    axes[1].set_xticks(list(positions), ["Starter", "Reliever"])
    axes[1].set_ylabel("Median next-appearance GS")
    axes[1].set_title("Role-specific threshold follow-up", loc="left", fontweight="bold")
    axes[1].legend(frameon=False)
    style_axis(axes[1])

    fig.suptitle(
        "The original score becomes more actionable when role and horizon are explicit",
        x=0.06, ha="left", fontsize=15, fontweight="bold", color=NAVY,
    )
    fig.text(
        0.06, 0.01,
        "Same player and same role are required for next-appearance comparisons; results are descriptive.",
        color=GRAY, fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.91))
    fig.savefig(FIGURES / "04_role_and_horizon.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_kim_taekyeon_case(frame) -> None:
    case = frame.loc[
        frame["선수"].eq("김택연")
        & frame["날짜"].between("2024-08-15", "2024-09-01")
    ].copy()
    focus_date = pd.Timestamp("2024-08-24")
    focus = case.loc[case["날짜"].eq(focus_date)].iloc[0]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(case["날짜"], case["피로도지수_점수"], color=NAVY, marker="o", linewidth=2.5)
    ax.axhline(72.6, color=ORANGE, linestyle="--", linewidth=1.8, label="Original alert candidate: 72.6")
    ax.scatter([focus_date], [focus["피로도지수_점수"]], color=RED, s=110, zorder=5)
    ax.annotate(
        f"Aug 24 · score {focus['피로도지수_점수']:.1f}\nERA {focus['ERA']:.0f} · WHIP {focus['WHIP']:.0f} · GS {focus['GS']:.0f}",
        xy=(focus_date, focus["피로도지수_점수"]), xytext=(-110, -75),
        textcoords="offset points", arrowprops={"arrowstyle": "->", "color": RED},
        bbox={"boxstyle": "round,pad=0.4", "fc": "white", "ec": RED},
    )
    ax.set_ylim(0, 108)
    ax.set_ylabel("Fatigue score percentile")
    ax.set_xlabel("Appearance date")
    ax.set_title(
        "Kim Taek-yeon case from the original presentation is reproduced",
        loc="left", fontsize=14, fontweight="bold", color=NAVY,
    )
    ax.legend(frameon=False, loc="lower left")
    style_axis(ax)
    fig.autofmt_xdate(rotation=25)
    fig.tight_layout()
    fig.savefig(FIGURES / "05_kim_taekyeon_case.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_temporal_model(metrics) -> None:
    result = metrics["temporal_starter_model"]
    labels = ["Historical mean\nbaseline", "Workload + recent form\nridge model"]
    values = [result["mean_baseline"]["mae"], result["model"]["mae"]]
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    bars = ax.bar(labels, values, color=[GRAY, BLUE], width=0.55)
    ax.set_ylabel("2024 next-start GS MAE (lower is better)")
    ax.set_title(
        "Time-ordered starter model adds modest predictive value",
        loc="left", fontsize=14, fontweight="bold", color=NAVY,
    )
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.08, f"{value:.2f}", ha="center")
    ax.text(
        0.98, 0.93,
        f"MAE improvement {result['mae_improvement_pct']:.1f}%\nTest R² {result['model']['r2']:.3f} · n={result['test_n']}",
        transform=ax.transAxes, ha="right", va="top",
        bbox={"boxstyle": "round,pad=0.4", "fc": "white", "ec": GRID},
    )
    ax.set_ylim(0, max(values) * 1.22)
    style_axis(ax)
    fig.tight_layout()
    fig.savefig(FIGURES / "06_temporal_starter_model.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    REPORTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    frame = load_dataset(DATA)
    metrics = build_audit_metrics(frame)
    summary = decile_summary(frame)
    forward = forward_validation_summary(frame)
    forward_deciles = forward_decile_summary(frame)
    (REPORTS / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    summary.to_csv(REPORTS / "decile_summary.csv", index=False, encoding="utf-8")
    forward.to_csv(REPORTS / "forward_validation.csv", index=False, encoding="utf-8")
    forward_deciles.to_csv(
        REPORTS / "next_appearance_deciles.csv", index=False, encoding="utf-8"
    )
    save_score_definition(frame)
    save_outcome_relationship(summary)
    save_validation_diagnostic(metrics)
    save_role_temporal_validation(forward)
    save_kim_taekyeon_case(frame)
    save_temporal_model(metrics)
    print(f"Built report artifacts in {REPORTS.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
