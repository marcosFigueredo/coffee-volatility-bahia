"""Generate Experiments/Results figures for the paper from the validated CSV artifacts.

Reads:
  dados_commodities_bahia/validated_comparison_pooled.csv
  dados_commodities_bahia/validated_incremental_gains.csv
  dados_commodities_bahia/regime_analysis_results.csv
  dados_commodities_bahia/feature_group_importance_summary.csv

Writes (PDF + PNG) into els-cas-templates/figs/:
  fig_pooled_comparison.{pdf,png}
  fig_incremental_gains.{pdf,png}
  fig_regime_analysis.{pdf,png}
  fig_feature_importance.{pdf,png}
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = Path(
    r"G:\My Drive\UNEB\PPGMSB\MarcosProducaoCientifica\2026\specialIssues\Machine Learning in Finance"
)
DATA_DIR = BASE_DIR / "dados_commodities_bahia"
FIG_DIR = BASE_DIR / "els-cas-templates" / "figs"
FIG_DIR.mkdir(parents=True, exist_ok=True)

COLORS = {
    "GARCH(1,1)": "#0B5FA5",
    "HAR-RV": "#F39C12",
    "RandomForest": "#2E9E5B",
    "LightGBM": "#D83A56",
    "XGBoost": "#7F5AA8",
    "E2_cross_commodity": "#0B5FA5",
    "E3_financial": "#F39C12",
    "E4_full": "#D83A56",
    "1_Own_History": "#0B5FA5",
    "2_Cross_Commodity": "#F39C12",
    "3_Financial_Macro": "#D83A56",
}

MODEL_LABELS = {
    "GARCH(1,1)": "GARCH(1,1)",
    "HAR-RV": "HAR-RV",
    "RandomForest": "Random Forest (best)",
    "LightGBM": "LightGBM (best)",
    "XGBoost": "XGBoost (best)",
}


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 9,
            "legend.fontsize": 7.4,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
        }
    )


def save(fig, name: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_pooled_comparison() -> None:
    df = pd.read_csv(DATA_DIR / "validated_comparison_pooled.csv")
    horizons = [1, 5, 20]

    rows = []
    for h in horizons:
        sub = df[df["horizon"] == h]
        for m in ["GARCH(1,1)", "HAR-RV"]:
            r = sub[sub["model"] == m].iloc[0]
            rows.append({"horizon": h, "model": m, "qlike": r.qlike, "mae": r.mae, "rmse": r.rmse})
        for m in ["RandomForest", "LightGBM", "XGBoost"]:
            ms = sub[sub["model"] == m]
            rows.append(
                {
                    "horizon": h,
                    "model": m,
                    "qlike": ms["qlike"].min(),
                    "mae": ms["mae"].min(),
                    "rmse": ms["rmse"].min(),
                }
            )
    agg = pd.DataFrame(rows)

    models = ["GARCH(1,1)", "HAR-RV", "RandomForest", "LightGBM", "XGBoost"]
    x = np.arange(len(horizons))
    width = 0.16

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.75))
    metrics = [("qlike", "QLIKE (log scale)", True), ("mae", "MAE", False), ("rmse", "RMSE", False)]

    for ax, (col, title, logscale) in zip(axes, metrics):
        for i, m in enumerate(models):
            vals = [agg[(agg.horizon == h) & (agg.model == m)][col].iloc[0] for h in horizons]
            ax.bar(x + (i - 2) * width, vals, width=width, color=COLORS[m], label=MODEL_LABELS[m])
        ax.set_xticks(x)
        ax.set_xticklabels([f"h={h}" for h in horizons])
        ax.set_title(title)
        if logscale:
            ax.set_yscale("log")
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.75)
        ax.grid(axis="x", visible=False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.06),
        ncol=5,
        frameon=False,
        columnspacing=1.1,
    )
    fig.subplots_adjust(wspace=0.32, bottom=0.28, top=0.90)
    save(fig, "fig_pooled_comparison")


def fig_incremental_gains() -> None:
    # gains on pooled losses (ratio of pooled means), from 14_robustness_checks.py
    df = pd.read_csv(DATA_DIR / "robust_pooled_gains.csv")
    df = df[(df["model"] == "RandomForest") & (df["scenario"] == "all_eps=0.0001")]
    horizons = [1, 5, 20]
    experiments = ["E2_cross_commodity", "E3_financial", "E4_full"]
    exp_labels = {"E2_cross_commodity": "E2 (cross-commodity)", "E3_financial": "E3 (financial)", "E4_full": "E4 (full)"}

    agg = (
        df.groupby(["horizon", "experiment"])[["qlike_gain_pct", "mae_gain_pct", "rmse_gain_pct"]]
        .mean()
        .reset_index()
    )

    x = np.arange(len(horizons))
    width = 0.25

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), sharey=False)
    metrics = [("qlike_gain_pct", "QLIKE gain (%)"), ("mae_gain_pct", "MAE gain (%)"), ("rmse_gain_pct", "RMSE gain (%)")]

    for ax, (col, title) in zip(axes, metrics):
        for i, e in enumerate(experiments):
            vals = [
                agg[(agg.horizon == h) & (agg.experiment == e)][col].iloc[0]
                if not agg[(agg.horizon == h) & (agg.experiment == e)].empty
                else np.nan
                for h in horizons
            ]
            ax.bar(x + (i - 1) * width, vals, width=width, color=COLORS[e], label=exp_labels[e])
        ax.axhline(0, color="#555555", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([f"h={h}" for h in horizons])
        ax.set_title(title)
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.75)
        ax.grid(axis="x", visible=False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.08), ncol=3, frameon=False)
    fig.subplots_adjust(wspace=0.4, bottom=0.32, top=0.88)
    save(fig, "fig_incremental_gains")


def fig_regime_analysis() -> None:
    """MAE gain over E1 by ex-ante volatility regime (from 14_robustness_checks.py)."""
    df = pd.read_csv(DATA_DIR / "robust_exante_regime_gains.csv")
    regimes = ["1_Low", "2_Medium", "3_High"]
    regime_labels = {"1_Low": "Low", "2_Medium": "Medium", "3_High": "High"}
    blocks = ["E2_cross_commodity", "E3_financial", "E4_full"]
    block_titles = {"E2_cross_commodity": "E2 (cross-commodity)", "E3_financial": "E3 (financial)", "E4_full": "E4 (full)"}
    models = ["RandomForest", "LightGBM", "XGBoost"]
    model_labels = {"RandomForest": "Random Forest", "LightGBM": "LightGBM", "XGBoost": "XGBoost"}

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7), sharey=True)
    x = np.arange(len(regimes))
    width = 0.25
    for ax, b in zip(axes, blocks):
        sub = df[df["experiment"] == b]
        for i, m in enumerate(models):
            vals = [sub[(sub.model == m) & (sub.regime == r)]["mae_gain_pct"].iloc[0] for r in regimes]
            ax.bar(x + (i - 1) * width, vals, width=width, color=COLORS[m], label=model_labels[m])
        ax.axhline(0, color="#555555", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([regime_labels[r] for r in regimes])
        ax.set_title(block_titles[b])
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.75)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("MAE gain over E1 (%)")
    axes[1].set_xlabel("Ex-ante volatility regime (trailing HVol$_{20}$ terciles)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.08), ncol=3, frameon=False)
    fig.subplots_adjust(wspace=0.12, bottom=0.32, top=0.88)
    save(fig, "fig_regime_analysis")


def fig_feature_importance() -> None:
    df = pd.read_csv(DATA_DIR / "feature_group_importance_summary.csv")
    df.columns = [c.lstrip("\ufeff") for c in df.columns]

    horizons = [1, 5, 20]
    group_labels = {
        "1_Own_History": "Own history",
        "2_Cross_Commodity": "Cross-commodity",
        "3_Financial_Macro": "Financial/macro",
    }
    groups = list(group_labels.keys())

    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.75), sharey=False)

    for ax, experiment, title in zip(axes, ["E2_cross_commodity", "E4_full"], ["E2 (own + cross-commodity)", "E4 (full)"]):
        sub = df[df["experiment"] == experiment]
        agg = sub.groupby(["horizon", "feature_group"])["total_mean_importance"].mean().reset_index()
        x = np.arange(len(horizons))
        width = 0.25
        present_groups = [g for g in groups if g in agg["feature_group"].unique()]
        for i, g in enumerate(present_groups):
            vals = [
                agg[(agg.horizon == h) & (agg.feature_group == g)]["total_mean_importance"].iloc[0]
                if not agg[(agg.horizon == h) & (agg.feature_group == g)].empty
                else 0.0
                for h in horizons
            ]
            ax.bar(x + (i - (len(present_groups) - 1) / 2) * width, vals, width=width, color=COLORS[g], label=group_labels[g])
        ax.set_xticks(x)
        ax.set_xticklabels([f"h={h}" for h in horizons])
        ax.set_title(title)
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.75)
        ax.grid(axis="x", visible=False)

    axes[0].set_ylabel("Mean feature importance")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.08), ncol=3, frameon=False)
    fig.subplots_adjust(wspace=0.15, bottom=0.34, top=0.88)
    save(fig, "fig_feature_importance")


def fig_variety_importance() -> None:
    df = pd.read_csv(DATA_DIR / "feature_group_importance_summary.csv")
    df.columns = [c.lstrip("﻿") for c in df.columns]
    df = df[df["experiment"] == "E4_full"]

    horizons = [1, 5, 20]
    group_labels = {
        "1_Own_History": "Own history",
        "2_Cross_Commodity": "Cross-commodity",
        "3_Financial_Macro": "Financial/macro",
    }
    groups = list(group_labels.keys())
    variety_titles = {"arabica": "Arabica (VDC + LEM)", "conilon": "Conilon (Eunápolis)"}

    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.75), sharey=False)

    for ax, variety in zip(axes, ["arabica", "conilon"]):
        sub = df[df["variety"] == variety]
        agg = sub.groupby(["horizon", "feature_group"])["total_mean_importance"].mean().reset_index()
        x = np.arange(len(horizons))
        width = 0.25
        for i, g in enumerate(groups):
            vals = [
                agg[(agg.horizon == h) & (agg.feature_group == g)]["total_mean_importance"].iloc[0]
                if not agg[(agg.horizon == h) & (agg.feature_group == g)].empty
                else 0.0
                for h in horizons
            ]
            ax.bar(x + (i - 1) * width, vals, width=width, color=COLORS[g], label=group_labels[g])
        ax.set_xticks(x)
        ax.set_xticklabels([f"h={h}" for h in horizons])
        ax.set_title(variety_titles[variety])
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.75)
        ax.grid(axis="x", visible=False)

    axes[0].set_ylabel("Mean feature importance (E4)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.08), ncol=3, frameon=False)
    fig.subplots_adjust(wspace=0.3, bottom=0.34, top=0.88)
    save(fig, "fig_variety_importance")


if __name__ == "__main__":
    style()
    fig_pooled_comparison()
    fig_incremental_gains()
    fig_regime_analysis()
    fig_feature_importance()
    fig_variety_importance()
    print("Figures written to", FIG_DIR)
