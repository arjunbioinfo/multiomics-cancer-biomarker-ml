#!/usr/bin/env python3
"""Generate publication-quality summary figures for multi-omics ML, SHAP, and survival modeling."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from lifelines import KaplanMeierFitter
from sklearn.metrics import roc_curve

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)


def plot_modality_benchmark(metrics_df: pd.DataFrame, oof_df: pd.DataFrame, cohort: str, out_path: Path) -> None:
    """2-panel figure: CV ROC-AUC grouped barplot + out-of-fold ROC curves per modality."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.0, 4.9))

    sns.barplot(
        data=metrics_df,
        x="modality",
        y="roc_auc",
        hue="model",
        palette="Set2",
        ax=ax1,
    )
    ax1.axhline(0.5, color="#555555", ls="--", lw=0.9)
    ax1.set_ylim(0.35, 1.0)
    ax1.set_xlabel("")
    ax1.set_ylabel("Cross-Validated ROC-AUC")
    ax1.set_title(f"{cohort}: Modality × Model Benchmark", fontweight="bold")
    ax1.tick_params(axis="x", rotation=15, labelsize=9)
    ax1.legend(title="Model", frameon=True, fontsize=8.2)

    # Best model per modality ROC curve
    colors = {"RNA_only": "#386CB0", "CNV_only": "#1B9E77", "MultiOmics_RNA+CNV": "#D95F02"}
    for mod in ("RNA_only", "CNV_only", "MultiOmics_RNA+CNV"):
        sub_m = metrics_df[metrics_df["modality"] == mod]
        if sub_m.empty:
            continue
        best_model = str(sub_m.iloc[0]["model"])
        best_auc = float(sub_m.iloc[0]["roc_auc"])
        sub_oof = oof_df[(oof_df["modality"] == mod) & (oof_df["model"] == best_model)]
        fpr, tpr, _ = roc_curve(sub_oof["y_true"], sub_oof["y_prob"])
        ax2.plot(fpr, tpr, lw=2.0, color=colors.get(mod, "#333333"), label=f"{mod} ({best_model}, AUC={best_auc:.2f})")

    ax2.plot([0, 1], [0, 1], color="#999999", ls="--", lw=0.9)
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    ax2.set_title("Out-of-Fold ROC Curves (Best Model per Modality)", fontweight="bold")
    ax2.legend(frameon=True, fontsize=8.0, loc="lower right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_shap_biomarkers(shap_df: pd.DataFrame, cohort: str, out_path: Path) -> None:
    """2-panel figure: Top 15 SHAP biomarkers colored by modality + modality share breakdown."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.0, 5.2), gridspec_kw={"width_ratios": [1.6, 1.0]})

    top = shap_df.head(15).copy().iloc[::-1]
    palette = {"RNA": "#D95F02", "CNV": "#386CB0"}
    colors = [palette.get(str(m), "#666666") for m in top["modality"]]
    labels = [f"{r.gene} ({r.modality})" for r in top.itertuples(index=False)]

    ax1.barh(labels, top["mean_abs_shap"], color=colors, alpha=0.88, height=0.68)
    ax1.set_xlabel("Mean |SHAP value| (impact on model output)")
    ax1.set_title(f"{cohort}: Top 15 Multi-Omics Biomarkers (SHAP)", fontweight="bold")
    ax1.tick_params(axis="y", labelsize=8.8)

    mod_share = shap_df.groupby("modality", as_index=False)["shap_share_pct"].sum()
    ax2.bar(
        mod_share["modality"],
        mod_share["shap_share_pct"],
        color=[palette.get(str(m), "#666666") for m in mod_share["modality"]],
        width=0.52,
    )
    for idx, row in mod_share.iterrows():
        ax2.text(idx, row["shap_share_pct"] + 1.5, f"{row['shap_share_pct']:.1f}%", ha="center", fontweight="bold", fontsize=9.5)
    ax2.set_ylim(0, 105)
    ax2.set_ylabel("Share of Total |SHAP| Attribution (%)")
    ax2.set_title("Omics Modality Attribution Share", fontweight="bold")

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_survival_stratification(surv_oof: pd.DataFrame, cohort: str, out_path: Path) -> None:
    """Out-of-fold cross-validated Kaplan-Meier overall survival curves (High vs. Low risk)."""
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    kmf = KaplanMeierFitter()

    for grp, col in (("High_Risk", "#D95F02"), ("Low_Risk", "#1B9E77")):
        sub = surv_oof[surv_oof["risk_group"] == grp]
        kmf.fit(sub["OS_years"], event_observed=sub["OS"], label=f"{grp.replace('_', ' ')} (n={len(sub)})")
        kmf.plot_survival_function(ax=ax, ci_show=True, color=col, lw=2.0)

    c_idx = float(surv_oof["oof_c_index"].iloc[0])
    p_lr = float(surv_oof["km_logrank_p"].iloc[0])
    ax.text(
        0.04,
        0.10,
        f"Out-of-fold C-index = {c_idx:.3f}\nLog-rank p = {p_lr:.3g}",
        transform=ax.transAxes,
        fontsize=9.5,
        fontweight="bold",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.9, "edgecolor": "#cccccc"},
    )
    ax.set_ylim(0, 1.03)
    ax.set_xlabel("Overall Survival (years)")
    ax.set_ylabel("Survival Probability")
    ax.set_title(f"{cohort}: Out-of-Fold Multi-Omics Survival Risk Stratification", fontweight="bold")
    ax.legend(frameon=True, loc="upper right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_cox_forest(cox_df: pd.DataFrame, cohort: str, out_path: Path) -> None:
    """Forest plot of multivariate penalized Cox hazard ratios for top SHAP multi-omics biomarkers."""
    df = cox_df.head(12).copy().reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7.8, 5.0))
    y = np.arange(len(df))

    hrs = np.clip(df["hr"].to_numpy(dtype=float), 0.1, 10.0)
    ci_l = np.clip(df["ci_low"].to_numpy(dtype=float), 0.1, 10.0)
    ci_h = np.clip(df["ci_high"].to_numpy(dtype=float), 0.1, 10.0)
    xerr = np.vstack([np.maximum(hrs - ci_l, 0.0), np.maximum(ci_h - hrs, 0.0)])

    colors = ["#D95F02" if m == "RNA" else "#386CB0" for m in df["modality"]]
    ax.errorbar(hrs, y, xerr=xerr, fmt="o", color="#222222", ecolor="#555555", capsize=3.5, lw=1.3, zorder=3)
    ax.scatter(hrs, y, c=colors, s=65, zorder=4)
    ax.axvline(1.0, color="#666666", ls="--", lw=1.0)
    ax.set_xscale("log")

    labels = [f"{r.gene} ({r.modality})" for r in df.itertuples(index=False)]
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    for idx, row in df.iterrows():
        ax.text(ci_h[idx] * 1.06, idx, f"HR={row['hr']:.2f} (p={row['pvalue']:.2g})", va="center", fontsize=8.2)

    ax.set_xlabel("Multivariate Cox Hazard Ratio per SD (log scale, 95% CI)")
    ax.set_title(f"{cohort}: Penalized Cox PH Hazard Ratios (Orange=RNA, Blue=CNV)", fontweight="bold")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot multi-omics ML, SHAP, and survival figures.")
    parser.add_argument("--metrics", required=True, type=Path)
    parser.add_argument("--oof", required=True, type=Path)
    parser.add_argument("--shap", required=True, type=Path)
    parser.add_argument("--surv-oof", required=True, type=Path)
    parser.add_argument("--surv-cox", required=True, type=Path)
    parser.add_argument("--cohort", default="TCGA-LUSC")
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    metrics_df = pd.read_csv(args.metrics, sep="\t")
    oof_df = pd.read_csv(args.oof, sep="\t")
    shap_df = pd.read_csv(args.shap, sep="\t")
    surv_oof = pd.read_csv(args.surv_oof, sep="\t")
    cox_df = pd.read_csv(args.surv_cox, sep="\t")

    plot_modality_benchmark(metrics_df, oof_df, args.cohort, args.out_dir / "modality_model_benchmark.png")
    plot_shap_biomarkers(shap_df, args.cohort, args.out_dir / "shap_biomarkers_summary.png")
    plot_survival_stratification(surv_oof, args.cohort, args.out_dir / "survival_risk_stratification.png")
    plot_cox_forest(cox_df, args.cohort, args.out_dir / "cox_hazard_forest.png")
    logger.info("Saved 4 summary figures to %s", args.out_dir)


if __name__ == "__main__":
    main()
