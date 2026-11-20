#!/usr/bin/env python3
"""Leak-free cross-validated multi-omics ML classification, SHAP explainability, and lifelines Cox/KM survival."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from lifelines import CoxPHFitter
from lifelines.statistics import logrank_test
from lifelines.utils import concordance_index
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class TopVarianceSelector(BaseEstimator, TransformerMixin):
    """Inside-fold unsupervised variance selector to prevent train/test feature leakage."""

    def __init__(self, k: int = 100) -> None:
        self.k = k

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> TopVarianceSelector:
        arr = np.asarray(X, dtype=float)
        variances = np.var(arr, axis=0)
        k_eff = min(self.k, arr.shape[1])
        self.selected_idx_ = np.argsort(variances)[::-1][:k_eff]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X, dtype=float)[:, self.selected_idx_]


def extract_binary_shap(shap_vals: object) -> np.ndarray:
    """Extract positive-class 2D SHAP matrix (n_samples, n_features) across shap versions."""
    if isinstance(shap_vals, list):
        arr = np.asarray(shap_vals[1] if len(shap_vals) > 1 else shap_vals[0], dtype=float)
    else:
        arr = np.asarray(shap_vals, dtype=float)
        if arr.ndim == 3:
            arr = arr[:, :, 1]
    return arr


def build_models(seed: int) -> dict[str, object]:
    """Return dictionary of scikit-learn classifiers for multi-omics benchmarking."""
    return {
        "LogisticRegression_L2": LogisticRegression(C=0.5, max_iter=1000, class_weight="balanced", random_state=seed),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, max_depth=6, min_samples_leaf=4, class_weight="balanced", random_state=seed, n_jobs=1
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=120, max_depth=3, learning_rate=0.05, random_state=seed
        ),
    }


def run_cv_benchmarks(
    rna_df: pd.DataFrame,
    cnv_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    top_rna: int,
    top_cnv: int,
    n_splits: int = 5,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run leak-free stratified CV across RNA-only, CNV-only, and Multi-Omics modalities."""
    valid_meta = meta_df.dropna(subset=["target"]).copy()
    samples = valid_meta["sample"].tolist()
    y = valid_meta["target"].astype(int).to_numpy()

    X_rna = rna_df.loc[samples].to_numpy(dtype=float)
    X_cnv = cnv_df.loc[samples].to_numpy(dtype=float)
    X_multi = np.hstack([X_rna, X_cnv])

    modalities = {
        "RNA_only": (X_rna, top_rna),
        "CNV_only": (X_cnv, top_cnv),
        "MultiOmics_RNA+CNV": (X_multi, top_rna + top_cnv),
    }

    min_class = int(min((y == 0).sum(), (y == 1).sum()))
    splits = min(n_splits, max(2, min_class))
    skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=seed)

    metric_rows: list[dict[str, object]] = []
    oof_rows: list[dict[str, object]] = []

    for mod_name, (X_mat, k_feat) in modalities.items():
        for model_name, clf in build_models(seed).items():
            oof_prob = np.zeros(len(y), dtype=float)
            for train_idx, test_idx in skf.split(X_mat, y):
                pipe = Pipeline(
                    [
                        ("select", TopVarianceSelector(k=k_feat)),
                        ("scale", StandardScaler()),
                        ("clf", clf),
                    ]
                )
                pipe.fit(X_mat[train_idx], y[train_idx])
                oof_prob[test_idx] = pipe.predict_proba(X_mat[test_idx])[:, 1]

            roc_auc = float(roc_auc_score(y, oof_prob))
            pr_auc = float(average_precision_score(y, oof_prob))
            bal_acc = float(balanced_accuracy_score(y, (oof_prob >= 0.5).astype(int)))
            brier = float(brier_score_loss(y, oof_prob))

            metric_rows.append(
                {
                    "modality": mod_name,
                    "model": model_name,
                    "n_samples": len(y),
                    "n_splits": splits,
                    "roc_auc": roc_auc,
                    "pr_auc": pr_auc,
                    "balanced_accuracy": bal_acc,
                    "brier_score": brier,
                }
            )
            for s_id, y_true, prob in zip(samples, y, oof_prob):
                oof_rows.append(
                    {
                        "sample": s_id,
                        "modality": mod_name,
                        "model": model_name,
                        "y_true": int(y_true),
                        "y_prob": float(prob),
                    }
                )

    metrics_df = pd.DataFrame(metric_rows).sort_values("roc_auc", ascending=False).reset_index(drop=True)
    oof_df = pd.DataFrame(oof_rows)
    return metrics_df, oof_df


def compute_shap_biomarkers(
    rna_df: pd.DataFrame,
    cnv_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    top_rna: int,
    top_cnv: int,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit multi-omics RandomForest and compute TreeExplainer SHAP biomarker importances."""
    valid_meta = meta_df.dropna(subset=["target"]).copy()
    samples = valid_meta["sample"].tolist()
    y = valid_meta["target"].astype(int).to_numpy()

    rna_sub = rna_df.loc[samples]
    cnv_sub = cnv_df.loc[samples]

    top_rna_cols = rna_sub.var(axis=0).nlargest(min(top_rna, rna_sub.shape[1])).index
    top_cnv_cols = cnv_sub.var(axis=0).nlargest(min(top_cnv, cnv_sub.shape[1])).index
    X_multi_df = pd.concat([rna_sub[top_rna_cols], cnv_sub[top_cnv_cols]], axis=1)

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_multi_df), index=X_multi_df.index, columns=X_multi_df.columns)

    rf = RandomForestClassifier(
        n_estimators=250, max_depth=6, min_samples_leaf=4, class_weight="balanced", random_state=seed, n_jobs=1
    )
    rf.fit(X_scaled, y)

    explainer = shap.TreeExplainer(rf)
    shap_mat = extract_binary_shap(explainer.shap_values(X_scaled))
    mean_abs_shap = np.mean(np.abs(shap_mat), axis=0)

    imp_df = pd.DataFrame(
        {
            "feature": X_scaled.columns,
            "modality": [f.split("__")[0] for f in X_scaled.columns],
            "gene": [f.split("__", 1)[1] for f in X_scaled.columns],
            "mean_abs_shap": mean_abs_shap,
        }
    ).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    total_shap = float(imp_df["mean_abs_shap"].sum()) or 1.0
    imp_df["shap_share_pct"] = 100.0 * imp_df["mean_abs_shap"] / total_shap
    return imp_df, X_multi_df


def run_lifelines_survival(
    rna_df: pd.DataFrame,
    cnv_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    shap_imp_df: pd.DataFrame,
    top_n_cox: int = 10,
    penalizer: float = 0.1,
    n_splits: int = 5,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit out-of-fold penalized Cox PH survival model on top SHAP multi-omics biomarkers."""
    surv_meta = meta_df.dropna(subset=["OS", "OS_time"]).copy().reset_index(drop=True)
    samples = surv_meta["sample"].tolist()
    top_feats = shap_imp_df["feature"].head(min(top_n_cox, len(shap_imp_df))).tolist()

    X_all = pd.concat([rna_df.loc[samples], cnv_df.loc[samples]], axis=1)[top_feats].copy()
    # Standardize features for stable Cox HR interpretation (per 1 SD)
    X_std = (X_all - X_all.mean(axis=0)) / X_all.std(axis=0).replace(0, 1.0)

    cox_df = X_std.copy()
    cox_df["OS_years"] = surv_meta["OS_time"].to_numpy(dtype=float) / 365.25
    cox_df["OS"] = surv_meta["OS"].to_numpy(dtype=int)

    # Out-of-fold cross-validated risk score
    events = cox_df["OS"].to_numpy()
    splits = min(n_splits, max(2, int(events.sum() // 2)))
    skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=seed)
    oof_risk = np.zeros(len(cox_df), dtype=float)

    for tr_idx, te_idx in skf.split(cox_df, events):
        cph = CoxPHFitter(penalizer=penalizer)
        cph.fit(cox_df.iloc[tr_idx], duration_col="OS_years", event_col="OS")
        oof_risk[te_idx] = cph.predict_partial_hazard(cox_df.iloc[te_idx][top_feats]).to_numpy().ravel()

    oof_c_index = float(concordance_index(cox_df["OS_years"], -oof_risk, cox_df["OS"]))
    med_risk = float(np.median(oof_risk))
    risk_group = np.where(oof_risk >= med_risk, "High_Risk", "Low_Risk")

    high_mask = risk_group == "High_Risk"
    lr = logrank_test(
        cox_df.loc[high_mask, "OS_years"],
        cox_df.loc[~high_mask, "OS_years"],
        event_observed_A=cox_df.loc[high_mask, "OS"],
        event_observed_B=cox_df.loc[~high_mask, "OS"],
    )

    surv_oof_df = pd.DataFrame(
        {
            "sample": samples,
            "OS_years": cox_df["OS_years"].to_numpy(),
            "OS": cox_df["OS"].to_numpy(),
            "oof_risk_score": oof_risk,
            "risk_group": risk_group,
            "oof_c_index": oof_c_index,
            "km_logrank_p": float(lr.p_value),
        }
    )

    # Full-cohort penalized multivariate Cox PH summary for forest plot
    cph_full = CoxPHFitter(penalizer=penalizer)
    cph_full.fit(cox_df, duration_col="OS_years", event_col="OS")
    s = cph_full.summary.reset_index().rename(columns={"covariate": "feature"})
    cox_biomarkers_df = pd.DataFrame(
        {
            "feature": s["feature"],
            "modality": [str(f).split("__")[0] for f in s["feature"]],
            "gene": [str(f).split("__", 1)[1] for f in s["feature"]],
            "hr": s["exp(coef)"].to_numpy(dtype=float),
            "ci_low": s["exp(coef) lower 95%"].to_numpy(dtype=float),
            "ci_high": s["exp(coef) upper 95%"].to_numpy(dtype=float),
            "pvalue": s["p"].to_numpy(dtype=float),
            "oof_c_index": oof_c_index,
            "km_logrank_p": float(lr.p_value),
        }
    ).sort_values("pvalue").reset_index(drop=True)

    logger.info("Survival Cox PH: OOF C-index=%.3f, KM log-rank p=%.3g.", oof_c_index, float(lr.p_value))
    return surv_oof_df, cox_biomarkers_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-omics ML benchmarking, SHAP, and lifelines survival.")
    parser.add_argument("--rna", required=True, type=Path)
    parser.add_argument("--cnv", required=True, type=Path)
    parser.add_argument("--meta", required=True, type=Path)
    parser.add_argument("--top-rna", type=int, default=120)
    parser.add_argument("--top-cnv", type=int, default=80)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--top-shap-cox", type=int, default=10)
    parser.add_argument("--penalizer", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-metrics", required=True, type=Path)
    parser.add_argument("--out-oof", required=True, type=Path)
    parser.add_argument("--out-shap", required=True, type=Path)
    parser.add_argument("--out-surv-oof", required=True, type=Path)
    parser.add_argument("--out-surv-cox", required=True, type=Path)
    args = parser.parse_args()

    rna_df = pd.read_csv(args.rna, sep="\t", index_col=0, compression="infer")
    cnv_df = pd.read_csv(args.cnv, sep="\t", index_col=0, compression="infer")
    meta_df = pd.read_csv(args.meta, sep="\t")

    metrics_df, oof_df = run_cv_benchmarks(
        rna_df, cnv_df, meta_df, args.top_rna, args.top_cnv, args.n_splits, args.seed
    )
    shap_imp_df, _ = compute_shap_biomarkers(
        rna_df, cnv_df, meta_df, args.top_rna, args.top_cnv, args.seed
    )
    surv_oof_df, cox_bio_df = run_lifelines_survival(
        rna_df, cnv_df, meta_df, shap_imp_df, args.top_shap_cox, args.penalizer, args.n_splits, args.seed
    )

    for p in (args.out_metrics, args.out_oof, args.out_shap, args.out_surv_oof, args.out_surv_cox):
        p.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(args.out_metrics, sep="\t", index=False)
    oof_df.to_csv(args.out_oof, sep="\t", index=False)
    shap_imp_df.to_csv(args.out_shap, sep="\t", index=False)
    surv_oof_df.to_csv(args.out_surv_oof, sep="\t", index=False)
    cox_bio_df.to_csv(args.out_surv_cox, sep="\t", index=False)
    logger.info("Top CV model:\n%s", metrics_df.head(3).to_string(index=False))


if __name__ == "__main__":
    main()
