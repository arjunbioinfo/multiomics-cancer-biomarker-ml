"""Unit and synthetic end-to-end smoke tests for multiomics-cancer-biomarker-ml."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "workflow" / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from download_multiomics import align_multiomics_cohort, parse_stage_binary  # noqa: E402
from plot_summary import (  # noqa: E402
    plot_cox_forest,
    plot_modality_benchmark,
    plot_shap_biomarkers,
    plot_survival_stratification,
)
from train_and_explain import (  # noqa: E402
    compute_shap_biomarkers,
    run_cv_benchmarks,
    run_lifelines_survival,
)


def make_synthetic_multiomics_cohort(seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create an 80-patient synthetic RNA + CNV + survival cohort with injected prognostic signal."""
    rng = np.random.default_rng(seed)
    n = 80
    samples = [f"TCGA-00-{i:04d}-01" for i in range(1, n + 1)]
    stage_bin = np.array([0] * (n // 2) + [1] * (n // 2), dtype=float)
    stage_str = ["Stage IA" if s == 0 else "Stage IIIB" for s in stage_bin]

    # Latent progression factor driving RNA markers, CNV drivers, and hazard
    latent = rng.normal(0, 1, size=n) + np.where(stage_bin == 1, 1.5, -1.5)
    hazard = np.exp(0.75 * latent)
    os_time = np.maximum(45.0, rng.exponential(scale=1200.0 / hazard))
    os_event = (os_time < 1600.0).astype(int)

    rna_genes = [f"RNA__GENE_{i:02d}" for i in range(30)]
    cnv_genes = [f"CNV__LOCUS_{i:02d}" for i in range(20)]

    rna_mat = rng.normal(6.0, 0.8, size=(n, len(rna_genes)))
    cnv_mat = rng.normal(0.0, 0.4, size=(n, len(cnv_genes)))

    # Inject strong signal in first 5 RNA and first 4 CNV features
    for j in range(5):
        rna_mat[:, j] += 1.2 * latent
    for j in range(4):
        cnv_mat[:, j] += 0.9 * latent

    rna_df = pd.DataFrame(rna_mat, index=samples, columns=rna_genes)
    cnv_df = pd.DataFrame(cnv_mat, index=samples, columns=cnv_genes)
    surv_df = pd.DataFrame({"sample": samples, "OS": os_event, "OS.time": os_time})
    clin_df = pd.DataFrame({"sampleID": samples, "pathologic_stage": stage_str})

    return align_multiomics_cohort(rna_df, cnv_df, surv_df, clin_df, min_os_days=30)


def test_parse_stage_binary() -> None:
    s = pd.Series(["Stage IA", "Stage IIB", "Stage IIIA", "Stage IV", "Unknown"])
    res = parse_stage_binary(s)
    assert list(res.iloc[:4]) == [0.0, 0.0, 1.0, 1.0]
    assert np.isnan(res.iloc[4])


def test_end_to_end_ml_shap_survival_and_plots(tmp_path: Path) -> None:
    rna_df, cnv_df, meta_df = make_synthetic_multiomics_cohort(seed=42)
    assert len(meta_df) == 80

    metrics_df, oof_df = run_cv_benchmarks(
        rna_df, cnv_df, meta_df, top_rna=20, top_cnv=15, n_splits=5, seed=42
    )
    assert len(metrics_df) == 9  # 3 modalities x 3 models
    assert metrics_df["roc_auc"].max() > 0.80

    shap_df, _ = compute_shap_biomarkers(
        rna_df, cnv_df, meta_df, top_rna=20, top_cnv=15, seed=42
    )
    assert not shap_df.empty
    assert abs(shap_df["shap_share_pct"].sum() - 100.0) < 1e-4

    surv_oof, cox_df = run_lifelines_survival(
        rna_df, cnv_df, meta_df, shap_df, top_n_cox=8, penalizer=0.1, n_splits=5, seed=42
    )
    assert len(surv_oof) == 80
    assert surv_oof["oof_c_index"].iloc[0] > 0.60
    assert len(cox_df) == 8

    plot_modality_benchmark(metrics_df, oof_df, "SYNTH-LUSC", tmp_path / "bench.png")
    plot_shap_biomarkers(shap_df, "SYNTH-LUSC", tmp_path / "shap.png")
    plot_survival_stratification(surv_oof, "SYNTH-LUSC", tmp_path / "surv.png")
    plot_cox_forest(cox_df, "SYNTH-LUSC", tmp_path / "cox.png")
    for fname in ("bench.png", "shap.png", "surv.png", "cox.png"):
        assert (tmp_path / fname).stat().st_size > 1000
