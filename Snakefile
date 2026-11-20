"""Reproducible Snakemake workflow for multi-omics cancer biomarker ML, SHAP, and lifelines survival."""

configfile: "config/config.yaml"

DATA_DIR = config.get("output_dirs", {}).get("data", "data/raw")
RESULTS_DIR = config.get("output_dirs", {}).get("results", "results")
FIGURES_DIR = config.get("output_dirs", {}).get("figures", "figures")


rule all:
    input:
        f"{RESULTS_DIR}/cv_benchmark_metrics.tsv",
        f"{RESULTS_DIR}/oof_predictions.tsv",
        f"{RESULTS_DIR}/shap_biomarker_importance.tsv",
        f"{RESULTS_DIR}/survival_oof_scores.tsv",
        f"{RESULTS_DIR}/survival_cox_biomarkers.tsv",
        f"{FIGURES_DIR}/modality_model_benchmark.png",
        f"{FIGURES_DIR}/shap_biomarkers_summary.png",
        f"{FIGURES_DIR}/survival_risk_stratification.png",
        f"{FIGURES_DIR}/cox_hazard_forest.png",


rule download_multiomics:
    output:
        rna=f"{DATA_DIR}/aligned_rna.tsv.gz",
        cnv=f"{DATA_DIR}/aligned_cnv.tsv.gz",
        meta=f"{DATA_DIR}/aligned_clinical.tsv",
        manifest=f"{DATA_DIR}/download_manifest.json",
    params:
        config_path=config.get("_config_path", "config/config.yaml"),
    shell:
        """
        python workflow/scripts/download_multiomics.py \
            --config {params.config_path} \
            --out-dir {DATA_DIR}
        """


rule train_and_explain:
    input:
        rna=rules.download_multiomics.output.rna,
        cnv=rules.download_multiomics.output.cnv,
        meta=rules.download_multiomics.output.meta,
    output:
        metrics=f"{RESULTS_DIR}/cv_benchmark_metrics.tsv",
        oof=f"{RESULTS_DIR}/oof_predictions.tsv",
        shap_imp=f"{RESULTS_DIR}/shap_biomarker_importance.tsv",
        surv_oof=f"{RESULTS_DIR}/survival_oof_scores.tsv",
        surv_cox=f"{RESULTS_DIR}/survival_cox_biomarkers.tsv",
    params:
        top_rna=config["features"]["top_rna_features"],
        top_cnv=config["features"]["top_cnv_features"],
        n_splits=config["ml"]["n_cv_splits"],
        top_shap_cox=config["survival"]["top_shap_features_for_cox"],
        penalizer=config["survival"]["cox_penalizer"],
        seed=config["random_seed"],
    shell:
        """
        python workflow/scripts/train_and_explain.py \
            --rna {input.rna} \
            --cnv {input.cnv} \
            --meta {input.meta} \
            --top-rna {params.top_rna} \
            --top-cnv {params.top_cnv} \
            --n-splits {params.n_splits} \
            --top-shap-cox {params.top_shap_cox} \
            --penalizer {params.penalizer} \
            --seed {params.seed} \
            --out-metrics {output.metrics} \
            --out-oof {output.oof} \
            --out-shap {output.shap_imp} \
            --out-surv-oof {output.surv_oof} \
            --out-surv-cox {output.surv_cox}
        """


rule plot_summary:
    input:
        metrics=rules.train_and_explain.output.metrics,
        oof=rules.train_and_explain.output.oof,
        shap_imp=rules.train_and_explain.output.shap_imp,
        surv_oof=rules.train_and_explain.output.surv_oof,
        surv_cox=rules.train_and_explain.output.surv_cox,
    output:
        bench_fig=f"{FIGURES_DIR}/modality_model_benchmark.png",
        shap_fig=f"{FIGURES_DIR}/shap_biomarkers_summary.png",
        surv_fig=f"{FIGURES_DIR}/survival_risk_stratification.png",
        cox_fig=f"{FIGURES_DIR}/cox_hazard_forest.png",
    params:
        cohort_name=config["cohort"]["name"],
    shell:
        """
        python workflow/scripts/plot_summary.py \
            --metrics {input.metrics} \
            --oof {input.oof} \
            --shap {input.shap_imp} \
            --surv-oof {input.surv_oof} \
            --surv-cox {input.surv_cox} \
            --cohort "{params.cohort_name}" \
            --out-dir {FIGURES_DIR}
        """
