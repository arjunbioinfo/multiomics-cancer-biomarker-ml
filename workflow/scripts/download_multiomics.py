#!/usr/bin/env python3
"""Download and align TCGA multi-omics (RNA-seq + GISTIC2 CNV) and clinical/survival data from UCSC Xena."""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests
import yaml

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def xena_download_url(host: str, dataset_id: str) -> str:
    """Build a UCSC Xena S3 direct download URL from host and dataset path."""
    return f"{host.rstrip('/')}/download/{quote(dataset_id, safe='')}"


def download_file_with_fallback(host: str, dataset_id: str, dest: Path, timeout: int = 180) -> tuple[str, int]:
    """Download a UCSC Xena dataset, trying .gz and non-.gz fallback if needed."""
    if dest.exists() and dest.stat().st_size > 0:
        logger.info("Using cached file: %s (%d bytes)", dest, dest.stat().st_size)
        return xena_download_url(host, dataset_id), dest.stat().st_size

    dest.parent.mkdir(parents=True, exist_ok=True)
    candidates = [dataset_id]
    if dataset_id.endswith(".gz"):
        candidates.append(dataset_id[:-3])
    else:
        candidates.append(dataset_id + ".gz")

    last_err: Exception | None = None
    for ds in candidates:
        url = xena_download_url(host, ds)
        logger.info("Downloading %s -> %s", url, dest)
        tmp = dest.with_suffix(dest.suffix + ".part")
        try:
            with requests.get(url, stream=True, timeout=timeout, headers={"User-Agent": "multiomics-cancer-biomarker-ml/1.0"}) as r:
                r.raise_for_status()
                with open(tmp, "wb") as fh:
                    for chunk in r.iter_content(chunk_size=1024 * 256):
                        if chunk:
                            fh.write(chunk)
            tmp.rename(dest)
            return url, dest.stat().st_size
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            if tmp.exists():
                tmp.unlink()
            logger.warning("Download attempt failed for %s: %s", url, exc)

    raise RuntimeError(f"Failed to download dataset {dataset_id} from {host}: {last_err}")


def parse_stage_binary(stage_series: pd.Series) -> pd.Series:
    """Map AJCC/FIGO stage strings to 0 (Early: Stage I-II) or 1 (Advanced: Stage III-IV)."""
    out = []
    for val in stage_series.astype(str):
        s = val.upper().strip()
        if re.search(r"STAGE\s*(III|IV)", s):
            out.append(1.0)
        elif re.search(r"STAGE\s*(I($|[^IV])|II($|[^I]))", s):
            out.append(0.0)
        else:
            out.append(np.nan)
    return pd.Series(out, index=stage_series.index, dtype=float)


def load_omics_matrix(path: Path, prefix: str) -> pd.DataFrame:
    """Load a gene-by-sample UCSC Xena matrix and return sample-by-feature DataFrame."""
    df = pd.read_csv(path, sep="\t", index_col=0, compression="infer", low_memory=False)
    df.index = df.index.astype(str)
    df = df[~df.index.duplicated(keep="first")]
    # Keep primary tumor samples (-01) when TCGA barcodes are present
    primary_cols = [c for c in df.columns if str(c).endswith("-01")]
    if len(primary_cols) >= 20:
        df = df[primary_cols]
    df = df.apply(pd.to_numeric, errors="coerce")
    mat = df.T  # samples x genes
    mat.index = mat.index.astype(str)
    mat.columns = [f"{prefix}__{g}" for g in mat.columns]
    # Drop all-NA columns and fill residual NAs with column median
    mat = mat.dropna(axis=1, how="all")
    mat = mat.fillna(mat.median(axis=0))
    return mat


def align_multiomics_cohort(
    rna_df: pd.DataFrame,
    cnv_df: pd.DataFrame,
    surv_df: pd.DataFrame,
    clin_df: pd.DataFrame,
    min_os_days: int = 30,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Intersect primary tumors across RNA, CNV, survival, and clinical tables."""
    sample_col = next((c for c in ("sample", "Sample", "_PATIENT") if c in surv_df.columns), surv_df.columns[0])
    surv = pd.DataFrame(
        {
            "sample": surv_df[sample_col].astype(str),
            "OS": pd.to_numeric(surv_df["OS"], errors="coerce"),
            "OS_time": pd.to_numeric(surv_df["OS.time"], errors="coerce"),
        }
    ).dropna(subset=["OS", "OS_time"])
    surv = surv[surv["OS_time"] >= min_os_days].drop_duplicates(subset=["sample"]).set_index("sample")

    clin_id = clin_df.columns[0]
    stage_col = next(
        (c for c in ("pathologic_stage", "clinical_stage", "ajcc_pathologic_tumor_stage", "tumor_stage") if c in clin_df.columns),
        None,
    )
    clin = pd.DataFrame({"sample": clin_df[clin_id].astype(str)})
    clin["stage_advanced"] = parse_stage_binary(clin_df[stage_col]) if stage_col else np.nan
    clin = clin.drop_duplicates(subset=["sample"]).set_index("sample")

    common_samples = sorted(set(rna_df.index) & set(cnv_df.index) & set(surv.index))
    if len(common_samples) < 30:
        raise ValueError(f"Too few overlapping multi-omics samples ({len(common_samples)}).")

    meta = surv.loc[common_samples].join(clin, how="left")
    # Classification target: Advanced stage (Stage III-IV vs I-II) if balanced enough; else 3-year survival risk
    valid_stage = meta["stage_advanced"].dropna()
    if (valid_stage == 1).sum() >= 15 and (valid_stage == 0).sum() >= 15:
        meta["target"] = meta["stage_advanced"]
        meta["target_name"] = "Advanced_Stage_III_IV"
    else:
        meta["target"] = ((meta["OS"] == 1) & (meta["OS_time"] <= 3 * 365.25)).astype(float)
        meta["target_name"] = "Early_Mortality_3yr"

    return rna_df.loc[common_samples], cnv_df.loc[common_samples], meta.reset_index()


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and align TCGA multi-omics data from UCSC Xena.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    c_cfg = cfg["cohort"]
    host = c_cfg["xena_host"]
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rna_raw = out_dir / "raw_rna.tsv.gz"
    cnv_raw = out_dir / "raw_cnv.tsv.gz"
    surv_raw = out_dir / "raw_survival.txt"
    clin_raw = out_dir / "raw_clinical.tsv"

    u_rna, b_rna = download_file_with_fallback(host, c_cfg["rna_dataset"], rna_raw)
    u_cnv, b_cnv = download_file_with_fallback(host, c_cfg["cnv_dataset"], cnv_raw)
    u_surv, b_surv = download_file_with_fallback(host, c_cfg["survival_dataset"], surv_raw)
    u_clin, b_clin = download_file_with_fallback(host, c_cfg["clinical_dataset"], clin_raw)

    rna_df = load_omics_matrix(rna_raw, prefix="RNA")
    cnv_df = load_omics_matrix(cnv_raw, prefix="CNV")
    surv_df = pd.read_csv(surv_raw, sep="\t", compression="infer", low_memory=False)
    clin_df = pd.read_csv(clin_raw, sep="\t", compression="infer", low_memory=False)

    rna_al, cnv_al, meta_al = align_multiomics_cohort(
        rna_df, cnv_df, surv_df, clin_df, min_os_days=int(c_cfg.get("min_os_days", 30))
    )

    rna_al.to_csv(out_dir / "aligned_rna.tsv.gz", sep="\t", compression="gzip")
    cnv_al.to_csv(out_dir / "aligned_cnv.tsv.gz", sep="\t", compression="gzip")
    meta_al.to_csv(out_dir / "aligned_clinical.tsv", sep="\t", index=False)

    manifest = {
        "cohort": c_cfg["name"],
        "n_samples": int(len(meta_al)),
        "n_rna_genes": int(rna_al.shape[1]),
        "n_cnv_genes": int(cnv_al.shape[1]),
        "target_name": str(meta_al["target_name"].iloc[0]),
        "sources": {
            "rna": {"url": u_rna, "bytes": b_rna},
            "cnv": {"url": u_cnv, "bytes": b_cnv},
            "survival": {"url": u_surv, "bytes": b_surv},
            "clinical": {"url": u_clin, "bytes": b_clin},
        },
    }
    with open(out_dir / "download_manifest.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    logger.info(
        "Aligned %s: %d samples, %d RNA genes, %d CNV genes (target=%s).",
        c_cfg["name"], len(meta_al), rna_al.shape[1], cnv_al.shape[1], manifest["target_name"]
    )


if __name__ == "__main__":
    main()
