"""Interim Parquet -> clean, de-duplicated modelling data.

Steps (training pool):
  1. quality filters   : missing label/id/date, duplicate IDs, very short texts
  2. exact dedup       : identical text after normalisation -> keep the earliest
  3. cap per product   : at most N rows per class (random, reproducible)
  4. near-dup groups   : MinHash + LSH, so near-identical complaints can later be
                         kept in the SAME split (prevents train/test leakage)
Drift set (2026): same filters, minus any text already seen in the training pool,
natural class mix kept (no cap).

    python -m src.data.clean

Outputs:
    data/processed/clean.parquet
    data/processed/drift.parquet
    data/sample/complaints_sample.csv
    reports/metrics/clean_summary.json
"""
from __future__ import annotations

import json
import time

import pandas as pd

from src.config import settings
from src.data.dedup import MinHasher, near_duplicate_groups, normalize_series, text_hash
from src.data.ingest import output_dir as interim_dir
from src.logging_utils import get_logger

logger = get_logger(__name__)

KEEP_COLUMNS = [
    "complaint_id", "date_received", "product", "sub_product", "issue", "sub_issue",
    "narrative", "company", "state", "submitted_via", "company_response",
    "timely_response", "source_file",
]
SAMPLE_COLUMNS = ["complaint_id", "date_received", "product", "issue", "company",
                  "state", "narrative"]


def load_role(role: str) -> pd.DataFrame:
    """Load every interim file for one role, adding n_chars and text_hash per file
    (normalising file by file keeps peak memory low)."""
    files = sorted(interim_dir().glob("*.parquet"))
    if not files:
        raise SystemExit(f"No Parquet files in {interim_dir()}. Run src.data.ingest first.")
    parts = []
    for path in files:
        df = pd.read_parquet(path, columns=KEEP_COLUMNS + ["role"])
        df = df[df["role"] == role].drop(columns="role")
        if df.empty:
            continue
        t0 = time.perf_counter()
        df["n_chars"] = df["narrative"].str.len()
        df["text_hash"] = text_hash(normalize_series(df["narrative"]))
        logger.info("Loaded %s: %s rows (hashed in %.1fs)", path.name, f"{len(df):,}",
                    time.perf_counter() - t0)
        parts.append(df)
    if not parts:
        raise SystemExit(f"No rows with role '{role}' found.")
    return pd.concat(parts, ignore_index=True)


def quality_filter(df: pd.DataFrame, min_chars: int) -> tuple[pd.DataFrame, dict]:
    stats = {"input": len(df)}
    df = df.dropna(subset=["product", "complaint_id", "date_received"])
    stats["after_drop_missing"] = len(df)
    df = df.drop_duplicates(subset="complaint_id")
    stats["after_drop_duplicate_ids"] = len(df)
    df = df[df["n_chars"] >= min_chars]
    stats["after_min_chars"] = len(df)
    return df.reset_index(drop=True), stats


def exact_dedup(df: pd.DataFrame) -> pd.DataFrame:
    """Keep the earliest complaint for each normalised text; record group size."""
    df = df.sort_values(["date_received", "complaint_id"], kind="stable")
    df["dup_count"] = df.groupby("text_hash")["text_hash"].transform("size").astype("int64")
    return df.drop_duplicates(subset="text_hash", keep="first").reset_index(drop=True)


def cap_per_product(df: pd.DataFrame, cap: int, seed: int) -> pd.DataFrame:
    """Shuffle once, then take the first `cap` rows of each product."""
    shuffled = df.sample(frac=1.0, random_state=seed)
    return shuffled.groupby("product", sort=False).head(cap).reset_index(drop=True)


def add_near_dup_groups(df: pd.DataFrame) -> pd.DataFrame:
    cfg = settings.cleaning.near_dup
    t0 = time.perf_counter()
    hasher = MinHasher(cfg.num_perm, cfg.shingle_words, seed=settings.project.seed)
    sigs = hasher.signatures(normalize_series(df["narrative"]))
    logger.info("MinHash signatures for %s texts in %.1fs", f"{len(df):,}",
                time.perf_counter() - t0)
    t0 = time.perf_counter()
    df["dup_group"] = near_duplicate_groups(sigs, cfg.bands, cfg.threshold)
    df["dup_group_size"] = df.groupby("dup_group")["dup_group"].transform("size").astype("int64")
    logger.info("LSH grouping in %.1fs", time.perf_counter() - t0)
    return df


def product_table(before: pd.DataFrame, unique: pd.DataFrame, final: pd.DataFrame) -> pd.DataFrame:
    table = pd.DataFrame({
        "after_filters": before["product"].value_counts(),
        "unique_texts": unique["product"].value_counts(),
        "in_model_set": final["product"].value_counts(),
    }).fillna(0).astype("int64")
    table["exact_dup_rate"] = (1 - table["unique_texts"] / table["after_filters"]).round(3)
    return table.sort_values("after_filters", ascending=False)


def main() -> None:
    cfg = settings.cleaning
    seed = settings.project.seed
    out_dir = settings.paths.processed
    out_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()

    # ---------------- training pool
    pool = load_role("train_pool")
    pool, train_funnel = quality_filter(pool, cfg.min_chars)
    pool_hashes = set(pool["text_hash"].tolist())        # for drift leakage check
    unique = exact_dedup(pool)
    train_funnel["after_exact_dedup"] = len(unique)
    model_set = cap_per_product(unique, cfg.max_per_product, seed)
    train_funnel["after_cap"] = len(model_set)
    model_set = add_near_dup_groups(model_set)
    model_set = model_set.sort_values(["date_received", "complaint_id"]).reset_index(drop=True)
    by_product = product_table(pool, unique, model_set)
    model_set.to_parquet(out_dir / "clean.parquet", index=False)

    # ---------------- drift set
    drift = load_role("drift")
    drift, drift_funnel = quality_filter(drift, cfg.min_chars)
    seen = drift["text_hash"].isin(pool_hashes)
    drift = drift[~seen]
    drift_funnel["after_remove_seen_in_train"] = len(drift)
    unknown = ~drift["product"].isin(set(model_set["product"]))
    drift = drift[~unknown]
    drift_funnel["after_remove_unknown_products"] = len(drift)
    drift = exact_dedup(drift)
    drift_funnel["after_exact_dedup"] = len(drift)
    drift.to_parquet(out_dir / "drift.parquet", index=False)

    # ---------------- small sample committed to Git
    sample = (model_set.sample(frac=1.0, random_state=seed)
              .groupby("product", sort=False).head(cfg.sample_per_product)
              .sort_values(["product", "complaint_id"]))
    settings.paths.sample.mkdir(parents=True, exist_ok=True)
    sample_path = settings.paths.sample / "complaints_sample.csv"
    sample[SAMPLE_COLUMNS].to_csv(sample_path, index=False, encoding="utf-8")

    # ---------------- report
    groups = model_set.drop_duplicates("dup_group")
    summary = {
        "train_funnel": train_funnel,
        "drift_funnel": drift_funnel,
        "model_set_rows": len(model_set),
        "near_dup_groups": int(model_set["dup_group"].nunique()),
        "rows_in_multi_member_groups": int((model_set["dup_group_size"] > 1).sum()),
        "largest_group_size": int(groups["dup_group_size"].max()),
        "drift_rows": len(drift),
        "drift_products": drift["product"].value_counts().to_dict(),
        "by_product": by_product.reset_index(names="product").to_dict(orient="records"),
        "seconds": round(time.perf_counter() - start, 1),
    }
    report_path = settings.paths.reports / "metrics" / "clean_summary.json"
    report_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    print("\nTraining pool funnel:")
    for step, n in train_funnel.items():
        print(f"  {step:<28} {n:>10,}")
    print("\nPer product:")
    print(by_product.to_string())
    print(f"\nNear-duplicate groups      : {summary['near_dup_groups']:,} "
          f"for {len(model_set):,} rows")
    print(f"Rows sharing a group       : {summary['rows_in_multi_member_groups']:,}")
    print(f"Largest group              : {summary['largest_group_size']:,} rows")
    print("\nDrift funnel:")
    for step, n in drift_funnel.items():
        print(f"  {step:<30} {n:>10,}")
    print(f"\nWrote {out_dir / 'clean.parquet'}")
    print(f"Wrote {out_dir / 'drift.parquet'}")
    print(f"Wrote {sample_path} ({len(sample)} rows)")
    print(f"Summary: {report_path}  ({summary['seconds']}s total)")


if __name__ == "__main__":
    main()