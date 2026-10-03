"""Raw zips -> Parquet. Keeps only complaints that have a narrative.

Reads each CSV straight out of its zip in 100k-row chunks (low memory),
keeps the configured columns, gives them clean names and proper types,
and writes one Parquet file per export to data/interim/narratives/.

    python -m src.data.ingest            # process every configured zip in data/raw
    python -m src.data.ingest --force    # rebuild Parquet files that already exist
"""
from __future__ import annotations

import argparse
import json
import time
import zipfile
from collections.abc import Iterator
from pathlib import Path

import pandas as pd

from src.config import settings
from src.data.download import human_size
from src.data.inspect_raw import largest_member
from src.logging_utils import get_logger

logger = get_logger(__name__)

CHUNK_ROWS = 100_000
SUMMARY_FILE = "ingest_summary.json"


def output_dir() -> Path:
    return settings.paths.interim / "narratives"


def check_columns(header: list[str]) -> None:
    """Fail early, with a useful message, if the CSV lacks a column we need."""
    missing = [col for col in settings.data.columns if col not in header]
    if missing:
        raise ValueError(
            f"CSV is missing expected column(s) {missing}.\n"
            f"Columns found: {header}\n"
            "Fix the 'columns' mapping in configs/config.yaml."
        )


def iter_narrative_chunks(zip_path: Path) -> Iterator[tuple[int, pd.DataFrame]]:
    """Yield (rows_read, rows_with_narrative) for every chunk of the CSV."""
    text_col = settings.data.text_column
    with zipfile.ZipFile(zip_path) as zf:
        member = largest_member(zf).filename

        with zf.open(member) as fh:                      # header only
            check_columns(pd.read_csv(fh, nrows=0).columns.tolist())

        with zf.open(member) as fh:
            reader = pd.read_csv(
                fh,
                dtype=str,
                usecols=list(settings.data.columns),   # parse only what we keep
                chunksize=CHUNK_ROWS,
                encoding="utf-8",
                encoding_errors="replace",
                on_bad_lines="warn",
            )
            for chunk in reader:
                text = chunk[text_col].fillna("").str.strip()
                has_text = text.ne("")
                kept = chunk.loc[has_text].copy()
                kept[text_col] = text[has_text]
                yield len(chunk), kept


def tidy(df: pd.DataFrame, source_file: str, role: str) -> pd.DataFrame:
    """Rename to clean column names, fix types, add lineage columns."""
    df = df.rename(columns=settings.data.columns)
    df = df[list(settings.data.columns.values())]            # fixed column order
    df["date_received"] = pd.to_datetime(df["date_received"], errors="coerce", format="mixed")
    df["complaint_id"] = pd.to_numeric(df["complaint_id"], errors="coerce").astype("Int64")
    df["source_file"] = source_file   # lineage: which export each row came from
    df["role"] = role                 # "train_pool" or "drift"
    return df.reset_index(drop=True)


def ingest_file(zip_path: Path, role: str, out_dir: Path, force: bool = False) -> dict:
    """Convert one export zip into one Parquet file. Returns a summary dict."""
    out_path = out_dir / f"{zip_path.stem}.parquet"

    if out_path.exists() and not force:
        logger.info("Already ingested, skipping (use --force to rebuild): %s", out_path.name)
        existing = pd.read_parquet(out_path, columns=["product"])  # read ONE column only
        return {
            "file": zip_path.name, "role": role, "skipped": True,
            "narratives_kept": len(existing),
            "products": existing["product"].value_counts().to_dict(),
        }

    start = time.perf_counter()
    rows_read, parts = 0, []
    for n_read, kept in iter_narrative_chunks(zip_path):
        rows_read += n_read
        parts.append(kept)
        logger.info("%s: %s rows read, %s with narrative",
                    zip_path.name, f"{rows_read:,}", f"{sum(len(p) for p in parts):,}")

    combined = (pd.concat(parts, ignore_index=True) if parts
                else pd.DataFrame(columns=list(settings.data.columns)))
    df = tidy(combined, zip_path.name, role)

    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(out_path.name + ".tmp")
    df.to_parquet(tmp_path, engine="pyarrow", compression="snappy", index=False)
    tmp_path.replace(out_path)          # only a complete file ever gets the real name

    with zipfile.ZipFile(zip_path) as zf:
        csv_bytes = largest_member(zf).file_size

    summary = {
        "file": zip_path.name,
        "role": role,
        "skipped": False,
        "rows_read": rows_read,
        "narratives_kept": len(df),
        "pct_with_narrative": round(len(df) / max(rows_read, 1), 4),
        "date_min": str(df["date_received"].min().date()) if len(df) else None,
        "date_max": str(df["date_received"].max().date()) if len(df) else None,
        "unparsed_dates": int(df["date_received"].isna().sum()),
        "missing_ids": int(df["complaint_id"].isna().sum()),
        "zip_bytes": zip_path.stat().st_size,
        "csv_bytes": csv_bytes,
        "parquet_bytes": out_path.stat().st_size,
        "seconds": round(time.perf_counter() - start, 1),
        "products": df["product"].value_counts().to_dict(),
    }
    logger.info("Wrote %s: %s rows, %s, %.1fs", out_path.name, f"{len(df):,}",
                human_size(summary["parquet_bytes"]), summary["seconds"])
    return summary


def count_cross_file_duplicates(out_dir: Path) -> int:
    """Complaint IDs appearing in more than one export (reads ONE column per file)."""
    files = sorted(out_dir.glob("*.parquet"))
    if not files:
        return 0
    ids = pd.concat([pd.read_parquet(f, columns=["complaint_id"]) for f in files],
                    ignore_index=True)
    return int(ids["complaint_id"].duplicated().sum())


def print_report(summaries: list[dict], duplicates: int, missing: list[str]) -> None:
    print("\nPer file:")
    for s in summaries:
        if s["skipped"]:
            print(f"  [{s['role']:<10}] {s['file']:<55} {s['narratives_kept']:>9,} (already done)")
        else:
            print(f"  [{s['role']:<10}] {s['file']:<55} {s['narratives_kept']:>9,} of "
                  f"{s['rows_read']:>9,} rows | {s['date_min']}..{s['date_max']} | "
                  f"CSV {human_size(s['csv_bytes'])} -> Parquet {human_size(s['parquet_bytes'])} "
                  f"| {s['seconds']}s")

    for role in ("train_pool", "drift"):
        totals: dict[str, int] = {}
        for s in summaries:
            if s["role"] == role:
                for product, n in s["products"].items():
                    totals[product] = totals.get(product, 0) + n
        if totals:
            print(f"\nNarratives per product ({role}), total {sum(totals.values()):,}:")
            for product, n in sorted(totals.items(), key=lambda kv: -kv[1]):
                print(f"  {n:>9,}  {product}")

    print(f"\nComplaint IDs found in more than one file: {duplicates:,}")
    if missing:
        print(f"Missing zips (not downloaded yet): {missing}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest CFPB export zips into Parquet.")
    parser.add_argument("--force", action="store_true", help="rebuild existing Parquet files")
    args = parser.parse_args()

    jobs = ([(name, "train_pool") for name in settings.data.train_files]
            + [(name, "drift") for name in settings.data.drift_files])
    out_dir = output_dir()
    summaries, missing = [], []

    for name, role in jobs:
        zip_path = settings.paths.raw / name
        if not zip_path.exists():
            logger.warning("Not downloaded yet, skipping: %s", name)
            missing.append(name)
            continue
        try:
            summaries.append(ingest_file(zip_path, role, out_dir, force=args.force))
        except (ValueError, zipfile.BadZipFile) as exc:
            raise SystemExit(f"Failed on {name}: {exc}") from exc

    if not summaries:
        raise SystemExit(f"No configured zips found in {settings.paths.raw}.")

    duplicates = count_cross_file_duplicates(out_dir)
    report = {"files": summaries, "missing_files": missing,
              "cross_file_duplicate_ids": duplicates}
    report_path = settings.paths.reports / "metrics" / SUMMARY_FILE
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print_report(summaries, duplicates, missing)
    print(f"\nSummary saved to {report_path}")


if __name__ == "__main__":
    main()