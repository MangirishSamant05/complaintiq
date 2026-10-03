"""Look inside a downloaded export zip WITHOUT loading it all into memory.

Examples (run from the project root):
    python -m src.data.inspect_raw                       # first zip in data/raw
    python -m src.data.inspect_raw --file CCDB_Export_16_March_2026.zip
    python -m src.data.inspect_raw --count               # also count all rows (slower)
"""
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import pandas as pd

from src.config import settings
from src.data.download import human_size
from src.logging_utils import get_logger

logger = get_logger(__name__)

SAMPLE_ROWS = 5_000
COUNT_CHUNK_ROWS = 100_000


def pick_zip(file_name: str | None) -> Path:
    if file_name:
        path = settings.paths.raw / file_name
        if not path.exists():
            raise SystemExit(f"Not found: {path}")
        return path
    zips = sorted(settings.paths.raw.glob("*.zip"))
    if not zips:
        raise SystemExit(f"No .zip files in {settings.paths.raw}. Run the download step first.")
    return zips[0]


def largest_member(zf: zipfile.ZipFile) -> zipfile.ZipInfo:
    return max(zf.infolist(), key=lambda info: info.file_size)


def read_csv_member(zf: zipfile.ZipFile, member: str, **kwargs) -> pd.DataFrame:
    """Read a CSV straight out of the zip (no extraction to disk)."""
    with zf.open(member) as fh:
        return pd.read_csv(fh, dtype=str, encoding="utf-8",
                           encoding_errors="replace", **kwargs)


def count_rows(zf: zipfile.ZipFile, member: str) -> None:
    """Stream the whole CSV in chunks, reading only 2 columns, to count rows."""
    text_col, label_col = settings.data.text_column, settings.data.label_column
    total = with_text = 0
    labels: pd.Series | None = None
    with zf.open(member) as fh:
        reader = pd.read_csv(fh, dtype=str, usecols=[text_col, label_col],
                             chunksize=COUNT_CHUNK_ROWS, encoding="utf-8",
                             encoding_errors="replace")
        for i, chunk in enumerate(reader, start=1):
            has_text = chunk[text_col].fillna("").str.strip().ne("")
            total += len(chunk)
            with_text += int(has_text.sum())
            counts = chunk.loc[has_text, label_col].value_counts()
            labels = counts if labels is None else labels.add(counts, fill_value=0)
            logger.info("...chunk %d: %s rows so far", i, f"{total:,}")
    print(f"\nTotal rows              : {total:,}")
    print(f"Rows with a narrative   : {with_text:,} ({with_text / max(total, 1):.1%})")
    if labels is not None:
        print("\nNarrative rows per product:")
        print(labels.sort_values(ascending=False).astype(int).to_string())


def inspect(path: Path, do_count: bool) -> None:
    text_col, label_col = settings.data.text_column, settings.data.label_column
    print(f"\n=== {path.name} ({human_size(path.stat().st_size)} on disk) ===")

    with zipfile.ZipFile(path) as zf:
        print("\nFiles inside the zip:")
        for info in zf.infolist():
            print(f"  {info.filename:<50} {human_size(info.file_size):>10} uncompressed")

        member = largest_member(zf)
        if not member.filename.lower().endswith(".csv"):
            with zf.open(member) as fh:
                head = fh.read(1500).decode("utf-8", errors="replace")
            print(f"\nLargest member is not a CSV. First 1500 characters:\n{head}")
            print("\n>>> Paste this output to Claude so the ingest step can be adapted.")
            return

        df = read_csv_member(zf, member.filename, nrows=SAMPLE_ROWS)

        print(f"\nColumns ({len(df.columns)}), with % non-empty in first {len(df):,} rows:")
        for col in df.columns:
            filled = df[col].fillna("").str.strip().ne("").mean()
            print(f"  {col:<40} {filled:6.1%}")

        missing = [c for c in (text_col, label_col) if c not in df.columns]
        if missing:
            print(f"\n!!! Expected column(s) not found: {missing}")
            print("    Fix text_column / label_column in configs/config.yaml to match the names above.")
            return

        narratives = df[text_col].dropna()
        narratives = narratives[narratives.str.strip() != ""]
        print(f"\nNarratives in sample : {len(narratives):,} of {len(df):,}")
        if len(narratives):
            lengths = narratives.str.len()
            print(f"Narrative length     : median {int(lengths.median())} chars, "
                  f"95th pct {int(lengths.quantile(0.95))}, max {int(lengths.max())}")
            print("\nExample narrative (first 400 chars):")
            print("  " + narratives.iloc[0][:400].replace("\n", " ") + " ...")

        print(f"\nTop products in sample (column '{label_col}'):")
        print(df[label_col].value_counts().head(15).to_string())

        if do_count:
            print("\nCounting all rows (streams the whole file, ~1-3 minutes)...")
            count_rows(zf, member.filename)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a downloaded CFPB export zip.")
    parser.add_argument("--file", help="zip file name inside data/raw (default: first one)")
    parser.add_argument("--count", action="store_true", help="count all rows and narratives")
    args = parser.parse_args()
    inspect(pick_zip(args.file), args.count)


if __name__ == "__main__":
    main()