"""Tests for the ingest step, using a tiny fake export zip (no real data needed)."""
import csv
import io
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from src.data.ingest import ingest_file, iter_narrative_chunks, tidy

# Same 16 columns as the real CFPB export
COLUMNS = [
    "Date received", "Product", "Sub-product", "Issue", "Sub-issue",
    "Consumer complaint narrative", "Company public response", "Company",
    "State", "ZIP code", "Tags", "Submitted via", "Date sent to company",
    "Company response to consumer", "Timely response?", "Complaint ID",
]
NARRATIVE_WITH_NEWLINE = 'I was charged {$500.00}, twice.\nThey said "wait" by XX/XX/2024.'


def make_row(product: str, narrative: str, complaint_id: str) -> dict:
    row = {col: "" for col in COLUMNS}
    row.update({
        "Date received": "2024-11-05", "Product": product, "Issue": "Some issue",
        "Consumer complaint narrative": narrative, "Company": "BANK INC",
        "State": "CA", "Complaint ID": complaint_id,
    })
    return row


def make_zip(path: Path, rows: list[dict], columns: list[str] = COLUMNS) -> Path:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(path.stem + ".csv", buf.getvalue())
    return path


@pytest.fixture
def sample_zip(tmp_path: Path) -> Path:
    rows = [
        make_row("Credit card", NARRATIVE_WITH_NEWLINE, "101"),
        make_row("Mortgage", "", "102"),                     # no narrative
        make_row("Debt collection", "   ", "103"),           # whitespace only
        make_row("Debt collection", "  They keep calling me.  ", "104"),
    ]
    return make_zip(tmp_path / "CCDB_Export_test.zip", rows)


def test_keeps_only_rows_with_narrative(sample_zip: Path):
    chunks = list(iter_narrative_chunks(sample_zip))
    rows_read = sum(n for n, _ in chunks)
    kept = pd.concat([df for _, df in chunks])
    assert rows_read == 4
    assert len(kept) == 2


def test_tidy_renames_types_and_preserves_text(sample_zip: Path):
    kept = pd.concat([df for _, df in iter_narrative_chunks(sample_zip)])
    df = tidy(kept, "CCDB_Export_test.zip", "train_pool")

    assert {"complaint_id", "product", "narrative", "source_file", "role"} <= set(df.columns)
    assert pd.api.types.is_datetime64_any_dtype(df["date_received"])
    assert str(df["complaint_id"].dtype) == "Int64"
    assert df.loc[df["complaint_id"] == 101, "narrative"].iloc[0] == NARRATIVE_WITH_NEWLINE
    assert df.loc[df["complaint_id"] == 104, "narrative"].iloc[0] == "They keep calling me."
    assert (df["role"] == "train_pool").all()


def test_ingest_file_writes_parquet_and_skips_on_rerun(sample_zip: Path, tmp_path: Path):
    out_dir = tmp_path / "out"
    summary = ingest_file(sample_zip, "train_pool", out_dir)
    assert summary["rows_read"] == 4
    assert summary["narratives_kept"] == 2

    df = pd.read_parquet(out_dir / "CCDB_Export_test.parquet")
    assert len(df) == 2
    assert set(df["product"]) == {"Credit card", "Debt collection"}

    again = ingest_file(sample_zip, "train_pool", out_dir)
    assert again["skipped"] is True
    assert again["narratives_kept"] == 2


def test_missing_column_raises(tmp_path: Path):
    columns = [c for c in COLUMNS if c != "Product"]
    bad = make_zip(tmp_path / "bad.zip", [make_row("x", "text", "1")], columns)
    with pytest.raises(ValueError, match="Product"):
        list(iter_narrative_chunks(bad))