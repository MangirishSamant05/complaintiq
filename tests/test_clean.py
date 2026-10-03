"""Tests for cleaning steps."""
import pandas as pd

from src.data.clean import cap_per_product, exact_dedup, quality_filter


def make_df() -> pd.DataFrame:
    return pd.DataFrame({
        "complaint_id": pd.array([1, 2, 3, 4, 4, None], dtype="Int64"),
        "date_received": pd.to_datetime(
            ["2024-05-01", "2024-04-01", "2024-06-01", "2024-07-01", "2024-07-01", "2024-07-02"]),
        "product": ["A", "A", "B", "B", "B", "B"],
        "narrative": ["x" * 80, "x" * 80, "y" * 80, "short", "short", "z" * 80],
        "n_chars": [80, 80, 80, 5, 5, 80],
        "text_hash": pd.array([10, 10, 20, 30, 30, 40], dtype="uint64"),
    })


def test_quality_filter_funnel():
    df, stats = quality_filter(make_df(), min_chars=50)
    assert stats == {"input": 6, "after_drop_missing": 5,
                     "after_drop_duplicate_ids": 4, "after_min_chars": 3}
    assert sorted(df["complaint_id"].tolist()) == [1, 2, 3]


def test_exact_dedup_keeps_earliest_and_counts():
    df, _ = quality_filter(make_df(), min_chars=50)
    out = exact_dedup(df)
    assert len(out) == 2
    kept_a = out[out["product"] == "A"].iloc[0]
    assert kept_a["complaint_id"] == 2          # 2024-04-01 is earlier than id 1
    assert kept_a["dup_count"] == 2


def test_cap_per_product_is_capped_and_reproducible():
    df = pd.DataFrame({"product": ["A"] * 50 + ["B"] * 5, "complaint_id": range(55)})
    first = cap_per_product(df, cap=10, seed=42)
    second = cap_per_product(df, cap=10, seed=42)
    assert first["product"].value_counts().to_dict() == {"A": 10, "B": 5}
    assert first["complaint_id"].tolist() == second["complaint_id"].tolist()