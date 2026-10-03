"""Tests for exact and near-duplicate detection."""
import numpy as np
import pandas as pd
import pytest

from src.data.dedup import (
    MinHasher, estimated_similarity, near_duplicate_groups,
    normalize_series, normalize_text, text_hash,
)

BASE = ("I am a victim of identity theft and the following accounts on my credit "
        "report do not belong to me. I never opened these accounts and I never "
        "authorized anyone to open them. Please block and remove these fraudulent "
        "accounts and inquiries from my report immediately as required by law. "
        "I have attached my police report and my identity theft report.")
UNRELATED = ("My mortgage servicer applied my monthly payment to the wrong month "
             "and then charged me a late fee even though I paid on time through "
             "online banking. I called them three times and nobody could explain it.")


def test_normalize_masks_amounts_digits_redactions():
    text = "On XX/XX/2024 they took {$1,250.00} from acct XXXX4321!"
    assert normalize_text(text) == "on xx xx 0 they took amt from acct xxxx0"


def test_series_matches_single_text_version():
    texts = pd.Series([BASE, UNRELATED, "Paid {$5.00} on 12/01/2024, XXXX."])
    expected = texts.map(normalize_text)
    pd.testing.assert_series_equal(normalize_series(texts), expected)


def test_exact_hash_ignores_case_amounts_and_dates():
    a = "On 01/02/2024 I paid {$100.00} to XXXX."
    b = "on 05/06/2025 i paid {$999.99} to xxxx"
    c = "On 01/02/2024 I paid {$100.00} to my landlord."
    hashes = text_hash(normalize_series(pd.Series([a, b, c])))
    assert hashes[0] == hashes[1]
    assert hashes[0] != hashes[2]


def test_minhash_similarity_ranks_texts_sensibly():
    hasher = MinHasher(num_perm=128, seed=1)
    edited = BASE.replace("police report", "FTC report")
    sig_base = hasher.signature(normalize_text(BASE))
    assert estimated_similarity(sig_base, hasher.signature(normalize_text(BASE))) == 1.0
    assert estimated_similarity(sig_base, hasher.signature(normalize_text(edited))) > 0.7
    assert estimated_similarity(sig_base, hasher.signature(normalize_text(UNRELATED))) < 0.2


def test_near_duplicate_groups_clusters_templates():
    texts = [
        BASE,
        BASE.replace("police report", "FTC report"),
        BASE.replace("immediately", "right away"),
        UNRELATED,
        "The bank closed my checking account without any notice and kept my deposit.",
    ]
    hasher = MinHasher(num_perm=64, seed=42)
    sigs = hasher.signatures([normalize_text(t) for t in texts])
    groups = near_duplicate_groups(sigs, bands=16, threshold=0.7)
    assert groups[0] == groups[1] == groups[2]
    assert len({groups[0], groups[3], groups[4]}) == 3


def test_bands_must_divide_num_perm():
    with pytest.raises(ValueError):
        near_duplicate_groups(np.zeros((3, 64), dtype=np.int64), bands=10)