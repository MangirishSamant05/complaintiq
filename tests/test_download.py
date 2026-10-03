"""Tests for download helpers (no internet needed)."""
import zipfile
from pathlib import Path

from src.data.download import human_size, verify_zip


def test_human_size():
    assert human_size(None) == "unknown"
    assert human_size(512) == "512.0 B"
    assert human_size(1536) == "1.5 KB"
    assert human_size(5 * 1024**3) == "5.0 GB"


def test_verify_zip_accepts_valid_zip(tmp_path: Path):
    good = tmp_path / "good.zip"
    with zipfile.ZipFile(good, "w") as zf:
        zf.writestr("complaints.csv", "a,b\n1,2\n")
    assert verify_zip(good)


def test_verify_zip_rejects_truncated_zip(tmp_path: Path):
    good = tmp_path / "good.zip"
    with zipfile.ZipFile(good, "w") as zf:
        zf.writestr("complaints.csv", "a,b\n" + "1,2\n" * 1000)
    truncated = tmp_path / "truncated.zip"
    truncated.write_bytes(good.read_bytes()[:-30])  # simulate an interrupted download
    assert not verify_zip(truncated)


def test_verify_zip_rejects_non_zip(tmp_path: Path):
    html = tmp_path / "error_page.zip"
    html.write_text("<html>Access denied</html>", encoding="utf-8")
    assert not verify_zip(html)