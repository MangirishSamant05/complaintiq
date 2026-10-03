"""Tests for configuration loading."""
from pathlib import Path

import pytest

from src.config import PROJECT_ROOT, get_settings, settings # type: ignore


def test_settings_load():
    assert settings.project.name == "complaintiq"
    assert isinstance(settings.project.seed, int)


def test_paths_are_absolute_and_inside_project():
    for path in settings.paths.model_dump().values():
        assert path.is_absolute()
        assert PROJECT_ROOT in path.parents


def test_data_files_configured():
    assert len(settings.data.train_files) > 0
    assert all(f.endswith(".zip") for f in settings.data.train_files)
    assert settings.data.base_url.startswith("https://")


def test_missing_config_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        get_settings(tmp_path / "does_not_exist.yaml")


def test_invalid_config_raises(tmp_path: Path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("project:\n  name: x\n", encoding="utf-8")  # paths/data missing
    with pytest.raises(ValueError):
        get_settings(bad)