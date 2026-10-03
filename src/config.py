"""Central configuration for ComplaintIQ.

Loads configs/config.yaml (project settings) and .env (machine-specific
settings), validates them with Pydantic, and exposes one `settings` object.

Usage anywhere in the project:
    from src.config import settings
    print(settings.paths.raw)
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError

# src/config.py -> parents[0] is src/, parents[1] is the project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


class ProjectConfig(BaseModel):
    name: str
    seed: int = 42


class PathsConfig(BaseModel):
    raw: Path
    interim: Path
    processed: Path
    sample: Path
    gold: Path
    models: Path
    reports: Path
    logs: Path


class DataConfig(BaseModel):
    base_url: str
    train_files: list[str]
    drift_files: list[str]
    text_column: str
    label_column: str


class Settings(BaseModel):
    project: ProjectConfig
    paths: PathsConfig
    data: DataConfig
    env: Literal["dev", "test", "prod"] = "dev"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"

    def ensure_dirs(self) -> None:
        """Create every folder in `paths` (plus a .gitkeep so Git tracks it)."""
        for folder in self.paths.model_dump().values():
            folder.mkdir(parents=True, exist_ok=True)
            (folder / ".gitkeep").touch(exist_ok=True)
        # report sub-folders used later
        for sub in ("metrics", "figures"):
            sub_dir = self.paths.reports / sub
            sub_dir.mkdir(parents=True, exist_ok=True)
            (sub_dir / ".gitkeep").touch(exist_ok=True)


@lru_cache(maxsize=None)
def get_settings(config_path: Path = DEFAULT_CONFIG_PATH) -> Settings:
    """Read, validate and cache the settings. Fails loudly on bad config."""
    load_dotenv(PROJECT_ROOT / ".env")  # silently skipped if .env is missing

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    # Turn relative paths into absolute paths anchored at the project root,
    # so scripts work no matter which folder you run them from.
    raw["paths"] = {k: PROJECT_ROOT / v for k, v in (raw.get("paths") or {}).items()}

    # Machine-specific values come from environment variables / .env
    raw["env"] = os.getenv("CIQ_ENV", "dev").lower()
    raw["log_level"] = os.getenv("CIQ_LOG_LEVEL", "INFO").upper()

    try:
        return Settings(**raw)
    except ValidationError as exc:
        raise ValueError(f"Invalid configuration in {config_path}:\n{exc}") from exc


settings = get_settings()


if __name__ == "__main__":
    # `python -m src.config` -> create folders and print a summary
    settings.ensure_dirs()
    print(f"Project      : {settings.project.name}")
    print(f"Environment  : {settings.env}")
    print(f"Log level    : {settings.log_level}")
    print(f"Seed         : {settings.project.seed}")
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Raw data dir : {settings.paths.raw}")
    print(f"Train files  : {len(settings.data.train_files)}")
    print(f"Drift files  : {len(settings.data.drift_files)}")
    print("Folders created/verified OK")