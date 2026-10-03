"""Download the CFPB complaint export zips listed in configs/config.yaml.

Features: size check before downloading, progress bar, automatic retries,
resume of interrupted downloads, and zip verification.

Examples (run from the project root):
    python -m src.data.download --sizes
    python -m src.data.download --only CCDB_Export_16_March_2026.zip
    python -m src.data.download
"""
from __future__ import annotations

import argparse
import time
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

from src.config import settings
from src.logging_utils import get_logger

logger = get_logger(__name__)

CHUNK_SIZE = 1024 * 1024          # write to disk 1 MB at a time
TIMEOUT = (15, 120)               # (connect, read) timeouts in seconds
MAX_RETRIES = 5
HEADERS = {"User-Agent": "complaintiq-student-project/0.1 (educational use)"}


def human_size(n_bytes: int | None) -> str:
    """1536 -> '1.5 KB'. None -> 'unknown'."""
    if n_bytes is None:
        return "unknown"
    size = float(n_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def configured_files() -> list[str]:
    return settings.data.train_files + settings.data.drift_files


def file_url(name: str) -> str:
    return settings.data.base_url + name


def remote_size(session: requests.Session, url: str) -> int | None:
    """Ask the server for the file size without downloading it (HTTP HEAD)."""
    resp = session.head(url, allow_redirects=True, timeout=TIMEOUT)
    resp.raise_for_status()
    length = resp.headers.get("Content-Length")
    return int(length) if length else None


def verify_zip(path: Path) -> bool:
    """A zip's index sits at the END of the file, so a truncated download fails here."""
    try:
        with zipfile.ZipFile(path) as zf:
            return len(zf.namelist()) > 0
    except (zipfile.BadZipFile, OSError):
        return False


def download_file(session: requests.Session, name: str, dest_dir: Path) -> Path:
    """Download one file into dest_dir, resuming a previous partial download."""
    url = file_url(name)
    final_path = dest_dir / name
    part_path = dest_dir / (name + ".part")

    if final_path.exists() and verify_zip(final_path):
        logger.info("Already downloaded, skipping: %s", name)
        return final_path

    expected = remote_size(session, url)
    logger.info("Downloading %s (%s)", name, human_size(expected))

    for attempt in range(1, MAX_RETRIES + 1):
        done = part_path.stat().st_size if part_path.exists() else 0
        headers = {"Range": f"bytes={done}-"} if done else {}
        try:
            with session.get(url, headers=headers, stream=True, timeout=TIMEOUT) as resp:
                if resp.status_code == 416:      # nothing left to fetch
                    break
                resp.raise_for_status()
                if done and resp.status_code != 206:
                    logger.warning("Server does not support resume; restarting %s", name)
                    done = 0
                mode = "ab" if done else "wb"
                with open(part_path, mode) as f, tqdm(
                    total=expected, initial=done, unit="B", unit_scale=True,
                    unit_divisor=1024, desc=name[:45],
                ) as bar:
                    for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                        f.write(chunk)
                        bar.update(len(chunk))
            break  # finished without a network error
        except (requests.ConnectionError, requests.Timeout,
                requests.exceptions.ChunkedEncodingError) as exc:
            wait = 2 ** attempt
            logger.warning("Attempt %d/%d failed for %s (%s). Retrying in %ds",
                           attempt, MAX_RETRIES, name, exc.__class__.__name__, wait)
            time.sleep(wait)
    else:
        raise RuntimeError(f"Gave up on {name} after {MAX_RETRIES} attempts. Re-run to resume.")

    got = part_path.stat().st_size
    if expected is not None and got != expected:
        raise RuntimeError(f"Size mismatch for {name}: {got} vs {expected} bytes. Re-run to resume.")
    if not verify_zip(part_path):
        raise RuntimeError(f"{name} is not a valid zip. Delete {part_path.name} and re-run.")

    part_path.replace(final_path)   # rename only once the file is complete and valid
    logger.info("Saved %s (%s)", final_path.name, human_size(got))
    return final_path


def show_sizes(session: requests.Session, files: list[str]) -> None:
    total = 0
    for name in files:
        try:
            size = remote_size(session, file_url(name))
            total += size or 0
            print(f"{human_size(size):>10}  {name}")
        except requests.HTTPError as exc:
            print(f"{'ERROR':>10}  {name}  ({exc.response.status_code})")
    print(f"{human_size(total):>10}  TOTAL for {len(files)} file(s)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download CFPB complaint exports.")
    parser.add_argument("--sizes", action="store_true",
                        help="only print the size of each file, download nothing")
    parser.add_argument("--only", nargs="+", metavar="FILE",
                        help="download only these file names (must be in config.yaml)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings.ensure_dirs()

    files = configured_files()
    if args.only:
        unknown = [f for f in args.only if f not in files]
        if unknown:
            raise SystemExit(f"Not listed in configs/config.yaml: {unknown}")
        files = args.only

    with requests.Session() as session:
        session.headers.update(HEADERS)

        if args.sizes:
            show_sizes(session, files)
            return

        failed = []
        for name in files:
            try:
                download_file(session, name, settings.paths.raw)
            except Exception as exc:  # keep going with the other files
                logger.error("FAILED %s: %s", name, exc)
                failed.append(name)

    if failed:
        raise SystemExit(f"{len(failed)} file(s) failed: {failed}")
    logger.info("All %d file(s) are in %s", len(files), settings.paths.raw)


if __name__ == "__main__":
    main()