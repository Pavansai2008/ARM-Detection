import os
from pathlib import Path
from typing import Iterable, Optional

import requests


class ModelDownloadError(RuntimeError):
    """Raised when a model file could not be downloaded."""


def _first_existing_path(filename: str, directories: Iterable[Path]) -> Optional[Path]:
    for directory in directories:
        candidate = directory / filename
        if candidate.exists():
            return candidate
    return None


def _get_download_url(env_keys: Iterable[str]) -> Optional[str]:
    for key in env_keys:
        url = os.environ.get(key)
        if url:
            return url
    return None


def _download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        with destination.open("wb") as file_handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file_handle.write(chunk)


def ensure_model_file(
    filename: str,
    directories: Iterable[Path],
    env_keys: Iterable[str],
) -> Path:
    """
    Return a path to the requested model file, downloading it if necessary.

    Args:
        filename: The model filename (e.g. ``best_arm_classifier.pth``).
        directories: A list of directories to search for the file or to use as the
            download destination.
        env_keys: Environment variable names that may contain a download URL.

    Returns:
        The resolved path to the model file.

    Raises:
        FileNotFoundError: If the file does not exist and no download URL is provided.
        ModelDownloadError: If downloading the file fails.
    """

    directories = list(directories)
    existing = _first_existing_path(filename, directories)
    if existing:
        return existing

    url = _get_download_url(env_keys)
    if not url:
        raise FileNotFoundError(
            f"{filename} not found in any of: "
            + ", ".join(str((directory / filename).resolve()) for directory in directories)
            + ". Provide a download URL via environment variable: "
            + ", ".join(env_keys)
        )

    destination = directories[0] / filename
    try:
        _download_file(url, destination)
    except Exception as exc:  # noqa: BLE001
        raise ModelDownloadError(f"Failed to download {filename} from {url}") from exc

    return destination

