"""
BHARAT-EARTH
Government Rainfall Dataset Ingestion
"""

from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)


def download_file(url: str, filename: str) -> Path:
    """Download a public dataset into data/raw."""

    output_file = RAW_DATA_DIR / filename

    response = requests.get(url, timeout=60)
    response.raise_for_status()

    output_file.write_bytes(response.content)

    return output_file


if __name__ == "__main__":
    print("BHARAT-EARTH | Government Rainfall Dataset")
    print(f"Raw data directory: {RAW_DATA_DIR}")
