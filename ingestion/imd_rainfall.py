from datetime import UTC, datetime
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

DATASET_URL = "https://www.data.gov.in/resource/sub-divisional-monthly-rainfall-1901-2017"


def download_dataset(url: str = DATASET_URL) -> Path:
    """Download the official rainfall dataset resource."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_file = RAW_DATA_DIR / f"rainfall_source_{timestamp}.html"

    response = requests.get(
        url,
        timeout=30,
        headers={"User-Agent": "BHARAT-EARTH/0.1"},
    )
    response.raise_for_status()

    output_file.write_bytes(response.content)

    return output_file


if __name__ == "__main__":
    print("BHARAT-EARTH | Government Rainfall Dataset")
    print(f"Raw data directory: {RAW_DATA_DIR}")

    try:
        output_file = download_dataset()
        print(f"Downloaded: {output_file}")
    except requests.RequestException as error:
        print(f"Dataset download failed: {error}")
