from __future__ import annotations
import json
import time
from pathlib import Path
from typing import Any
import requests


BASE_URL = "https://portal.spatial.nsw.gov.au/server/rest/services/NSW_Geocoded_Addressing_Theme/FeatureServer/1/query"

# repo root
ROOT = Path(__file__).resolve().parents[3]

RAW_DIR = ROOT / "data" / "raw" / "nsw_addressing"
CHUNKS_DIR = RAW_DIR / "chunks"
IDS_PATH = RAW_DIR / "ids.json"

CHUNK_SIZE = 1000
SLEEP_SECONDS = 0.3
TIMEOUT = 60


def ensure_dirs() -> None:
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)


def request_json(params: dict[str, Any]) -> dict[str, Any]:
    response = requests.get(BASE_URL, params=params, timeout=TIMEOUT)
    response.raise_for_status()
    return response.json()


def fetch_all_ids() -> list[int]:
    params = {
        "where": "1=1",
        "returnIdsOnly": "true",
        "f": "json",
    }
    data = request_json(params)

    object_ids = data.get("objectIds", [])
    if not object_ids:
        raise RuntimeError("No objectIds returned. Check the query URL or layer access.")

    with IDS_PATH.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return object_ids


def chunk_list(values: list[int], chunk_size: int) -> list[list[int]]:
    return [values[i : i + chunk_size] for i in range(0, len(values), chunk_size)]


def fetch_chunk(object_ids: list[int], chunk_idx: int, total_chunks: int) -> Path:
    payload = {
        "objectIds": ",".join(str(x) for x in object_ids),
        "outFields": "*",
        "returnGeometry": "true",
        "f": "geojson",
    }

    response = requests.post(BASE_URL, data=payload, timeout=TIMEOUT)

    if not response.ok:
        print(f"Chunk {chunk_idx}/{total_chunks} failed with status {response.status_code}")
        print(response.text[:1000])
        response.raise_for_status()

    out_path = CHUNKS_DIR / f"addresspoint_{chunk_idx:04d}_of_{total_chunks:04d}.geojson"
    out_path.write_text(response.text, encoding="utf-8")
    return out_path


def main() -> None:
    ensure_dirs()

    print("Fetching all object IDs...")
    object_ids = fetch_all_ids()
    print(f"Total object IDs: {len(object_ids)}")

    chunks = chunk_list(object_ids, CHUNK_SIZE)
    total_chunks = len(chunks)
    print(f"Downloading {total_chunks} chunks...")

    for idx, chunk_ids in enumerate(chunks, start=1):
        out_path = CHUNKS_DIR / f"addresspoint_{idx:04d}_of_{total_chunks:04d}.geojson"
        if out_path.exists():
            print(f"[{idx}/{total_chunks}] Skipping existing chunk: {out_path.name}")
            continue

        print(f"[{idx}/{total_chunks}] Downloading {len(chunk_ids)} records...")
        fetch_chunk(chunk_ids, idx, total_chunks)
        time.sleep(SLEEP_SECONDS)

    print("Done. All chunks downloaded.")


if __name__ == "__main__":
    main()