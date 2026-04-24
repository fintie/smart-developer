from __future__ import annotations
import json
import time
from pathlib import Path
from typing import Any
import requests


BASE_URL = "https://portal.spatial.nsw.gov.au/server/rest/services/NSW_Land_Parcel_Property_Theme/FeatureServer/12/query"

ROOT = Path(__file__).resolve().parents[3]

RAW_DIR = ROOT / "data" / "raw" / "nsw_property"
CHUNKS_DIR = RAW_DIR / "chunks"
IDS_PATH = RAW_DIR / "ids.json"

CHUNK_SIZE = 1000
SLEEP_SECONDS = 0.3
TIMEOUT = 60

OUT_FIELDS = ",".join(
    [
        "RID",
        "gurasid",
        "principaladdresssiteoid",
        "addressstringoid",
        "propertytype",
        "valnetpropertystatus",
        "valnetpropertytype",
        "dissolveparcelcount",
        "valnetlotcount",
        "propid",
        "superlot",
        "address",
        "housenumber",
        "urbanity",
        "Shape__Area",
        "Shape__Length",
    ]
)


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
    print(data)

    object_ids = data.get("objectIds", [])
    if not object_ids:
        raise RuntimeError("No objectIds returned. Check the query URL or layer access.")

    with IDS_PATH.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return object_ids


def chunk_list(values: list[int], chunk_size: int) -> list[list[int]]:
    return [values[i : i + chunk_size] for i in range(0, len(values), chunk_size)]


def fetch_chunk_recursive(object_ids: list[int], chunk_name: str) -> None:
    payload = {
        "objectIds": ",".join(str(x) for x in object_ids),
        "outFields": OUT_FIELDS,
        "returnGeometry": "true",
        "f": "geojson",
    }

    out_path = CHUNKS_DIR / f"{chunk_name}.geojson"
    if out_path.exists():
        print(f"Skipping existing chunk: {out_path.name}")
        return

    response = requests.post(BASE_URL, data=payload, timeout=TIMEOUT)

    if response.ok:
        out_path.write_text(response.text, encoding="utf-8")
        print(f"Saved {out_path.name} ({len(object_ids)} ids)")
        return

    print(f"Failed chunk {chunk_name} with {len(object_ids)} ids, status={response.status_code}")

    if len(object_ids) == 1:
        print(f"Single OBJECTID failed permanently: {object_ids[0]}")
        bad_path = CHUNKS_DIR / "failed_ids.txt"
        with bad_path.open("a", encoding="utf-8") as f:
            f.write(str(object_ids[0]) + "\n")
        return

    mid = len(object_ids) // 2
    left = object_ids[:mid]
    right = object_ids[mid:]

    fetch_chunk_recursive(left, f"{chunk_name}_a")
    fetch_chunk_recursive(right, f"{chunk_name}_b")


def main() -> None:
    ensure_dirs()

    print("Fetching all object IDs...")
    object_ids = fetch_all_ids()
    print(f"Total object IDs: {len(object_ids)}")

    chunks = chunk_list(object_ids, CHUNK_SIZE)
    total_chunks = len(chunks)
    print(f"Downloading {total_chunks} chunks...")

    for idx, chunk_ids in enumerate(chunks, start=1):
        chunk_name = f"property_{idx:04d}_of_{total_chunks:04d}"
        print(f"[{idx}/{total_chunks}] Processing {len(chunk_ids)} ids...")
        fetch_chunk_recursive(chunk_ids, chunk_name)
        time.sleep(SLEEP_SECONDS)

    print("Done. All property chunks downloaded.")


if __name__ == "__main__":
    main()