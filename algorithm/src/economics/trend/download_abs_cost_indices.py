from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import requests
import yaml


DEFAULT_CONFIG_PATH = Path("algorithm/configs/economics/trend/abs_cost_indices.yaml")


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def download_url(url: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    headers = {
        "Accept": "text/csv,application/vnd.sdmx.data+csv,*/*",
        "User-Agent": "smart-developer-research-prototype/0.1",
    }

    print(f"Downloading: {url}")
    response = requests.get(url, headers=headers, timeout=120)
    response.raise_for_status()

    content_type = response.headers.get("content-type", "")
    print(f"Status: {response.status_code}")
    print(f"Content-Type: {content_type}")
    print(f"Bytes: {len(response.content):,}")

    output_path.write_bytes(response.content)
    print(f"Saved: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--source", choices=["all", "wpi", "ppi"], default="all")
    args = parser.parse_args()

    config = load_yaml(Path(args.config))
    sources = config["sources"]

    selected_sources = (
        list(sources.keys())
        if args.source == "all"
        else [args.source]
    )

    for source_name in selected_sources:
        source = sources[source_name]
        download_url(
            url=source["url"],
            output_path=Path(source["raw_output_csv"]),
        )


if __name__ == "__main__":
    main()