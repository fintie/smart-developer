from __future__ import annotations

import argparse
import re
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup


DEFAULT_PAGE_URL = "https://valuation.property.nsw.gov.au/embed/propertySalesInformation"


def safe_filename_from_url(url: str) -> str:
    parsed = urlparse(url)
    name = Path(parsed.path).name
    if name:
        return name

    clean = re.sub(r"[^A-Za-z0-9_.-]+", "_", url)
    return clean[:180] + ".zip"


def find_download_links(page_url: str) -> list[str]:
    response = requests.get(page_url, timeout=60)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    links: list[str] = []

    for a in soup.find_all("a", href=True):
        href = a["href"]
        candidate = urljoin(page_url, href)

        if re.search(r"/__psi/(weekly|yearly)/\d{4,8}\.zip$", candidate, flags=re.I):
            links.append(candidate)

    seen = set()
    deduped = []
    for link in links:
        if link not in seen:
            seen.add(link)
            deduped.append(link)

    return deduped


def classify_link(url: str) -> tuple[str, str]:
    """
    Returns (kind, key)
    kind: weekly or yearly
    key: YYYYMMDD for weekly, YYYY for yearly
    """
    match = re.search(r"/__psi/(weekly|yearly)/(\d{4,8})\.zip$", url)
    if not match:
        return "unknown", ""
    return match.group(1), match.group(2)


def filter_links(
    links: list[str],
    kind: str | None,
    years: set[str] | None,
    dates: set[str] | None,
) -> list[str]:
    out = []

    for link in links:
        link_kind, key = classify_link(link)

        if kind and link_kind != kind:
            continue

        if years:
            if link_kind == "yearly":
                if key not in years:
                    continue
            elif link_kind == "weekly":
                if key[:4] not in years:
                    continue

        if dates:
            if key not in dates:
                continue

        out.append(link)

    return out


def download_file(url: str, output_dir: Path) -> Path:
    kind, _ = classify_link(url)
    target_dir = output_dir / kind
    target_dir.mkdir(parents=True, exist_ok=True)

    filename = safe_filename_from_url(url)
    output_path = target_dir / filename

    if output_path.exists() and output_path.stat().st_size > 0:
        print(f"Already exists, skipping: {output_path}")
        return output_path

    with requests.get(url, stream=True, timeout=180) as r:
        r.raise_for_status()

        with output_path.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--page-url", default=DEFAULT_PAGE_URL)
    parser.add_argument("--output-dir", default="data/raw/nsw_psi")
    parser.add_argument("--list-only", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--kind", choices=["weekly", "yearly"], default=None,
                        help="Download only weekly or yearly PSI zip files.")
    parser.add_argument("--years", nargs="*", default=None,
                        help="Years to download/filter, e.g. --years 2024 2025 2026")
    parser.add_argument("--dates", nargs="*", default=None,
                        help="Weekly date keys to download/filter, e.g. --dates 20260511 20260518")

    args = parser.parse_args()

    years = set(args.years) if args.years else None
    dates = set(args.dates) if args.dates else None

    links = find_download_links(args.page_url)
    links = filter_links(links, kind=args.kind, years=years, dates=dates)

    print(f"Found {len(links)} matching PSI zip links")
    for i, link in enumerate(links, 1):
        kind, key = classify_link(link)
        print(f"{i}. [{kind}] {key} -> {link}")

    if args.list_only:
        return

    selected = links[: args.limit] if args.limit else links
    output_dir = Path(args.output_dir)

    for i, link in enumerate(selected, 1):
        print(f"\n[{i}/{len(selected)}] Downloading {link}")
        path = download_file(link, output_dir)
        print(f"Saved to {path}")


if __name__ == "__main__":
    main()