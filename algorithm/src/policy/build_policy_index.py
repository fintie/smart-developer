from __future__ import annotations
import argparse
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import requests
import yaml
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter


DEFAULT_SOURCES_PATH = Path("algorithm/configs/policies/policy_sources.yaml")
DEFAULT_INDEX_DIR = Path("algorithm/artifacts/policy_index/chroma")
DEFAULT_RAW_DIR = Path("algorithm/artifacts/policy_index/raw")


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\n+", "\n", text)
    return text.strip()


def fetch_webpage_text(url: str, timeout: int = 60) -> str:
    headers = {
        "User-Agent": "SmartDeveloperPolicyIndexer/0.1"
    }

    response = requests.get(url, headers=headers, timeout=timeout)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
        tag.decompose()

    main = soup.find("main")
    if main is not None:
        text = main.get_text("\n", strip=True)
    else:
        text = soup.get_text("\n", strip=True)

    return clean_text(text)


def build_documents(sources_config: dict[str, Any], raw_dir: Path) -> list[Document]:
    raw_dir.mkdir(parents=True, exist_ok=True)

    docs: list[Document] = []
    retrieved_at = datetime.now(timezone.utc).isoformat()

    for source in sources_config.get("sources", []):
        policy_id = source["policy_id"]
        policy_name = source["policy_name"]
        url = source["url"]

        print(f"Fetching {policy_id}: {url}")

        try:
            text = fetch_webpage_text(url)
        except Exception as exc:
            print(f"  Failed to fetch {url}: {exc}")
            continue

        raw_path = raw_dir / f"{policy_id}.txt"
        raw_path.write_text(text, encoding="utf-8")

        metadata = {
            "policy_id": policy_id,
            "policy_name": policy_name,
            "jurisdiction": source.get("jurisdiction", "NSW"),
            "source_type": source.get("source_type", "webpage"),
            "source_url": url,
            "tags": ",".join(source.get("tags", [])),
            "retrieved_at": retrieved_at,
        }

        docs.append(Document(page_content=text, metadata=metadata))

        print(f"  chars={len(text):,} saved={raw_path}")

    return docs


def split_documents(docs: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(docs)

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i

    return chunks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", default=str(DEFAULT_SOURCES_PATH))
    parser.add_argument("--index-dir", default=str(DEFAULT_INDEX_DIR))
    parser.add_argument("--raw-dir", default=str(DEFAULT_RAW_DIR))
    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()

    sources_path = Path(args.sources)
    index_dir = Path(args.index_dir)
    raw_dir = Path(args.raw_dir)

    if args.reset and index_dir.exists():
        import shutil
        shutil.rmtree(index_dir)

    config = load_yaml(sources_path)
    docs = build_documents(config, raw_dir=raw_dir)

    if not docs:
        raise RuntimeError("No policy documents were loaded.")

    chunks = split_documents(docs)

    print()
    print(f"Loaded documents: {len(docs):,}")
    print(f"Created chunks: {len(chunks):,}")
    print(f"Building Chroma index at: {index_dir}")

    embeddings = HuggingFaceEmbeddings(model_name=args.embedding_model)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(index_dir),
    )

    vectorstore.persist()

    print("Policy index built successfully.")
    print(f"Index dir: {index_dir}")


if __name__ == "__main__":
    main()
