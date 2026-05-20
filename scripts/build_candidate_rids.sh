#!/usr/bin/env bash
set -euo pipefail

CANDIDATES_PATH="${CANDIDATES_PATH:-data/processed/retrieval/candidate_sites.parquet}"
OUTPUT_PATH="${OUTPUT_PATH:-data/processed/retrieval/candidate_rids.txt}"

python - <<PY
import pandas as pd
from pathlib import Path

candidates_path = Path("$CANDIDATES_PATH")
output_path = Path("$OUTPUT_PATH")

df = pd.read_parquet(candidates_path)

output_path.parent.mkdir(parents=True, exist_ok=True)

rids = (
    df["RID"]
    .dropna()
    .astype(str)
    .str.replace(r"\\.0$", "", regex=True)
    .drop_duplicates()
    .sort_values()
)

output_path.write_text("\\n".join(rids) + "\\n")

print("Candidates:", len(df))
print("Unique RIDs:", len(rids))
print("Saved to:", output_path)
PY
