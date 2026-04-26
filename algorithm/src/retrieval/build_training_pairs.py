from __future__ import annotations
import json
from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
QUERY_PATH = ROOT / "data" / "processed" / "retrieval" / "query_intents.jsonl"
CANDIDATE_PATH = ROOT / "data" / "processed" / "retrieval" / "candidate_sites.parquet"
OUTPUT_DIR = ROOT / "data" / "processed" / "retrieval"
OUTPUT_PATH = OUTPUT_DIR / "training_pairs.parquet"

CANDIDATE_TEXT_COL = "candidate_text_clean"

POS_THRESHOLD = 70.0
NEG_THRESHOLD = 20.0

POSITIVES_PER_QUERY = 200
RANDOM_NEGATIVES_PER_QUERY = 200
HARD_NEGATIVES_PER_QUERY = 200

RANDOM_SEED = 42


def load_queries(path: Path) -> pd.DataFrame:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def score_col_for_strategy(strategy: str) -> str:
    return f"{strategy}_score"


def build_pairs_for_query(
    query_row: pd.Series,
    candidate_df: pd.DataFrame,
) -> pd.DataFrame:
    strategy = str(query_row["strategy"])
    query_id = str(query_row["query_id"])
    query_text = str(query_row["text"])
    score_col = score_col_for_strategy(strategy)

    if score_col not in candidate_df.columns:
        raise KeyError(f"Missing score column: {score_col}")

    df = candidate_df.copy()
    if CANDIDATE_TEXT_COL not in df.columns:
        raise KeyError(f"Missing candidate text column: {CANDIDATE_TEXT_COL}")
    df["candidate_score"] = df[score_col]

    positives = (
        df[df["candidate_score"] >= POS_THRESHOLD]
        .sort_values("candidate_score", ascending=False)
        .head(POSITIVES_PER_QUERY)
        .copy()
    )
    positives["label"] = 1
    positives["pair_type"] = "positive"

    random_negatives_pool = df[df["candidate_score"] <= NEG_THRESHOLD].copy()
    random_negatives = random_negatives_pool.sample(
        n=min(RANDOM_NEGATIVES_PER_QUERY, len(random_negatives_pool)),
        random_state=RANDOM_SEED,
    ).copy()
    random_negatives["label"] = 0
    random_negatives["pair_type"] = "random_negative"

    hard_negative_pool = df[
        (df["candidate_score"] > NEG_THRESHOLD)
        & (df["candidate_score"] < POS_THRESHOLD)
    ].copy()

    if "top_strategy" in hard_negative_pool.columns:
        hard_negative_pool = hard_negative_pool[
            hard_negative_pool["top_strategy"].notna()
            & (hard_negative_pool["top_strategy"] != strategy)
        ].copy()

    hard_negatives = (
        hard_negative_pool
        .sort_values("candidate_score", ascending=False)
        .head(HARD_NEGATIVES_PER_QUERY)
        .copy()
    )
    hard_negatives["label"] = 0
    hard_negatives["pair_type"] = "hard_negative"

    pairs = pd.concat(
        [positives, random_negatives, hard_negatives],
        ignore_index=True,
    )

    keep_cols = [
        "RID",
        CANDIDATE_TEXT_COL,
        "candidate_score",
        "label",
        "pair_type",
    ]

    optional_cols = [
        "address",
        "top_strategy",
        "top_strategy_score",
        "primary_zoning_code",
        "lot_size_band",
        "constraint_severity_band",
        "station_distance_band",
    ]
    for c in optional_cols:
        if c in pairs.columns:
            keep_cols.append(c)

    pairs = pairs[keep_cols].copy()
    pairs = pairs.rename(columns={"RID": "candidate_rid"})

    pairs["query_id"] = query_id
    pairs["strategy"] = strategy
    pairs["query_text"] = query_text

    ordered_cols = [
        "query_id",
        "strategy",
        "query_text",
        "candidate_rid",
        CANDIDATE_TEXT_COL,
        "candidate_score",
        "label",
        "pair_type",
    ] + [c for c in pairs.columns if c not in {
        "query_id",
        "strategy",
        "query_text",
        "candidate_rid",
        CANDIDATE_TEXT_COL,
        "candidate_score",
        "label",
        "pair_type",
    }]

    return pairs[ordered_cols]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Reading queries: {QUERY_PATH}")
    queries = load_queries(QUERY_PATH)
    print(f"Queries: {len(queries)}")

    print(f"Reading candidates: {CANDIDATE_PATH}")
    candidates = pd.read_parquet(CANDIDATE_PATH)
    print(f"Candidates: {len(candidates)}")

    all_pairs = []
    for _, query_row in queries.iterrows():
        print(f"Building pairs for {query_row['query_id']} / {query_row['strategy']}")
        pairs = build_pairs_for_query(query_row, candidates)
        print(
            "  pairs:",
            len(pairs),
            "| positives:",
            int((pairs["pair_type"] == "positive").sum()),
            "| random negatives:",
            int((pairs["pair_type"] == "random_negative").sum()),
            "| hard negatives:",
            int((pairs["pair_type"] == "hard_negative").sum()),
        )
        all_pairs.append(pairs)

    training_pairs = pd.concat(all_pairs, ignore_index=True)

    print("Total pairs:", len(training_pairs))
    print(training_pairs["label"].value_counts(dropna=False))
    print(training_pairs["pair_type"].value_counts(dropna=False))

    print(f"Writing: {OUTPUT_PATH}")
    training_pairs.to_parquet(OUTPUT_PATH, index=False)

    print("Verifying output...")
    check = pd.read_parquet(OUTPUT_PATH)
    print("Verified rows:", len(check))
    print(check.head())


if __name__ == "__main__":
    main()