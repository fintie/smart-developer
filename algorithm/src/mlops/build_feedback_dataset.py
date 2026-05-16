from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sqlalchemy import text

from algorithm.src.mlops.db import engine


POSITIVE_EVENTS = {
    "click": 0.5,
    "save": 1.0,
    "select": 1.0,
    "manual_positive": 1.0,
}

NEGATIVE_EVENTS = {
    "dismiss": 0.0,
    "manual_negative": 0.0,
}


def load_logged_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    requests_sql = """
    SELECT
        request_id,
        created_at,
        strategy,
        query_text,
        locality,
        address_contains,
        retrieval_model,
        reranking_model,
        use_dcn_reranker,
        with_explanations,
        latency_ms,
        result_count
    FROM retrieval_requests
    """

    results_sql = """
    SELECT
        request_id,
        rid,
        rank_position,
        address,
        base_site_address,
        primary_zoning_code,
        zoning_band,
        lot_size_band,
        constraint_severity_band,
        station_distance_band,
        strategy_score,
        retrieval_similarity,
        fusion_score,
        dcn_prob,
        final_rank_score,
        result_payload
    FROM retrieval_results
    """

    feedback_sql = """
    SELECT
        request_id,
        rid,
        rank_position,
        event_type,
        event_value,
        user_note,
        created_at AS feedback_created_at
    FROM user_feedback
    """

    with engine.connect() as conn:
        requests = pd.read_sql(text(requests_sql), conn)
        results = pd.read_sql(text(results_sql), conn)
        feedback = pd.read_sql(text(feedback_sql), conn)

    return requests, results, feedback


def event_to_label(event_type: str) -> float | None:
    if event_type in POSITIVE_EVENTS:
        return POSITIVE_EVENTS[event_type]
    if event_type in NEGATIVE_EVENTS:
        return NEGATIVE_EVENTS[event_type]
    return None


def aggregate_feedback(feedback: pd.DataFrame) -> pd.DataFrame:
    if feedback.empty:
        return pd.DataFrame(
            columns=[
                "request_id",
                "rid",
                "feedback_label",
                "feedback_event_count",
                "positive_event_count",
                "negative_event_count",
                "feedback_events",
            ]
        )

    fb = feedback.copy()
    fb["rid"] = fb["rid"].astype(str)
    fb["event_label"] = fb["event_type"].apply(event_to_label)

    fb["is_positive_event"] = fb["event_type"].isin(POSITIVE_EVENTS)
    fb["is_negative_event"] = fb["event_type"].isin(NEGATIVE_EVENTS)

    # Multiple feedback events can happen for the same request-result pair.
    # Use max label so save/select dominates click, and positive dominates neutral events.
    grouped = (
        fb.groupby(["request_id", "rid"], as_index=False)
        .agg(
            feedback_label=("event_label", "max"),
            feedback_event_count=("event_type", "count"),
            positive_event_count=("is_positive_event", "sum"),
            negative_event_count=("is_negative_event", "sum"),
            feedback_events=("event_type", lambda x: sorted(set(x))),
        )
    )

    return grouped


def build_feedback_dataset(
    include_unlabelled: bool = True,
    weak_negative_for_unfeedbacked: bool = False,
) -> pd.DataFrame:
    requests, results, feedback = load_logged_data()

    if requests.empty or results.empty:
        return pd.DataFrame()

    results = results.copy()
    results["rid"] = results["rid"].astype(str)

    feedback_agg = aggregate_feedback(feedback)

    df = results.merge(
        requests,
        on="request_id",
        how="left",
        suffixes=("", "_request"),
    )

    df = df.merge(
        feedback_agg,
        on=["request_id", "rid"],
        how="left",
    )

    df["has_feedback"] = df["feedback_event_count"].fillna(0).astype(int) > 0

    if weak_negative_for_unfeedbacked:
        # Use carefully: shown-but-not-clicked is not always a true negative.
        df["label"] = df["feedback_label"].fillna(0.0)
        df["label_source"] = df["feedback_label"].apply(
            lambda x: "explicit_feedback" if pd.notna(x) else "weak_negative_unfeedbacked"
        )
    else:
        df["label"] = df["feedback_label"]
        df["label_source"] = df["feedback_label"].apply(
            lambda x: "explicit_feedback" if pd.notna(x) else "unlabelled"
        )

    if not include_unlabelled:
        df = df[df["label"].notna()].copy()

    # Useful derived features for future reranker training.
    df["shown"] = 1
    df["clicked"] = df["feedback_events"].apply(
        lambda events: int(isinstance(events, list) and "click" in events)
    )
    df["saved"] = df["feedback_events"].apply(
        lambda events: int(isinstance(events, list) and "save" in events)
    )
    df["selected"] = df["feedback_events"].apply(
        lambda events: int(isinstance(events, list) and "select" in events)
    )
    df["dismissed"] = df["feedback_events"].apply(
        lambda events: int(isinstance(events, list) and "dismiss" in events)
    )

    # Keep output column order reasonably readable.
    preferred_cols = [
        "request_id",
        "rid",
        "rank_position",
        "strategy",
        "query_text",
        "locality",
        "address_contains",
        "address",
        "base_site_address",
        "primary_zoning_code",
        "zoning_band",
        "lot_size_band",
        "constraint_severity_band",
        "station_distance_band",
        "strategy_score",
        "retrieval_similarity",
        "fusion_score",
        "dcn_prob",
        "final_rank_score",
        "shown",
        "clicked",
        "saved",
        "selected",
        "dismissed",
        "has_feedback",
        "feedback_event_count",
        "positive_event_count",
        "negative_event_count",
        "feedback_events",
        "label",
        "label_source",
        "retrieval_model",
        "reranking_model",
        "use_dcn_reranker",
    ]

    cols = [c for c in preferred_cols if c in df.columns]
    extra_cols = [c for c in df.columns if c not in cols]

    return df[cols + extra_cols].copy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a feedback-derived reranker dataset from MLOps logs.")
    parser.add_argument("--output", default="algorithm/artifacts/feedback/feedback_reranker_dataset.parquet",
                        help="Output parquet path.")
    parser.add_argument("--labelled-only", action="store_true", help="Only keep rows with explicit feedback labels.")
    parser.add_argument("--weak-negative-unfeedbacked", action="store_true",
                        help="Treat shown-but-unfeedbacked results as weak negatives. Use carefully.")

    args = parser.parse_args()

    df = build_feedback_dataset(
        include_unlabelled=not args.labelled_only,
        weak_negative_for_unfeedbacked=args.weak_negative_unfeedbacked,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_parquet(output_path, index=False)

    print(f"Saved feedback dataset to: {output_path}")
    print(f"Rows: {len(df)}")

    if not df.empty:
        print()
        print("Label source counts:")
        print(df["label_source"].value_counts(dropna=False))

        print()
        print("Label counts:")
        print(df["label"].value_counts(dropna=False))

        print()
        print("Feedback event counts:")
        if "feedback_events" in df.columns:
            print(df["feedback_events"].astype(str).value_counts(dropna=False).head(20))


if __name__ == "__main__":
    main()