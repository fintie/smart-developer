from __future__ import annotations
import argparse
from textwrap import shorten
import pandas as pd
from algorithm.src.retrieval.hybrid_retrieve import HybridRetriever, RetrievalRequest


def compact_columns(strategy: str) -> list[str]:
    score_col = f"{strategy}_score"
    return [
        "RID",
        "address",
        "primary_zoning_code",
        "lot_size_band",
        "constraint_severity_band",
        "station_distance_band",
        "top_strategy",
        score_col,
        "retrieval_similarity",
        "fusion_score",
        "explanation",
    ]


def print_results(
    df: pd.DataFrame,
    strategy: str,
    top_k: int,
    with_explanations: bool
) -> None:
    score_col = f"{strategy}_score"
    cols = [c for c in compact_columns(strategy) if c in df.columns]
    view = df[cols].head(top_k).copy()

    if with_explanations and "explanation" in view.columns:
        view["explanation"] = view["explanation"].fillna("").apply(lambda x: shorten(str(x), width=260, placeholder="..."))
    elif "explanation" in view.columns:
        view = view.drop(columns=["explanation"])

    if score_col in view.columns:
        view = view.rename(columns={score_col: "strategy_score"})

    print("\n=== Retrieval Results ===")
    print(view.to_string(index=False))

    if with_explanations and "explanation" in df.columns:
        print("\n=== Full Explanations ===")
        for i, (_, row) in enumerate(df.head(top_k).iterrows(), start=1):
            print(f"\n[{i}] {row.get('address', 'Unknown address')}")
            print(row.get("explanation", ""))


def main() -> None:
    parser = argparse.ArgumentParser(description="Terminal demo for Smart Developer retrieval.")
    parser.add_argument("--experiment", default="two_tower_v1",
                        help="Experiment name from algorithm/configs/model.yaml")
    parser.add_argument("--strategy", help="Strategy name, e.g. low_rise_apartment")
    parser.add_argument("--query-text", help="Free-text intent query")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--recall-k", type=int, default=200)
    parser.add_argument("--alpha", type=float, default=0.5, help="Retrieval similarity weight")
    parser.add_argument("--beta", type=float, default=0.5, help="Heuristic score weight")
    parser.add_argument("--with-explanations", action="store_true")
    parser.add_argument("--no-dcn-reranker", action="store_true")
    parser.add_argument("--dcn-experiment", default="dcn_reranker_v1")
    parser.add_argument("--dcn-experiment", default="dcn_reranker_v1")
    parser.add_argument("--no-dedupe", action="store_true")
    args = parser.parse_args()

    strategy = args.strategy or input(
        "Enter strategy "
        "(single_dwelling_rebuild / assembly_opportunity / granny_flat / "
        "land_bank_hold / townhouse_multi_dwelling / low_rise_apartment / dual_occupancy):\n> "
    ).strip()

    query_text = args.query_text or input("Enter a development intent query:\n> ").strip()

    print(f"\nLoading retriever: {args.experiment}")
    retriever = HybridRetriever(experiment=args.experiment)

    request = RetrievalRequest(
        strategy=strategy,
        query_text=query_text,
        top_k=args.top_k,
        recall_k=args.recall_k,
        alpha=args.alpha,
        beta=args.beta,
        dedupe_by_address=not args.no_dedupe,
        attach_explanations=args.with_explanations,
        use_dcn_reranker=args.use_dcn_reranker,
        dcn_experiment=args.dcn_experiment,
    )

    results = retriever.retrieve(request)
    print_results(results, strategy=strategy, top_k=args.top_k, with_explanations=args.with_explanations)


if __name__ == "__main__":
    main()