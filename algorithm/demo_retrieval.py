from __future__ import annotations
import argparse
from textwrap import shorten
import pandas as pd
from algorithm.src.retrieval.hybrid_retrieve import HybridRetriever, RetrievalRequest
from algorithm.src.agent.query_planner import plan_query


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
        "serving_boost",
        "dcn_prob",
        "dcn_rank_score",
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
        view["explanation"] = view["explanation"].fillna("").apply(
            lambda x: shorten(str(x), width=260, placeholder="...")
        )
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
                        help="Retrieval experiment name from algorithm/configs/model.yaml")
    parser.add_argument("--strategy", help="Strategy name, e.g. low_rise_apartment")
    parser.add_argument("--query-text", help="Free-text intent query")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--recall-k", type=int, default=200)
    parser.add_argument("--alpha", type=float, default=0.5, help="Retrieval similarity weight for fusion mode")
    parser.add_argument("--beta", type=float, default=0.5, help="Heuristic score weight for fusion mode")
    parser.add_argument("--with-explanations", action="store_true")
    parser.add_argument("--no-dcn-reranker", action="store_true", help="Disable DCN reranker and use fusion-only reranking")
    parser.add_argument("--dcn-experiment", default="dcn_reranker_v1",
                        help="DCN reranker experiment name from algorithm/configs/model.yaml")
    parser.add_argument("--locality", help="Optional locality/suburb text filter against address")
    parser.add_argument("--address-contains", help="Optional address text filter")
    parser.add_argument("--no-dedupe", action="store_true")
    parser.add_argument("--use-query-planner", action="store_true",
                        help="Use lightweight query planner to rewrite query and detect strategy conflicts")
    args = parser.parse_args()

    strategy = args.strategy or input(
        "Enter strategy "
        "(single_dwelling_rebuild / assembly_opportunity / granny_flat / "
        "land_bank_hold / townhouse_multi_dwelling / low_rise_apartment / dual_occupancy):\n> "
    ).strip()

    query_text = args.query_text or input(
        "Enter a development intent query:\n> "
    ).strip()

    use_dcn_reranker = not args.no_dcn_reranker

    print(f"\nLoading retriever: {args.experiment}")
    print(f"DCN reranker: {'enabled' if use_dcn_reranker else 'disabled'}")
    if use_dcn_reranker:
        print(f"DCN experiment: {args.dcn_experiment}")

    retriever = HybridRetriever(experiment=args.experiment)

    if args.use_query_planner:
        plan = plan_query(
            query_text=query_text,
            strategy=strategy,
        )

        print("\n=== Query Planner ===")
        print(f"Selected strategy: {plan.selected_strategy}")
        print(f"Sanitised query: {plan.sanitised_query}")

        if plan.warnings:
            print("\nWarnings:")
            for warning in plan.warnings:
                print(f"- {warning}")

        if plan.suggested_alternatives:
            print("\nSuggested alternatives:")
            for alt in plan.suggested_alternatives:
                print(f"- {alt}")

        strategy = plan.selected_strategy
        query_text = plan.rewritten_query

    request = RetrievalRequest(
        strategy=strategy,
        query_text=query_text,
        top_k=args.top_k,
        recall_k=args.recall_k,
        alpha=args.alpha,
        beta=args.beta,
        dedupe_by_address=not args.no_dedupe,
        attach_explanations=args.with_explanations,
        use_dcn_reranker=use_dcn_reranker,
        dcn_experiment=args.dcn_experiment,
        locality=args.locality,
        address_contains=args.address_contains,
    )

    results = retriever.retrieve(request)
    print_results(
        results,
        strategy=strategy,
        top_k=args.top_k,
        with_explanations=args.with_explanations,
    )


if __name__ == "__main__":
    main()