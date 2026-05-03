from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from algorithm.src.inference.predictor import PredictionRequest, SmartDeveloperPredictor
from algorithm.src.explanation.report import ReportConfig, build_site_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a Smart Developer site recommendation report.")
    parser.add_argument("--strategy", required=True)
    parser.add_argument("--query-text", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--recall-k", type=int, default=200)
    parser.add_argument("--retrieval-experiment", default="two_tower_v1")
    parser.add_argument("--dcn-experiment", default="dcn_reranker_v1")
    parser.add_argument("--no-dcn-reranker", action="store_true")
    parser.add_argument("--no-explanations", action="store_true")
    parser.add_argument("--no-dedupe", action="store_true")
    parser.add_argument("--audience", default="developer")
    parser.add_argument("--title", default="Smart Developer Site Recommendation Report")
    parser.add_argument("--output", help="Optional output markdown path, e.g. reports/apartment_report.md")

    args = parser.parse_args()

    predictor = SmartDeveloperPredictor(
        default_retrieval_model=args.retrieval_experiment,
        default_reranking_model=args.dcn_experiment,
    )

    request = PredictionRequest(
        strategy=args.strategy,
        query_text=args.query_text,
        top_k=args.top_k,
        recall_k=args.recall_k,
        with_explanations=not args.no_explanations,
        retrieval_model=args.retrieval_experiment,
        use_dcn_reranker=not args.no_dcn_reranker,
        reranking_model=args.dcn_experiment,
        dedupe_by_address=not args.no_dedupe,
    )

    response = predictor.predict(request)
    results_df = pd.DataFrame(response["results"])

    report = build_site_report(
        results=results_df,
        strategy=args.strategy,
        query_text=args.query_text,
        config=ReportConfig(
            title=args.title,
            audience=args.audience,
            include_explanations=not args.no_explanations,
            include_risks=True,
            include_table=True,
        ),
    )

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report, encoding="utf-8")
        print(f"Saved report to: {output_path}")
    else:
        print(report)


if __name__ == "__main__":
    main()