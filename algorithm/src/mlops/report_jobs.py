from __future__ import annotations
import time
from pathlib import Path
from typing import Any
import pandas as pd
from sqlalchemy import text

from algorithm.src.explanation.report import ReportConfig, build_site_report
from algorithm.src.explanation.report_export import export_markdown_report_to_pdf
from algorithm.src.explanation.template_generator import add_template_explanations
from algorithm.src.mlops.db import engine
from algorithm.src.mlops.logger import (
    complete_report_job,
    create_report_job,
    fail_report_job,
    mark_report_job_running,
)


def _load_request_and_results(request_id: str) -> tuple[dict[str, Any], pd.DataFrame]:
    request_sql = """
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
        result_count,
        request_payload,
        metadata_payload
    FROM retrieval_requests
    WHERE request_id = :request_id
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
    WHERE request_id = :request_id
    ORDER BY rank_position
    """

    with engine.connect() as conn:
        request_df = pd.read_sql(text(request_sql), conn, params={"request_id": request_id})
        results_df = pd.read_sql(text(results_sql), conn, params={"request_id": request_id})

    if request_df.empty:
        raise ValueError(f"No retrieval request found for request_id={request_id}")

    if results_df.empty:
        raise ValueError(f"No retrieval results found for request_id={request_id}")

    request = request_df.iloc[0].to_dict()

    # Expand result_payload back into columns where available.
    rows: list[dict[str, Any]] = []
    for _, row in results_df.iterrows():
        base = row.to_dict()
        payload = base.get("result_payload")

        if isinstance(payload, dict):
            merged = dict(payload)
            # DB columns should win for stable logged values.
            for key, value in base.items():
                if key != "result_payload":
                    merged[key] = value
            rows.append(merged)
        else:
            rows.append(base)

    expanded_results = pd.DataFrame(rows)

    return request, expanded_results


def _attach_template_explanations_if_needed(
    results_df: pd.DataFrame,
    strategy: str,
    explanation_mode: str,
) -> pd.DataFrame:
    if explanation_mode != "template":
        return results_df

    records = results_df.to_dict(orient="records")
    enriched = add_template_explanations(
        records,
        strategy=strategy,
        output_field="fast_explanation",
    )

    out = pd.DataFrame(enriched)

    # build_site_report currently expects "explanation" for the rationale section.
    # For template mode, use fast_explanation as explanation.
    out["explanation"] = out["fast_explanation"]

    return out


def generate_report_from_request_id(
    *,
    request_id: str,
    explanation_mode: str = "template",
    output_markdown: bool = True,
    output_pdf: bool = True,
    output_dir: str | Path = "algorithm/artifacts/reports",
    audience: str = "developer",
    title: str = "Smart Developer Site Recommendation Report",
) -> dict[str, Any]:
    """
    Generate a markdown/PDF report from already logged retrieval results.

    This first MVP runs synchronously. Later it can be moved behind a queue worker.
    """
    t0 = time.perf_counter()

    report_id = create_report_job(
        request_id=request_id,
        explanation_mode=explanation_mode,
    )

    markdown_path: Path | None = None
    pdf_path: Path | None = None

    try:
        mark_report_job_running(report_id)

        request, results_df = _load_request_and_results(request_id)

        strategy = str(request["strategy"])
        query_text = str(request["query_text"])

        results_df = _attach_template_explanations_if_needed(
            results_df=results_df,
            strategy=strategy,
            explanation_mode=explanation_mode,
        )

        report = build_site_report(
            results=results_df,
            strategy=strategy,
            query_text=query_text,
            config=ReportConfig(
                title=title,
                audience=audience,
            ),
        )

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if output_markdown:
            markdown_path = output_dir / f"{report_id}.md"
            markdown_path.write_text(report, encoding="utf-8")

        if output_pdf:
            pdf_path = output_dir / f"{report_id}.pdf"
            export_markdown_report_to_pdf(
                markdown_text=report,
                output_pdf_path=pdf_path,
            )

        latency_ms = round((time.perf_counter() - t0) * 1000, 2)

        complete_report_job(
            report_id=report_id,
            output_markdown_path=str(markdown_path) if markdown_path else None,
            output_pdf_path=str(pdf_path) if pdf_path else None,
            latency_ms=latency_ms,
        )

        return {
            "status": "ready",
            "report_id": report_id,
            "request_id": request_id,
            "explanation_mode": explanation_mode,
            "markdown_path": str(markdown_path) if markdown_path else None,
            "pdf_path": str(pdf_path) if pdf_path else None,
            "latency_ms": latency_ms,
        }

    except Exception as exc:
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        fail_report_job(
            report_id=report_id,
            error_message=str(exc),
            latency_ms=latency_ms,
        )
        raise


def get_report_job(report_id: str) -> dict[str, Any]:
    sql = """
    SELECT
        report_id,
        request_id,
        status,
        explanation_mode,
        output_markdown_path,
        output_pdf_path,
        latency_ms,
        error_message,
        created_at,
        completed_at
    FROM report_jobs
    WHERE report_id = :report_id
    """

    with engine.connect() as conn:
        df = pd.read_sql(text(sql), conn, params={"report_id": report_id})

    if df.empty:
        raise ValueError(f"Report job not found: {report_id}")

    row = df.iloc[0].to_dict()

    # Make timestamps JSON-friendly.
    for key in ["created_at", "completed_at"]:
        if key in row and row[key] is not None:
            row[key] = str(row[key])

    return row