from __future__ import annotations

import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from algorithm.src.explanation.template_generator import add_template_explanations
from algorithm.src.inference.predictor import (
    DEFAULT_RERANKING_MODEL,
    DEFAULT_RETRIEVAL_MODEL,
    PredictionRequest,
    SmartDeveloperPredictor,
)
from algorithm.src.mlops.logger import log_retrieval_response, log_user_feedback
from algorithm.src.mlops.report_jobs import generate_report_from_request_id, get_report_job


ROOT_DIR = Path(__file__).resolve().parents[1]
FRONTEND_DIR = ROOT_DIR / "frontend"
REPORTS_DIR = ROOT_DIR / "algorithm" / "artifacts" / "reports"


class RetrieveSitesPayload(BaseModel):
    strategy: str
    query_text: str

    top_k: int = Field(default=5, ge=1, le=50)
    recall_k: int = Field(default=1000, ge=10, le=10000)

    # Slow local LLM explanation. Usually false for product search.
    with_explanations: bool = False

    # Fast deterministic explanation for site cards.
    use_template_explanations: bool = True

    retrieval_model: str = DEFAULT_RETRIEVAL_MODEL
    use_dcn_reranker: bool = True
    reranking_model: str = DEFAULT_RERANKING_MODEL

    alpha: float = 0.5
    beta: float = 0.5
    dedupe_by_address: bool = True

    locality: str | None = None
    address_contains: str | None = None

    # If true:
    # - exact location matches are preferred
    # - if insufficient/no exact matches, broader results are returned with warning metadata
    location_fallback: bool = True

    user_id: str | None = None
    session_id: str | None = None

    log_request: bool = True
    debug: bool = False


class FeedbackPayload(BaseModel):
    request_id: str
    event_type: str

    rid: str | int | None = None
    rank_position: int | None = None
    event_value: dict[str, Any] | None = None
    user_note: str | None = None

    user_id: str | None = None
    session_id: str | None = None


class ReportJobPayload(BaseModel):
    request_id: str
    explanation_mode: str = "template"

    output_markdown: bool = True
    output_pdf: bool = True

    audience: str = "developer"
    title: str = "Smart Developer Site Recommendation Report"


class ServiceState:
    predictor: SmartDeveloperPredictor | None = None
    startup_latency_ms: float | None = None
    model_load_and_warmup_latency_ms: float | None = None
    is_ready: bool = False


state = ServiceState()


PRODUCT_RESULT_FIELDS = [
    "RID",
    "address",
    "base_site_address",
    "primary_zoning_code",
    "primary_zoning_class",
    "zoning_band",
    "lot_size_band",
    "lot_size_proxy_sqm",
    "constraint_severity_band",
    "station_distance_band",
    "distance_to_station_m",
    "within_800m_catchment",
    "heritage_flag",
    "flood_flag",
    "bushfire_flag",
    "top_strategy",
    "top_strategy_score",
    "strategy_score",
    "retrieval_similarity",
    "fusion_score",
    "dcn_prob",
    "dcn_rank_score",
    "fast_explanation",
    "explanation",
]


def _filter_product_response(response: dict[str, Any]) -> dict[str, Any]:
    filtered = dict(response)
    filtered["results"] = [
        {k: item.get(k) for k in PRODUCT_RESULT_FIELDS if k in item}
        for item in response.get("results", [])
    ]
    return filtered


def _warmup_predictor(predictor: SmartDeveloperPredictor) -> float:
    warmup_request = PredictionRequest(
        strategy="low_rise_apartment",
        query_text=(
            "I want a site for low-rise apartment redevelopment near a train station, "
            "with high development zoning, a large site, and limited planning constraints."
        ),
        top_k=1,
        recall_k=1000,
        with_explanations=False,
        retrieval_model=DEFAULT_RETRIEVAL_MODEL,
        use_dcn_reranker=True,
        reranking_model=DEFAULT_RERANKING_MODEL,
        locality="WAITARA",
        location_fallback=True,
    )

    t0 = time.perf_counter()
    predictor.predict(warmup_request)
    return round((time.perf_counter() - t0) * 1000, 2)


@asynccontextmanager
async def lifespan(app: FastAPI):
    startup_t0 = time.perf_counter()

    predictor = SmartDeveloperPredictor()
    state.predictor = predictor

    state.model_load_and_warmup_latency_ms = _warmup_predictor(predictor)
    state.startup_latency_ms = round((time.perf_counter() - startup_t0) * 1000, 2)
    state.is_ready = True

    yield

    state.is_ready = False
    state.predictor = None


app = FastAPI(
    title="Smart Developer Demo Backend",
    version="0.1.0",
    lifespan=lifespan,
)


# Static frontend assets from existing build.
if (FRONTEND_DIR / "assets").exists():
    app.mount(
        "/assets",
        StaticFiles(directory=str(FRONTEND_DIR / "assets")),
        name="assets",
    )


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "status": "ready" if state.is_ready else "starting",
        "service_startup_latency_ms": state.startup_latency_ms,
        "model_load_and_warmup_latency_ms": state.model_load_and_warmup_latency_ms,
        "warm_request_expected": state.is_ready,
        "production_models": {
            "retrieval_model": DEFAULT_RETRIEVAL_MODEL,
            "reranking_model": DEFAULT_RERANKING_MODEL,
        },
    }


@app.post("/api/retrieve-sites")
def retrieve_sites(payload: RetrieveSitesPayload) -> dict[str, Any]:
    if state.predictor is None or not state.is_ready:
        return {
            "status": "error",
            "message": "Predictor is not ready.",
        }

    request = PredictionRequest(
        strategy=payload.strategy,
        query_text=payload.query_text,
        top_k=payload.top_k,
        recall_k=payload.recall_k,
        with_explanations=payload.with_explanations,
        retrieval_model=payload.retrieval_model,
        use_dcn_reranker=payload.use_dcn_reranker,
        reranking_model=payload.reranking_model,
        alpha=payload.alpha,
        beta=payload.beta,
        dedupe_by_address=payload.dedupe_by_address,
        locality=payload.locality,
        address_contains=payload.address_contains,
        location_fallback=payload.location_fallback,
    )

    response = state.predictor.predict(request)

    # Attach fast deterministic explanations after retrieval/reranking.
    # This avoids local Ollama latency in the normal search path.
    if payload.use_template_explanations:
        response["results"] = add_template_explanations(
            response.get("results", []),
            strategy=payload.strategy,
            output_field="fast_explanation",
        )
        response["metadata"]["use_template_explanations"] = True
    else:
        response["metadata"]["use_template_explanations"] = False

    response["service"] = {
        "mode": "warm_predictor_singleton",
        "service_startup_latency_ms": state.startup_latency_ms,
        "model_load_and_warmup_latency_ms": state.model_load_and_warmup_latency_ms,
    }

    # Log full response before product-field filtering.
    if payload.log_request:
        try:
            log_retrieval_response(
                response,
                user_id=payload.user_id,
                session_id=payload.session_id,
            )
            response["logging"] = {
                "enabled": True,
                "status": "logged",
            }
        except Exception as exc:
            response["logging"] = {
                "enabled": True,
                "status": "failed",
                "error": str(exc),
            }

    if not payload.debug:
        response = _filter_product_response(response)

    return response


@app.post("/api/feedback")
def feedback(payload: FeedbackPayload) -> dict[str, Any]:
    try:
        feedback_id = log_user_feedback(
            request_id=payload.request_id,
            rid=payload.rid,
            rank_position=payload.rank_position,
            event_type=payload.event_type,
            event_value={
                **(payload.event_value or {}),
                "user_id": payload.user_id,
                "session_id": payload.session_id,
            },
            user_note=payload.user_note,
        )

        return {
            "status": "logged",
            "feedback_id": feedback_id,
            "request_id": payload.request_id,
        }

    except Exception as exc:
        return {
            "status": "failed",
            "error": str(exc),
            "request_id": payload.request_id,
        }


@app.post("/api/report-jobs")
def create_report_job_endpoint(payload: ReportJobPayload) -> dict[str, Any]:
    try:
        result = generate_report_from_request_id(
            request_id=payload.request_id,
            explanation_mode=payload.explanation_mode,
            output_markdown=payload.output_markdown,
            output_pdf=payload.output_pdf,
            audience=payload.audience,
            title=payload.title,
        )
        return result

    except Exception as exc:
        return {
            "status": "failed",
            "request_id": payload.request_id,
            "error": str(exc),
        }


@app.get("/api/report-jobs/{report_id}")
def get_report_job_endpoint(report_id: str) -> dict[str, Any]:
    try:
        return get_report_job(report_id)

    except Exception as exc:
        return {
            "status": "failed",
            "report_id": report_id,
            "error": str(exc),
        }


@app.get("/api/report-jobs/{report_id}/pdf")
def download_report_pdf(report_id: str):
    job = get_report_job(report_id)
    pdf_path = job.get("output_pdf_path")

    if not pdf_path:
        return {
            "status": "failed",
            "error": "No PDF path found for this report job.",
        }

    path = ROOT_DIR / pdf_path
    if not path.exists():
        return {
            "status": "failed",
            "error": f"PDF file not found: {pdf_path}",
        }

    return FileResponse(
        path=str(path),
        media_type="application/pdf",
        filename=path.name,
    )


@app.get("/api/report-jobs/{report_id}/markdown")
def download_report_markdown(report_id: str):
    job = get_report_job(report_id)
    md_path = job.get("output_markdown_path")

    if not md_path:
        return {
            "status": "failed",
            "error": "No markdown path found for this report job.",
        }

    path = ROOT_DIR / md_path
    if not path.exists():
        return {
            "status": "failed",
            "error": f"Markdown file not found: {md_path}",
        }

    return FileResponse(
        path=str(path),
        media_type="text/markdown",
        filename=path.name,
    )


@app.get("/demo", response_class=HTMLResponse)
def demo_page() -> str:
    """
    Minimal built-in demo page.

    This is useful when the existing static frontend build is not wired
    to the new algorithm/MLOps API yet.
    """
    return """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Smart Developer Demo</title>
  <style>
    body {
      font-family: -apple-system, BlinkMacSystemFont, sans-serif;
      margin: 40px;
      background: #f7f7f7;
      color: #111;
    }
    .card {
      background: white;
      padding: 18px;
      border-radius: 12px;
      margin-bottom: 14px;
      box-shadow: 0 1px 4px rgba(0,0,0,0.08);
    }
    textarea, input, select {
      width: 100%;
      padding: 10px;
      margin-top: 6px;
      margin-bottom: 12px;
      border: 1px solid #ddd;
      border-radius: 8px;
      box-sizing: border-box;
    }
    button {
      padding: 10px 16px;
      border: 0;
      border-radius: 8px;
      background: #111;
      color: white;
      cursor: pointer;
      margin-right: 8px;
      margin-top: 4px;
    }
    button.secondary {
      background: #555;
    }
    button.danger {
      background: #8b1d1d;
    }
    .muted {
      color: #666;
      font-size: 13px;
      line-height: 1.4;
    }
    .result {
      border-top: 1px solid #eee;
      padding-top: 12px;
      margin-top: 12px;
    }
    .pill {
      display: inline-block;
      background: #f0f0f0;
      border-radius: 999px;
      padding: 4px 8px;
      font-size: 12px;
      margin-right: 6px;
      margin-bottom: 4px;
    }
    .warning {
      background: #fff7e6;
      border: 1px solid #ffd591;
      padding: 10px;
      border-radius: 8px;
      margin-top: 10px;
      color: #7a4b00;
    }
    pre {
      background: #111;
      color: #eee;
      padding: 12px;
      overflow: auto;
      border-radius: 8px;
    }
  </style>
</head>
<body>
  <h1>Smart Developer Demo</h1>
  <p class="muted">
    Fast retrieval + template explanation + feedback logging + PDF report job.
  </p>

  <div class="card">
    <label>Strategy</label>
    <select id="strategy" onchange="fillDefaultQuery()">
      <option value="single_dwelling_rebuild">House redevelopment</option>
      <option value="low_rise_apartment">Low-rise apartment</option>
      <option value="dual_occupancy">Dual occupancy</option>
      <option value="townhouse_multi_dwelling">Townhouse / multi-dwelling</option>
      <option value="granny_flat">Granny flat</option>
      <option value="land_bank_hold">Land bank / hold</option>
      <option value="assembly_opportunity">Assembly opportunity</option>
    </select>

    <label>Query</label>
    <textarea id="query" rows="3">I want a site for detached house redevelopment on standard residential land, with low planning constraints and a suitable lot size.</textarea>

    <label>Locality filter, optional</label>
    <input id="locality" placeholder="e.g. GYMEA BAY, WAITARA, BREAKFAST POINT" />

    <label>
      <input id="locationFallback" type="checkbox" checked style="width: auto; margin-right: 6px;" />
      Use broader fallback if exact locality has insufficient results
    </label>

    <br />

    <button onclick="retrieveSites()">Retrieve Sites</button>
    <button class="secondary" onclick="generateReport()">Generate PDF Report</button>
  </div>

  <div class="card">
    <h2>Response</h2>
    <div id="summary" class="muted">Run a retrieval request to start.</div>
    <div id="results"></div>
  </div>

<script>
let lastResponse = null;
let lastReport = null;

const DEFAULT_QUERIES = {
  single_dwelling_rebuild:
    "I want a site for detached house redevelopment on standard residential land, with low planning constraints and a suitable lot size.",
  low_rise_apartment:
    "I want a site for low-rise apartment redevelopment near a train station, with high development zoning, a large site, and limited planning constraints.",
  dual_occupancy:
    "I want a site for dual occupancy development on residential land with suitable lot size and low planning constraints.",
  townhouse_multi_dwelling:
    "I want a site for townhouse or multi-dwelling redevelopment with suitable zoning, good site scale, and limited planning constraints.",
  granny_flat:
    "I want a residential site suitable for adding a granny flat or secondary dwelling.",
  land_bank_hold:
    "I want a site with long-term land banking potential, supportive planning signals, and limited major constraints.",
  assembly_opportunity:
    "I want a site with assembly or aggregation potential for future redevelopment."
};

function fillDefaultQuery() {
  const strategy = document.getElementById("strategy").value;
  document.getElementById("query").value = DEFAULT_QUERIES[strategy] || DEFAULT_QUERIES.single_dwelling_rebuild;
}

async function retrieveSites() {
  const summary = document.getElementById("summary");
  const container = document.getElementById("results");

  summary.innerText = "Retrieving sites...";
  container.innerHTML = "";

  const payload = {
    strategy: document.getElementById("strategy").value,
    query_text: document.getElementById("query").value,
    top_k: 5,
    recall_k: 1000,
    with_explanations: false,
    use_template_explanations: true,
    locality: document.getElementById("locality").value || null,
    location_fallback: document.getElementById("locationFallback").checked,
    user_id: "demo_user",
    session_id: "web_demo",
    log_request: true
  };

  try {
    const res = await fetch("/api/retrieve-sites", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(payload)
    });

    if (!res.ok) {
      const text = await res.text();
      throw new Error(`HTTP ${res.status}: ${text}`);
    }

    lastResponse = await res.json();
    renderResults(lastResponse);
  } catch (err) {
    summary.innerText = "Request failed.";
    container.innerHTML = `
      <div class="result">
        <h3>Error</h3>
        <p class="muted">${err.message}</p>
      </div>
    `;
    console.error(err);
  }
}

function renderResults(data) {
  const loc = data.metadata?.locality || "none";
  const exactCount = data.metadata?.exact_location_match_count ?? "n/a";
  const fallbackUsed = data.metadata?.location_fallback_used ?? false;
  const warning = data.metadata?.location_warning
    ? ` | ${data.metadata.location_warning}`
    : "";

  document.getElementById("summary").innerText =
    `Request ID: ${data.request_id} | Latency: ${data.metadata?.latency_ms} ms | Location: ${loc} | Exact matches: ${exactCount} | Fallback: ${fallbackUsed} | Logging: ${data.logging?.status || "off"}${warning}`;

  const container = document.getElementById("results");
  container.innerHTML = "";

  if (data.metadata?.location_warning) {
    const warningDiv = document.createElement("div");
    warningDiv.className = "warning";
    warningDiv.innerText = data.metadata.location_warning;
    container.appendChild(warningDiv);
  }

  if (!data.results || data.results.length === 0) {
    const div = document.createElement("div");
    div.className = "result";
    div.innerHTML = `
      <h3>No matching sites found</h3>
      <p class="muted">
        No candidates matched the selected strategy and location filter.
        Try enabling fallback, removing the locality filter, increasing recall, or using a different strategy.
      </p>
    `;
    container.appendChild(div);
    return;
  }

  data.results.forEach((r, idx) => {
    const div = document.createElement("div");
    div.className = "result";

    const score = Number(r.strategy_score || 0).toFixed(1);
    const distance = r.distance_to_station_m != null
      ? `${Number(r.distance_to_station_m).toFixed(0)} m`
      : "n/a";

    div.innerHTML = `
      <h3>${idx + 1}. ${r.base_site_address || r.address || "Unknown site"}</h3>
      <div>
        <span class="pill">RID: ${r.RID}</span>
        <span class="pill">Zoning: ${r.primary_zoning_code || "n/a"}</span>
        <span class="pill">Lot: ${r.lot_size_band || "n/a"}</span>
        <span class="pill">Constraint: ${r.constraint_severity_band || "n/a"}</span>
        <span class="pill">Station: ${r.station_distance_band || "n/a"} (${distance})</span>
      </div>
      <p><b>Strategy score:</b> ${score}</p>
      <p>${r.fast_explanation || ""}</p>
      <button onclick="sendFeedback('${data.request_id}', '${r.RID}', ${idx + 1}, 'save')">Save</button>
      <button class="danger" onclick="sendFeedback('${data.request_id}', '${r.RID}', ${idx + 1}, 'dismiss')">Dismiss</button>
    `;
    container.appendChild(div);
  });
}

async function sendFeedback(requestId, rid, rank, eventType) {
  try {
    const res = await fetch("/api/feedback", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        request_id: requestId,
        rid: rid,
        rank_position: rank,
        event_type: eventType,
        event_value: { source: "web_demo" },
        user_note: `Web demo ${eventType}`,
        user_id: "demo_user",
        session_id: "web_demo"
      })
    });

    const data = await res.json();

    if (data.status === "logged") {
      alert(`Feedback logged: ${data.feedback_id}`);
    } else {
      alert(`Feedback failed: ${data.error}`);
    }
  } catch (err) {
    alert(`Feedback request failed: ${err.message}`);
    console.error(err);
  }
}

async function generateReport() {
  if (!lastResponse) {
    alert("Run retrieval first.");
    return;
  }

  try {
    const res = await fetch("/api/report-jobs", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        request_id: lastResponse.request_id,
        explanation_mode: "template",
        output_markdown: true,
        output_pdf: true,
        audience: "developer",
        title: "Smart Developer Site Recommendation Report"
      })
    });

    lastReport = await res.json();

    if (lastReport.status === "ready") {
      window.open(`/api/report-jobs/${lastReport.report_id}/pdf`, "_blank");
    } else {
      alert(JSON.stringify(lastReport, null, 2));
    }
  } catch (err) {
    alert(`Report generation failed: ${err.message}`);
    console.error(err);
  }
}
</script>
</body>
</html>
    """


@app.get("/{path:path}")
def serve_frontend(path: str):
    """
    Serve existing built frontend.

    If the frontend has client-side routing, unknown paths fall back to index.html.
    Use /demo for the built-in temporary demo page.
    """
    requested = FRONTEND_DIR / path

    if path and requested.exists() and requested.is_file():
        return FileResponse(str(requested))

    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))

    return {
        "status": "ok",
        "message": "Frontend index.html not found. Use /demo for built-in demo page.",
    }