import { useEffect, useMemo, useRef, useState } from "react";
import {
  exportReportPdf,
  generateAIPropertySummary,
  searchSites,
  sendFeedback,
  sendRecommendationFeedback,
  type AIPropertySummaryResponse,
  type SearchResponse,
  type SiteResult,
} from "./api";
import "./App.css";
import "leaflet/dist/leaflet.css";
import { ResultsMap } from "./components/ResultsMap";

type RankingProfile =
  | "balanced"
  | "policy_upside"
  | "budget_sensitive"
  | "high_value";

const STRATEGIES = [
  {
    value: "single_dwelling_rebuild",
    label: "Single dwelling rebuild",
    query:
      "I want a site for detached house redevelopment on standard residential land, with low planning constraints and a suitable lot size.",
  },
  {
    value: "low_rise_apartment",
    label: "Low-rise apartment",
    query:
      "I want a site for low-rise apartment redevelopment near a train station, with high development zoning, a large site, and limited planning constraints.",
  },
  {
    value: "dual_occupancy",
    label: "Dual occupancy",
    query:
      "I want a residential site suitable for dual occupancy, with appropriate zoning, a suitable lot size, and low planning constraints.",
  },
  {
    value: "granny_flat",
    label: "Granny flat",
    query:
      "I want a residential site suitable for a granny flat or secondary dwelling, with low constraints and a practical lot size.",
  },
];

const RANKING_PROFILES: Array<{
  value: RankingProfile;
  label: string;
  description: string;
}> = [
  {
    value: "balanced",
    label: "Balanced",
    description: "Balances strategy fit, policy upside, value, and cost.",
  },
  {
    value: "policy_upside",
    label: "Policy Upside",
    description: "Prioritises sites with stronger planning-policy signals.",
  },
  {
    value: "budget_sensitive",
    label: "Budget Sensitive",
    description: "Prioritises lower-cost and more cost-efficient opportunities.",
  },
  {
    value: "high_value",
    label: "High Value",
    description: "Prioritises sites with stronger market and redevelopment value signals.",
  },
];

function formatNumber(value: unknown, digits = 1) {
  if (typeof value !== "number" || Number.isNaN(value)) return "N/A";
  return value.toFixed(digits);
}

function formatMoney(value?: number | null) {
  if (typeof value !== "number" || Number.isNaN(value)) return "N/A";

  return new Intl.NumberFormat("en-AU", {
    style: "currency",
    currency: "AUD",
    maximumFractionDigits: 0,
  }).format(value);
}

function formatScore(value?: number | null) {
  if (typeof value !== "number" || Number.isNaN(value)) return "N/A";
  return value.toFixed(1);
}

function formatDistance(value: unknown) {
  if (typeof value !== "number" || Number.isNaN(value)) return "N/A";
  return `${Math.round(value)} m`;
}

function formatProfileLabel(value?: string | null) {
  if (!value) return "N/A";
  return (
    RANKING_PROFILES.find((profile) => profile.value === value)?.label ??
    value.replaceAll("_", " ")
  );
}

function splitAssessmentText(value: string) {
  return value
    .split(/(?<=[.!?])\s+/)
    .map((sentence) => sentence.trim())
    .filter(Boolean);
}

function SiteCard({
  site,
  index,
  requestId,
  onFeedback,
  onAISummary,
  aiSummaryState,
}: {
  site: SiteResult;
  index: number;
  requestId: string;
  onFeedback: (eventType: string, site: SiteResult, index: number) => void;
  onAISummary: (site: SiteResult, index: number) => void;
  aiSummaryState?: {
    loading: boolean;
    data?: AIPropertySummaryResponse;
    error?: string;
  };
}) {
  const address = site.base_site_address || site.address || "Unknown address";

  const explanation =
    site.agent_pitch ||
    site.cost_value_explanation ||
    site.policy_explanation ||
    site.fast_explanation ||
    site.explanation ||
    "No explanation available.";
  const assessmentSentences = splitAssessmentText(explanation);
  const assessmentLead = assessmentSentences[0] ?? explanation;
  const assessmentDetails = assessmentSentences.slice(1);

  return (
    <article className="site-card">
      <div className="site-card-header">
        <div>
          <div className="rank">#{index + 1}</div>
          <h3>{address}</h3>
        </div>

        <div className="score-box">
          <span>Opportunity</span>
          <strong>
            {formatNumber(site.agent_opportunity_score ?? site.strategy_score)}
          </strong>
        </div>
      </div>

      <div className="tag-row">
        <span>Zoning: {site.primary_zoning_code ?? "N/A"}</span>
        <span>Lot: {site.lot_size_band ?? "N/A"}</span>
        <span>Constraints: {site.constraint_severity_band ?? "N/A"}</span>
        <span>Station: {formatDistance(site.distance_to_station_m)}</span>
        <span>
          Policy:{" "}
          {typeof site.policy_upside_score === "number"
            ? `${site.policy_signal_band ?? "signal"} (${site.policy_upside_score.toFixed(0)})`
            : "N/A"}
        </span>
        <span>
          Profile: {formatProfileLabel(site.ranking_profile)}
        </span>
        <span>
          Map:{" "}
          {typeof site.latitude === "number" && typeof site.longitude === "number"
            ? `${site.latitude.toFixed(4)}, ${site.longitude.toFixed(4)}`
            : "N/A"}
        </span>
      </div>

      <div className="meta-grid">
        <div>
          <span>Opportunity</span>
          <strong>{formatScore(site.agent_opportunity_score ?? site.strategy_score)}</strong>
        </div>
        <div>
          <span>Strategy fit</span>
          <strong>{formatScore(site.strategy_score)}</strong>
        </div>
        <div>
          <span>Policy upside</span>
          <strong>
            {formatScore(site.policy_upside_score)}
            {site.policy_signal_band ? ` · ${site.policy_signal_band}` : ""}
          </strong>
        </div>
        <div>
          <span>Value potential</span>
          <strong>
            {formatScore(site.value_potential_score)}
            {site.value_potential_band ? ` · ${site.value_potential_band}` : ""}
          </strong>
        </div>
        <div>
          <span>Cost efficiency</span>
          <strong>{formatScore(site.cost_efficiency_score)}</strong>
        </div>
        <div>
          <span>Cost risk</span>
          <strong>
            {formatScore(site.cost_risk_score)}
            {site.cost_band ? ` · ${site.cost_band}` : ""}
          </strong>
        </div>
      </div>

      <div className="economics-box">
        <div className="section-title">Economics</div>

        <div className="economics-grid">
          <div>
            <span>ML transaction value: </span>
            <strong>{formatMoney(site.ml_estimated_market_value)}</strong>
          </div>
          <div>
            <span>Site acquisition proxy: </span>
            <strong>{formatMoney(site.estimated_acquisition_cost)}</strong>
          </div>
          <div>
            <span>Development cost: </span>
            <strong>{formatMoney(site.estimated_development_cost)}</strong>
          </div>
          <div>
            <span>Total project cost: </span>
            <strong>{formatMoney(site.estimated_total_project_cost)}</strong>
          </div>
        </div>

        {site.ml_value_confidence && (
          <p className="small-note">
            ML value confidence: {site.ml_value_confidence}
            {typeof site.ml_value_error_pct === "number"
              ? ` · median error approx ${(site.ml_value_error_pct * 100).toFixed(1)}%`
              : ""}
          </p>
        )}
      </div>

      <section className="assessment-card">
        <div className="assessment-header">
          <div>
            <h4>Recommendation rationale</h4>
          </div>
        </div>

        <p className="assessment-lead">{assessmentLead}</p>

        {assessmentDetails.length > 0 && (
          <ul className="assessment-list">
            {assessmentDetails.map((sentence, sentenceIndex) => (
              <li key={sentenceIndex}>{sentence}</li>
            ))}
          </ul>
        )}
      </section>

      {site.policy_evidence && site.policy_evidence.length > 0 && (
        <details className="policy-evidence">
          <summary>
            Policy evidence ({site.policy_evidence.length})
          </summary>

          <div className="policy-evidence-list">
            {site.policy_evidence.map((evidence, evidenceIndex) => (
              <div key={evidenceIndex} className="policy-evidence-item">
                <strong>
                  {evidence.policy_name || evidence.policy_id || "Policy source"}
                </strong>

                {evidence.snippet && <p>{evidence.snippet}</p>}

                {evidence.source_url && (
                  <a
                    href={evidence.source_url}
                    target="_blank"
                    rel="noreferrer"
                  >
                    Open source
                  </a>
                )}
              </div>
            ))}
          </div>
        </details>
      )}

      <div className="meta-grid compact">
        <div>
          <span>Top strategy</span>
          <strong>{site.top_strategy ?? "N/A"}</strong>
        </div>
        <div>
          <span>Within 800m</span>
          <strong>{site.within_800m_catchment ? "Yes" : "No"}</strong>
        </div>
        <div>
          <span>Heritage</span>
          <strong>{site.heritage_flag ? "Yes" : "No"}</strong>
        </div>
        <div>
          <span>Flood</span>
          <strong>{site.flood_flag ? "Yes" : "No"}</strong>
        </div>
      </div>

      <div className="button-row">
        <button onClick={() => onAISummary(site, index)} disabled={aiSummaryState?.loading}>
          {aiSummaryState?.loading ? "Generating..." : "Generate AI Summary"}
        </button>
        <button onClick={() => onFeedback("click", site, index)}>Click</button>
        <button onClick={() => onFeedback("save", site, index)}>Save</button>
        <button className="secondary" onClick={() => onFeedback("dismiss", site, index)}>
          Dismiss
        </button>
      </div>

      {aiSummaryState?.error && (
        <div className="ai-summary-error">{aiSummaryState.error}</div>
      )}

      {aiSummaryState?.data && (
        <section className="ai-property-card">
          <div className="ai-property-header">
            <div>
              <span className="section-kicker">AI property summary</span>
              <h4>{aiSummaryState.data.summary.headline}</h4>
            </div>
            <span className="ai-source">
              {aiSummaryState.data.source === "google_gemini"
                ? aiSummaryState.data.model
                : "local structured summary"}
            </span>
          </div>

          <div className="ai-value-panel">
            <span>Estimated value</span>
            <strong>{aiSummaryState.data.summary.value_estimate.label}</strong>
            {aiSummaryState.data.summary.value_estimate.range_label && (
              <p>{aiSummaryState.data.summary.value_estimate.range_label}</p>
            )}
            <small>
              Confidence:{" "}
              {aiSummaryState.data.summary.value_estimate.confidence ?? "N/A"}
            </small>
          </div>

          <div className="ai-summary-grid">
            <div>
              <h5>Basic information</h5>
              <ul>
                {aiSummaryState.data.summary.basic_info.map((item, itemIndex) => (
                  <li key={itemIndex}>{item}</li>
                ))}
              </ul>
            </div>
            <div>
              <h5>User requirement fit</h5>
              <p>{aiSummaryState.data.summary.requirement_match}</p>
              <p>{aiSummaryState.data.summary.value_estimate.explanation}</p>
            </div>
            <div>
              <h5>Opportunity</h5>
              <ul>
                {aiSummaryState.data.summary.opportunity_notes.map(
                  (item, itemIndex) => (
                    <li key={itemIndex}>{item}</li>
                  ),
                )}
              </ul>
            </div>
            <div>
              <h5>Risks</h5>
              <ul>
                {aiSummaryState.data.summary.risk_notes.map((item, itemIndex) => (
                  <li key={itemIndex}>{item}</li>
                ))}
              </ul>
            </div>
          </div>

          <p className="ai-disclaimer">
            {aiSummaryState.data.summary.disclaimer}
          </p>

          <section className="ai-suggestion-card">
            <div className="ai-suggestion-header">
              <div>
                <span className="section-kicker">AI suggestion</span>
                <h5>{aiSummaryState.data.ai_suggestion.headline}</h5>
              </div>
            </div>

            <p>{aiSummaryState.data.ai_suggestion.suggestion}</p>

            {aiSummaryState.data.ai_suggestion.next_steps.length > 0 && (
              <div className="ai-suggestion-section">
                <strong>Suggested next steps</strong>
                <ul>
                  {aiSummaryState.data.ai_suggestion.next_steps.map(
                    (step, stepIndex) => (
                      <li key={stepIndex}>{step}</li>
                    ),
                  )}
                </ul>
              </div>
            )}

            {aiSummaryState.data.external_sources.length > 0 && (
              <details className="ai-source-list">
                <summary>
                  External sources ({aiSummaryState.data.external_sources.length})
                </summary>
                <div>
                  {aiSummaryState.data.external_sources.map((source, sourceIndex) => (
                    <a
                      key={sourceIndex}
                      href={source.link}
                      target="_blank"
                      rel="noreferrer"
                    >
                      <strong>{source.title}</strong>
                      {source.snippet && <span>{source.snippet}</span>}
                    </a>
                  ))}
                </div>
              </details>
            )}

          </section>
        </section>
      )}

      <div className="rid">RID: {site.RID ?? "N/A"} · Request: {requestId}</div>
    </article>
  );
}

function App() {
  const [strategy, setStrategy] = useState(STRATEGIES[0].value);
  const [queryText, setQueryText] = useState(STRATEGIES[0].query);
  const [locality, setLocality] = useState("");
  const [topK, setTopK] = useState(5);
  const [rankingProfile, setRankingProfile] =
    useState<RankingProfile>("balanced");

  const [loading, setLoading] = useState(false);
  const [feedbackMessage, setFeedbackMessage] = useState("");
  const [reportMessage, setReportMessage] = useState("");
  const [error, setError] = useState("");
  const [feedbackDialogOpen, setFeedbackDialogOpen] = useState(false);
  const [recommendationRating, setRecommendationRating] = useState<number | null>(
    null,
  );
  const [recommendationNote, setRecommendationNote] = useState("");
  const [recommendationFeedbackSubmitting, setRecommendationFeedbackSubmitting] =
    useState(false);

  const [searchResponse, setSearchResponse] = useState<SearchResponse | null>(null);
  const [aiSummaries, setAiSummaries] = useState<
    Record<
      string,
      {
        loading: boolean;
        data?: AIPropertySummaryResponse;
        error?: string;
      }
    >
  >({});
  const feedbackDialogTimerRef = useRef<number | null>(null);
  const successMessageTimerRef = useRef<number | null>(null);

  useEffect(() => {
    return () => {
      if (feedbackDialogTimerRef.current !== null) {
        window.clearTimeout(feedbackDialogTimerRef.current);
      }
      if (successMessageTimerRef.current !== null) {
        window.clearTimeout(successMessageTimerRef.current);
      }
    };
  }, []);

  useEffect(() => {
    if (!feedbackMessage && !reportMessage) return;

    if (successMessageTimerRef.current !== null) {
      window.clearTimeout(successMessageTimerRef.current);
    }

    successMessageTimerRef.current = window.setTimeout(() => {
      setFeedbackMessage("");
      setReportMessage("");
      successMessageTimerRef.current = null;
    }, 15_000);

    return () => {
      if (successMessageTimerRef.current !== null) {
        window.clearTimeout(successMessageTimerRef.current);
        successMessageTimerRef.current = null;
      }
    };
  }, [feedbackMessage, reportMessage]);

  const selectedStrategyLabel = useMemo(() => {
    return STRATEGIES.find((item) => item.value === strategy)?.label ?? strategy;
  }, [strategy]);

  const selectedRankingProfile = useMemo(() => {
    return (
      RANKING_PROFILES.find((profile) => profile.value === rankingProfile) ??
      RANKING_PROFILES[0]
    );
  }, [rankingProfile]);

  function handleStrategyChange(value: string) {
    setStrategy(value);
    const selected = STRATEGIES.find((item) => item.value === value);
    if (selected) {
      setQueryText(selected.query);
    }
  }

  async function handleSearch() {
    setLoading(true);
    setError("");
    setFeedbackMessage("");
    setReportMessage("");
    setSearchResponse(null);
    setFeedbackDialogOpen(false);
    setRecommendationRating(null);
    setRecommendationNote("");
    setAiSummaries({});
    if (feedbackDialogTimerRef.current !== null) {
      window.clearTimeout(feedbackDialogTimerRef.current);
      feedbackDialogTimerRef.current = null;
    }

    try {
      const response = await searchSites({
        strategy,
        query_text: queryText,
        top_k: topK,
        recall_k: 1000,
        locality: locality.trim() ? locality.trim().toUpperCase() : null,
        address_contains: null,
        with_explanations: false,
        use_template_explanations: true,
        ranking_profile: rankingProfile,
        log_request: true,
        debug: false,
        user_id: "demo_user",
        session_id: "frontend_demo",
      });

      setSearchResponse(response);
      if (response.feedback_prompt?.enabled && response.results.length > 0) {
        feedbackDialogTimerRef.current = window.setTimeout(() => {
          setFeedbackDialogOpen(true);
          feedbackDialogTimerRef.current = null;
        }, 60_000);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Search failed");
    } finally {
      setLoading(false);
    }
  }

  async function handleFeedback(eventType: string, site: SiteResult, index: number) {
    if (!searchResponse?.request_id) return;

    setFeedbackMessage("");
    setError("");

    try {
      const result = await sendFeedback({
        request_id: searchResponse.request_id,
        event_type: eventType,
        rid: site.RID ?? null,
        rank_position: index + 1,
        event_value: {
          address: site.base_site_address || site.address,
          strategy_score: site.strategy_score,
          agent_opportunity_score: site.agent_opportunity_score,
          ranking_profile: site.ranking_profile,
          cost_efficiency_score: site.cost_efficiency_score,
        },
        user_note: null,
        user_id: "demo_user",
        session_id: "frontend_demo",
      });

      setFeedbackMessage(
        `Feedback logged: ${eventType} (${result.feedback_id ?? "saved"})`
      );
    } catch (err) {
      setError(err instanceof Error ? err.message : "Feedback failed");
    }
  }

  function siteStateKey(site: SiteResult, index: number) {
    return `${site.RID ?? "site"}-${index}`;
  }

  async function handleAISummary(site: SiteResult, index: number) {
    const key = siteStateKey(site, index);

    setAiSummaries((current) => ({
      ...current,
      [key]: {
        ...current[key],
        loading: true,
        error: "",
      },
    }));
    setError("");

    try {
      const data = await generateAIPropertySummary({
        query_text: queryText,
        user_requirements: queryText,
        site,
        user_id: "demo_user",
        session_id: "frontend_demo",
      });

      setAiSummaries((current) => ({
        ...current,
        [key]: {
          loading: false,
          data,
        },
      }));
    } catch (err) {
      setAiSummaries((current) => ({
        ...current,
        [key]: {
          loading: false,
          error:
            err instanceof Error ? err.message : "AI property summary failed",
        },
      }));
    }
  }

  async function handleRecommendationFeedbackSubmit() {
    if (!searchResponse?.request_id || recommendationRating === null) return;

    setRecommendationFeedbackSubmitting(true);
    setError("");

    try {
      const result = await sendRecommendationFeedback({
        request_id: searchResponse.request_id,
        rating: recommendationRating,
        user_note: recommendationNote.trim() || null,
        user_id: "demo_user",
        session_id: "frontend_demo",
      });

      setFeedbackMessage(
        `Recommendation feedback logged: ${result.rating_label ?? recommendationRating}`,
      );
      setFeedbackDialogOpen(false);
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Recommendation feedback failed",
      );
    } finally {
      setRecommendationFeedbackSubmitting(false);
    }
  }

  async function handleCreateReport() {
    if (!searchResponse?.results?.length) {
      setReportMessage("Run a search first before generating a report.");
      return;
    }

    setReportMessage("Generating PDF report...");
    setError("");

    try {
      await exportReportPdf({
        strategy,
        query_text: queryText,
        results: searchResponse.results,
        title: "Smart Developer Site Recommendation Report",
        audience: "developer / real estate agent",
        output_format: "pdf",
        max_rows: Math.min(searchResponse.results.length, 5),
      });

      setReportMessage("PDF report downloaded.");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Report generation failed");
      setReportMessage("");
    }
  }

  const latency = searchResponse?.metadata?.latency_ms;
  const resultCount = searchResponse?.results?.length ?? 0;
  const responseProfile =
    typeof searchResponse?.metadata?.ranking_profile === "string"
      ? searchResponse.metadata.ranking_profile
      : "";

  return (
    <main className="page">
      <aside className="app-nav" aria-label="Primary navigation">
        <div className="nav-brand">
          <span>SD</span>
        </div>
        <nav>
          <a className="active" href="#search">
            Search
          </a>
          <a href="#saved">Saved</a>
          <a href="#profile">Profile</a>
          <a href="#setting">Setting</a>
        </nav>
      </aside>

      <div className="workspace">
        <section className="hero">
          <div>
            <p className="eyebrow">NextGenius · Smart Developer</p>
            <h1>AI Site Recommendation Platform</h1>
            <p className="subtitle">
              Search development sites with policy-aware ranking, economics-aware
              scoring, ML market value estimates, and agent-facing explanations.
            </p>
          </div>

          <div className="status-card">
            <span>Demo flow</span>
            <strong>Search → Feedback → Report</strong>
          </div>
        </section>

        <section className="layout" id="search">
          <aside className="panel search-panel">
            <h2>Search Criteria</h2>

            <label>
              Strategy
              <select
                value={strategy}
                onChange={(event) => handleStrategyChange(event.target.value)}
              >
                {STRATEGIES.map((item) => (
                  <option key={item.value} value={item.value}>
                    {item.label}
                  </option>
                ))}
              </select>
            </label>

            <label>
              Locality filter
              <input
                value={locality}
                onChange={(event) => setLocality(event.target.value)}
                placeholder="e.g. WOLLI CREEK, WAITARA, GYMEA BAY"
              />
            </label>

            <label>
              Ranking Profile
              <select
                value={rankingProfile}
                onChange={(event) =>
                  setRankingProfile(event.target.value as RankingProfile)
                }
              >
                {RANKING_PROFILES.map((profile) => (
                  <option key={profile.value} value={profile.value}>
                    {profile.label}
                  </option>
                ))}
              </select>
            </label>

            <div className="profile-note">
              <strong>{selectedRankingProfile.label}:</strong>{" "}
              {selectedRankingProfile.description}
            </div>

            <label>
              Top K
              <input
                type="number"
                min={1}
                max={20}
                value={topK}
                onChange={(event) => setTopK(Number(event.target.value))}
              />
            </label>

            <label>
              Query text
              <textarea
                value={queryText}
                onChange={(event) => setQueryText(event.target.value)}
                rows={7}
              />
            </label>

            <button className="primary-button" onClick={handleSearch} disabled={loading}>
              {loading ? "Searching..." : "Find Sites"}
            </button>

            <div className="demo-note">
              <strong>Current mode:</strong>
              <br />
              Two-tower retrieval + DCN reranking + NSW policy RAG + ML market value
              model + development cost and cost-efficiency scoring.
            </div>
          </aside>

          <section className="results-panel">
            <div className="results-header">
              <div>
                <p className="eyebrow">Results</p>
                <h2>{selectedStrategyLabel}</h2>
                {responseProfile && (
                  <p className="result-profile">
                    Ranking profile: {formatProfileLabel(responseProfile)}
                  </p>
                )}
              </div>

              {searchResponse && (
                <button
                  className="report-button"
                  onClick={handleCreateReport}
                  disabled={!searchResponse.results.length}
                >
                  Generate Report
                </button>
              )}
            </div>

            {searchResponse && (
              <div className="summary-row">
                <div>
                  <span>Request ID</span>
                  <strong>{searchResponse.request_id}</strong>
                </div>
                <div>
                  <span>Results</span>
                  <strong>{resultCount}</strong>
                </div>
                <div>
                  <span>Latency</span>
                  <strong>
                    {typeof latency === "number" ? `${latency.toFixed(1)} ms` : "N/A"}
                  </strong>
                </div>
                <div>
                  <span>Logging</span>
                  <strong>{String(searchResponse.logging?.status ?? "N/A")}</strong>
                </div>
              </div>
            )}

            {searchResponse && searchResponse.results.length > 0 && (
              <ResultsMap results={searchResponse.results} />
            )}

            {feedbackMessage && <div className="success-message">{feedbackMessage}</div>}
            {reportMessage && <div className="success-message">{reportMessage}</div>}
            {error && <div className="error-message">{error}</div>}

            {!searchResponse && !loading && (
              <div className="empty-state">
                Run a search to display ranked development sites.
              </div>
            )}

            {searchResponse && !loading && searchResponse.results.length === 0 && (
              <div className="empty-state">
                No exact matches found for this locality. Try removing the locality
                filter or using a nearby suburb.
              </div>
            )}

            {loading && <div className="empty-state">Loading ranked sites...</div>}

            <div className="site-list">
              {searchResponse?.results?.map((site, index) => (
                <SiteCard
                  key={`${site.RID ?? index}-${index}`}
                  site={site}
                  index={index}
                  requestId={searchResponse.request_id}
                  onFeedback={handleFeedback}
                  onAISummary={handleAISummary}
                  aiSummaryState={aiSummaries[siteStateKey(site, index)]}
                />
              ))}
            </div>
          </section>
        </section>
      </div>

      {feedbackDialogOpen && searchResponse?.feedback_prompt && (
        <div className="feedback-modal-backdrop" role="presentation">
          <section
            className="feedback-modal"
            aria-labelledby="recommendation-feedback-title"
            role="dialog"
            aria-modal="true"
          >
            <div className="feedback-modal-header">
              <div>
                <p className="eyebrow">Feedback</p>
                <h2 id="recommendation-feedback-title">
                  {searchResponse.feedback_prompt.title}
                </h2>
              </div>
              <button
                className="icon-button"
                aria-label="Close feedback dialog"
                onClick={() => setFeedbackDialogOpen(false)}
              >
                x
              </button>
            </div>

            <div className="rating-row" aria-label="Recommendation rating">
              {[1, 2, 3, 4, 5].map((rating) => (
                <button
                  key={rating}
                  className={
                    recommendationRating !== null && rating <= recommendationRating
                      ? "star-button selected"
                      : "star-button"
                  }
                  aria-label={`Rate ${rating} out of 5`}
                  onClick={() => setRecommendationRating(rating)}
                  type="button"
                >
                  ★
                </button>
              ))}
            </div>

            <div className="rating-scale-labels">
              <span>Very unsatisfied</span>
              <span>Very satisfied</span>
            </div>

            <label className="feedback-note-label">
              Note
              <textarea
                value={recommendationNote}
                onChange={(event) => setRecommendationNote(event.target.value)}
                rows={3}
                placeholder="Optional"
              />
            </label>

            <div className="feedback-actions">
              <button
                className="feedback-secondary"
                onClick={() => setFeedbackDialogOpen(false)}
                type="button"
              >
                Not now
              </button>
              <button
                className="primary-button feedback-submit"
                disabled={
                  recommendationRating === null ||
                  recommendationFeedbackSubmitting
                }
                onClick={handleRecommendationFeedbackSubmit}
                type="button"
              >
                {recommendationFeedbackSubmitting ? "Submitting..." : "Submit"}
              </button>
            </div>
          </section>
        </div>
      )}
    </main>
  );
}

export default App;
