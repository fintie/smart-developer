import { useEffect, useMemo, useRef, useState } from "react";
import {
  exportReportPdf,
  generateAIPropertySummary,
  registerUser,
  searchSites,
  sendFeedback,
  sendRecommendationFeedback,
  type RegisterUserResponse,
  type SearchResponse,
  type SiteResult,
} from "./api";
import "./App.css";
import "leaflet/dist/leaflet.css";
import { ResultsMap } from "./components/ResultsMap";
import { SiteCard } from "./components/SiteCard";
import type { AISummaryState } from "./components/AISummaryPanel";
import { SearchPanel } from "./components/SearchPanel";
import { RecommendationFeedbackModal } from "./components/RecommendationFeedbackModal";
import { formatProfileLabel } from "./lib/format";
import { STRATEGIES, type RankingProfile } from "./lib/strategies";

function siteStateKey(site: SiteResult, index: number) {
  return `${site.RID ?? "site"}-${index}`;
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
  const [registerUsername, setRegisterUsername] = useState("");
  const [registerPassword, setRegisterPassword] = useState("");
  const [registerLoading, setRegisterLoading] = useState(false);
  const [registerError, setRegisterError] = useState("");
  const [registerResult, setRegisterResult] =
    useState<RegisterUserResponse | null>(null);

  const [searchResponse, setSearchResponse] = useState<SearchResponse | null>(null);
  const [aiSummaries, setAiSummaries] = useState<Record<string, AISummaryState>>(
    {},
  );
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

  async function handleRegister(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setRegisterLoading(true);
    setRegisterError("");
    setRegisterResult(null);

    try {
      const result = await registerUser({
        username: registerUsername.trim(),
        password: registerPassword,
      });
      setRegisterResult(result);
      setRegisterPassword("");
    } catch (err) {
      setRegisterError(err instanceof Error ? err.message : "Register failed");
    } finally {
      setRegisterLoading(false);
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
          <SearchPanel
            strategy={strategy}
            queryText={queryText}
            locality={locality}
            topK={topK}
            rankingProfile={rankingProfile}
            loading={loading}
            onStrategyChange={handleStrategyChange}
            onQueryTextChange={setQueryText}
            onLocalityChange={setLocality}
            onTopKChange={setTopK}
            onRankingProfileChange={setRankingProfile}
            onSearch={handleSearch}
          />

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

        <section className="profile-section" id="profile">
          <div className="profile-shell">
            <div>
              <p className="eyebrow">Profile</p>
              <h2>Create Account</h2>
              <p className="profile-copy">
                Register a demo user through the FastAPI backend. The backend
                hashes the password, creates a user record, and returns a token.
              </p>
            </div>

            <form className="register-card" onSubmit={handleRegister}>
              <label>
                Username
                <input
                  value={registerUsername}
                  onChange={(event) => setRegisterUsername(event.target.value)}
                  placeholder="Enter username"
                  autoComplete="username"
                  required
                  maxLength={50}
                />
              </label>

              <label>
                Password
                <input
                  value={registerPassword}
                  onChange={(event) => setRegisterPassword(event.target.value)}
                  placeholder="Enter password"
                  type="password"
                  autoComplete="new-password"
                  required
                  maxLength={72}
                />
              </label>

              <button
                className="primary-button"
                type="submit"
                disabled={
                  registerLoading ||
                  !registerUsername.trim() ||
                  !registerPassword
                }
              >
                {registerLoading ? "Registering..." : "Register"}
              </button>

              {registerResult && (
                <div className="register-success">
                  <span>{registerResult.message}</span>
                  <strong>{registerResult.data.userInfo.username}</strong>
                  <small>Token: {registerResult.data.token}</small>
                </div>
              )}

              {registerError && (
                <div className="register-error">{registerError}</div>
              )}
            </form>
          </div>
        </section>
      </div>

      {feedbackDialogOpen && searchResponse?.feedback_prompt && (
        <RecommendationFeedbackModal
          title={searchResponse.feedback_prompt.title}
          rating={recommendationRating}
          note={recommendationNote}
          submitting={recommendationFeedbackSubmitting}
          onRatingChange={setRecommendationRating}
          onNoteChange={setRecommendationNote}
          onSubmit={handleRecommendationFeedbackSubmit}
          onClose={() => setFeedbackDialogOpen(false)}
        />
      )}
    </main>
  );
}

export default App;
