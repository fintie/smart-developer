import { useEffect, useMemo, useRef, useState } from "react";
import {
  changePassword,
  exportReportPdf,
  generateAIPropertySummary,
  getCollections,
  loginUser,
  registerUser,
  removeCollection,
  saveCollection,
  searchSites,
  sendFeedback,
  sendRecommendationFeedback,
  type SearchResponse,
  type SiteResult,
  type CollectionItem,
} from "./api";
import "./App.css";
import "leaflet/dist/leaflet.css";
import { ResultsMap } from "./components/ResultsMap";
import { SiteCard } from "./components/SiteCard";
import type { AISummaryState } from "./components/AISummaryPanel";
import { SearchPanel } from "./components/SearchPanel";
import { RecommendationFeedbackModal } from "./components/RecommendationFeedbackModal";
import { AuthDialog } from "./components/AuthDialog";
import { DashboardLayout, type SignedInUser } from "./components/DashboardLayout";
import { CollectionPage } from "./components/CollectionPage";
import { formatProfileLabel } from "./lib/format";
import { STRATEGIES, type RankingProfile } from "./lib/strategies";

function siteStateKey(site: SiteResult, index: number) {
  return `${site.RID ?? "site"}-${index}`;
}

function App() {
  const [authDialogOpen, setAuthDialogOpen] = useState(false);
  const [authLoading, setAuthLoading] = useState(false);
  const [authError, setAuthError] = useState("");
  const [authMessage, setAuthMessage] = useState("");
  const [activePage, setActivePage] = useState<"dashboard" | "collection">("dashboard");
  const [collections, setCollections] = useState<CollectionItem[]>([]);
  const [collectionsLoading, setCollectionsLoading] = useState(false);
  const [savingRid, setSavingRid] = useState<string | null>(null);
  const [currentUser, setCurrentUser] = useState<SignedInUser | null>(() => {
    try {
      const stored = window.localStorage.getItem("smartDeveloperUser");
      return stored ? JSON.parse(stored) as SignedInUser : null;
    } catch {
      return null;
    }
  });
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
  const [authToken, setAuthToken] = useState(() => window.localStorage.getItem("smartDeveloperToken") ?? "");

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

  useEffect(() => {
    if (!currentUser || !authToken) {
      setCollections([]);
      return;
    }
    let cancelled = false;
    setCollectionsLoading(true);
    getCollections(authToken)
      .then((items) => { if (!cancelled) setCollections(items); })
      .catch((err) => { if (!cancelled) setError(err instanceof Error ? err.message : "Could not load collection"); })
      .finally(() => { if (!cancelled) setCollectionsLoading(false); });
    return () => { cancelled = true; };
  }, [currentUser, authToken]);

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

  async function handleAISummary(site: SiteResult, index: number, stateKey?: string) {
    const key = stateKey ?? siteStateKey(site, index);

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

  function collectionRid(site: SiteResult) {
    return String(site.RID ?? site.base_site_address ?? site.address ?? "unknown");
  }

  async function handleSave(site: SiteResult, index: number) {
    if (!currentUser || !authToken) {
      handleOpenAccount();
      return;
    }
    const rid = collectionRid(site);
    setSavingRid(rid);
    setError("");
    try {
      const saved = await saveCollection(authToken, site);
      setCollections((items) => items.some((item) => item.id === saved.id) ? items : [saved, ...items]);
      setFeedbackMessage("Site added to your collection.");
      await handleFeedback("save", site, index);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not save site");
    } finally {
      setSavingRid(null);
    }
  }

  async function handleRemoveCollection(item: CollectionItem) {
    try {
      await removeCollection(authToken, item.id);
      setCollections((items) => items.filter((entry) => entry.id !== item.id));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not remove site");
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

  function saveAuthenticatedUser(user: SignedInUser, token: string) {
    setCurrentUser(user);
    setAuthToken(token);
    window.localStorage.setItem("smartDeveloperUser", JSON.stringify(user));
    window.localStorage.setItem("smartDeveloperToken", token);
    setAuthDialogOpen(false);
    setAuthError("");
  }

  async function handleLogin(username: string, password: string) {
    setAuthLoading(true);
    setAuthError("");
    try {
      const result = await loginUser({ username, password });
      saveAuthenticatedUser(result.data.user_info, result.data.token);
    } catch (err) {
      setAuthError(err instanceof Error ? err.message : "Login failed");
    } finally {
      setAuthLoading(false);
    }
  }

  async function handleDialogRegister(username: string, password: string) {
    setAuthLoading(true);
    setAuthError("");
    try {
      const result = await registerUser({ username, password });
      saveAuthenticatedUser(result.data.userInfo, result.data.token);
    } catch (err) {
      setAuthError(err instanceof Error ? err.message : "Registration failed");
    } finally {
      setAuthLoading(false);
    }
  }

  function handleSignOut() {
    setCurrentUser(null);
    setAuthToken("");
    window.localStorage.removeItem("smartDeveloperUser");
    window.localStorage.removeItem("smartDeveloperToken");
    setCollections([]);
    setActivePage("dashboard");
  }

  function handleOpenAccount() {
    setAuthError("");
    setAuthMessage("");
    setAuthDialogOpen(true);
  }

  async function handleChangePassword(oldPassword: string, newPassword: string) {
    setAuthLoading(true);
    setAuthMessage("");
    setAuthError("");

    try {
      const result = await changePassword({
        token: authToken.trim(),
        oldPassword,
        newPassword,
      });
      setAuthMessage(result.message);
    } catch (err) {
      setAuthError(err instanceof Error ? err.message : "Change password failed");
    } finally {
      setAuthLoading(false);
    }
  }

  const latency = searchResponse?.metadata?.latency_ms;
  const resultCount = searchResponse?.results?.length ?? 0;
  const responseProfile =
    typeof searchResponse?.metadata?.ranking_profile === "string"
      ? searchResponse.metadata.ranking_profile
      : "";

  return (
    <DashboardLayout user={currentUser} onOpenAuth={handleOpenAccount} onSignOut={handleSignOut} activePage={activePage} collectionCount={collections.length} onOpenCollection={() => setActivePage("collection")} onOpenDashboard={() => setActivePage("dashboard")}>
      {activePage === "dashboard" ? <div className="workspace">
        <section className="hero">
          <div>
            <p className="eyebrow">NextGenius · Smart Developer</p>
            <h1>AI Site Recommendation Platform</h1>
            <p className="subtitle">
              Search development sites with policy-aware ranking, economics-aware
              scoring, ML market value estimates, and agent-facing explanations.
            </p>
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

          <section className="results-panel" id="results">
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
                  onSave={handleSave}
                  isSaved={collections.some((item) => item.rid === collectionRid(site))}
                  saving={savingRid === collectionRid(site)}
                  onAISummary={handleAISummary}
                  aiSummaryState={aiSummaries[siteStateKey(site, index)]}
                />
              ))}
            </div>
          </section>
        </section>

      </div> : <CollectionPage items={collections} loading={collectionsLoading} onBack={() => setActivePage("dashboard")} onRemove={handleRemoveCollection} onAISummary={(item, index) => handleAISummary(item.site, index, `collection-${item.id}`)} getAISummaryState={(item) => aiSummaries[`collection-${item.id}`]} />}

      <AuthDialog
        key={authDialogOpen ? currentUser?.username ?? "guest-open" : "closed"}
        open={authDialogOpen}
        user={currentUser}
        loading={authLoading}
        error={authError}
        message={authMessage}
        onClose={() => { setAuthDialogOpen(false); setAuthError(""); setAuthMessage(""); }}
        onLogin={handleLogin}
        onRegister={handleDialogRegister}
        onChangePassword={handleChangePassword}
      />

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
    </DashboardLayout>
  );
}

export default App;
