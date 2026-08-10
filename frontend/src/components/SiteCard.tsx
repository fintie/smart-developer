import { buildPropertyImageUrl, type SiteResult } from "../api";
import {
  formatDistance,
  formatNumber,
  formatMoney,
  formatProfileLabel,
  formatScore,
  splitAssessmentText,
} from "../lib/format";
import { AISummaryPanel, type AISummaryState } from "./AISummaryPanel";

type Props = {
  site: SiteResult;
  index: number;
  requestId: string;
  onFeedback: (eventType: string, site: SiteResult, index: number) => void;
  onSave: (site: SiteResult, index: number) => void;
  isSaved?: boolean;
  saving?: boolean;
  onAISummary: (site: SiteResult, index: number) => void;
  aiSummaryState?: AISummaryState;
};

export function SiteCard({
  site,
  index,
  requestId,
  onFeedback,
  onSave,
  isSaved,
  saving,
  onAISummary,
  aiSummaryState,
}: Props) {
  const address = site.base_site_address || site.address || "Unknown address";
  const propertyImageUrl = buildPropertyImageUrl(site);

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
        <span>Profile: {formatProfileLabel(site.ranking_profile)}</span>
        <span>
          Map:{" "}
          {typeof site.latitude === "number" && typeof site.longitude === "number"
            ? `${site.latitude.toFixed(4)}, ${site.longitude.toFixed(4)}`
            : "N/A"}
        </span>
      </div>

      {propertyImageUrl && (
        <img
          className="property-image"
          src={propertyImageUrl}
          alt={`${address} map preview`}
          loading="lazy"
          onError={(event) => {
            event.currentTarget.style.display = "none";
          }}
        />
      )}

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
          <summary>Policy evidence ({site.policy_evidence.length})</summary>

          <div className="policy-evidence-list">
            {site.policy_evidence.map((evidence, evidenceIndex) => (
              <div key={evidenceIndex} className="policy-evidence-item">
                <strong>
                  {evidence.policy_name || evidence.policy_id || "Policy source"}
                </strong>

                {evidence.snippet && <p>{evidence.snippet}</p>}

                {evidence.source_url && (
                  <a href={evidence.source_url} target="_blank" rel="noreferrer">
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
        <button className={isSaved ? "saved-button" : ""} onClick={() => onSave(site, index)} disabled={saving || isSaved}>{saving ? "Saving…" : isSaved ? "✓ Saved" : "Save"}</button>
        <button className="secondary" onClick={() => onFeedback("dismiss", site, index)}>
          Dismiss
        </button>
      </div>

      {aiSummaryState?.error && (
        <div className="ai-summary-error">{aiSummaryState.error}</div>
      )}

      {aiSummaryState?.data && <AISummaryPanel data={aiSummaryState.data} />}

      <div className="rid">RID: {site.RID ?? "N/A"} · Request: {requestId}</div>
    </article>
  );
}
