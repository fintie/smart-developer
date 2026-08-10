import { useEffect } from "react";
import { buildPropertyImageUrl, type CollectionItem } from "../api";
import { formatDistance, formatMoney, formatProfileLabel, formatScore } from "../lib/format";
import { AISummaryPanel, type AISummaryState } from "./AISummaryPanel";

type Props = { item: CollectionItem | null; onClose: () => void; onAISummary: () => void; aiSummaryState?: AISummaryState };

function value(value?: string | null) {
  return value || "N/A";
}

export function CollectionDetailDialog({ item, onClose, onAISummary, aiSummaryState }: Props) {
  useEffect(() => {
    if (!item) return;
    const closeOnEscape = (event: KeyboardEvent) => event.key === "Escape" && onClose();
    window.addEventListener("keydown", closeOnEscape);
    document.body.classList.add("dialog-open");
    return () => {
      window.removeEventListener("keydown", closeOnEscape);
      document.body.classList.remove("dialog-open");
    };
  }, [item, onClose]);

  if (!item) return null;
  const site = item.site;
  const image = buildPropertyImageUrl(site);
  const explanation = site.agent_pitch || site.cost_value_explanation || site.policy_explanation || site.fast_explanation || site.explanation;

  return <div className="collection-detail-backdrop" onMouseDown={(event) => event.target === event.currentTarget && onClose()}>
    <section className="collection-detail" role="dialog" aria-modal="true" aria-labelledby="collection-detail-title">
      <header className="collection-detail-topbar"><div><p>Saved search snapshot</p><strong>Property detail</strong></div><button type="button" onClick={onClose} aria-label="Close property detail">×</button></header>
      <div className="collection-detail-scroll">
        {image && <img className="collection-detail-map" src={image} alt={`${item.address} map preview`} />}
        <div className="collection-detail-hero"><div><p>RID {item.rid}</p><h1 id="collection-detail-title">{item.address}</h1><span>Saved {new Date(item.created_at).toLocaleString()}</span></div><div className="detail-score"><span>Opportunity score</span><strong>{formatScore(site.agent_opportunity_score ?? site.strategy_score)}</strong></div></div>

        <section className="detail-section"><div className="detail-section-heading"><p>Site overview</p><h2>Planning and location</h2></div><div className="detail-grid">
          <div><span>Zoning</span><strong>{value(site.primary_zoning_code)}{site.primary_zoning_class ? ` · ${site.primary_zoning_class}` : ""}</strong></div>
          <div><span>Land area</span><strong>{typeof site.lot_size_proxy_sqm === "number" ? `${site.lot_size_proxy_sqm.toLocaleString()} m²` : value(site.lot_size_band)}</strong></div>
          <div><span>Station distance</span><strong>{formatDistance(site.distance_to_station_m)}</strong></div>
          <div><span>Ranking profile</span><strong>{formatProfileLabel(site.ranking_profile)}</strong></div>
          <div><span>Within 800 m</span><strong>{site.within_800m_catchment ? "Yes" : "No"}</strong></div>
          <div><span>Top strategy</span><strong>{value(site.top_strategy)}</strong></div>
        </div></section>

        <section className="detail-section"><div className="detail-section-heading"><p>Financial snapshot</p><h2>Value and estimated costs</h2></div><div className="detail-grid financial">
          <div><span>Estimated market value</span><strong>{formatMoney(site.ml_estimated_market_value)}</strong></div>
          <div><span>Acquisition cost</span><strong>{formatMoney(site.estimated_acquisition_cost)}</strong></div>
          <div><span>Development cost</span><strong>{formatMoney(site.estimated_development_cost)}</strong></div>
          <div><span>Total project cost</span><strong>{formatMoney(site.estimated_total_project_cost)}</strong></div>
          <div><span>Local median sale price</span><strong>{formatMoney(site.locality_median_sale_price)}</strong></div>
          <div><span>Value confidence</span><strong>{value(site.ml_value_confidence)}</strong></div>
        </div></section>

        <section className="detail-section"><div className="detail-section-heading"><p>Risk screening</p><h2>Constraints and signals</h2></div><div className="detail-risk-row"><span className={site.heritage_flag ? "warning" : "clear"}>Heritage · {site.heritage_flag ? "Flagged" : "Clear"}</span><span className={site.flood_flag ? "warning" : "clear"}>Flood · {site.flood_flag ? "Flagged" : "Clear"}</span><span className={site.bushfire_flag ? "warning" : "clear"}>Bushfire · {site.bushfire_flag ? "Flagged" : "Clear"}</span><span>Policy upside · {formatScore(site.policy_upside_score)}</span></div></section>

        {explanation && <section className="detail-section recommendation"><div className="detail-section-heading"><p>Original recommendation</p><h2>Why this site ranked</h2></div><p>{explanation}</p></section>}

        <section className="detail-section detail-ai-section"><div className="detail-ai-action"><div><p>AI property analysis</p><h2>Generate a fresh summary from this saved snapshot</h2><span>The AI uses the full property data that was stored when you added this site to your collection.</span></div><button type="button" onClick={onAISummary} disabled={aiSummaryState?.loading}>{aiSummaryState?.loading ? "Generating…" : aiSummaryState?.data ? "Regenerate AI Summary" : "Generate AI Summary"}</button></div>
          {aiSummaryState?.error && <div className="ai-summary-error">{aiSummaryState.error}</div>}
          {aiSummaryState?.data && <AISummaryPanel data={aiSummaryState.data} />}
        </section>
      </div>
    </section>
  </div>;
}
