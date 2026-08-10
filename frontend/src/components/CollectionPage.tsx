import { useState } from "react";
import { buildPropertyImageUrl, type CollectionItem } from "../api";
import { formatMoney, formatScore } from "../lib/format";
import { CollectionDetailDialog } from "./CollectionDetailDialog";
import type { AISummaryState } from "./AISummaryPanel";

type CollectionPageProps = {
  items: CollectionItem[];
  loading: boolean;
  onBack: () => void;
  onRemove: (item: CollectionItem) => void;
  onAISummary: (item: CollectionItem, index: number) => void;
  getAISummaryState: (item: CollectionItem) => AISummaryState | undefined;
};

const zoningNames: Record<string, string> = {
  R1: "General Residential",
  R2: "Low Density Residential",
  R3: "Medium Density Residential",
  R4: "High Density Residential",
  R5: "Large Lot Residential",
};

const lotBandNames: Record<string, string> = {
  s: "Small lot",
  m: "Medium lot",
  l: "Large lot",
  small: "Small lot",
  medium: "Medium lot",
  large: "Large lot",
};

function zoningLabel(item: CollectionItem) {
  const code = item.site.primary_zoning_code;
  if (!code) return "Zoning: N/A";
  const name = item.site.primary_zoning_class || zoningNames[code.toUpperCase()];
  return name ? `Zoning: ${code} · ${name}` : `Zoning: ${code}`;
}

function lotSizeLabel(item: CollectionItem) {
  const area = item.site.lot_size_proxy_sqm;
  if (typeof area === "number" && Number.isFinite(area) && area > 0) {
    return `Land area: ${area.toLocaleString(undefined, { maximumFractionDigits: 0 })} m²`;
  }
  const band = item.site.lot_size_band?.trim();
  if (!band) return "Land area: N/A";
  return `Lot size: ${lotBandNames[band.toLowerCase()] || band}`;
}

export function CollectionPage({ items, loading, onBack, onRemove, onAISummary, getAISummaryState }: CollectionPageProps) {
  const [selected, setSelected] = useState<{ item: CollectionItem; index: number } | null>(null);
  return (
    <section className="collection-page">
      <header className="collection-heading">
        <div><p className="eyebrow">Saved opportunities</p><h1>Collection</h1><p>Keep your strongest development opportunities together for later review.</p></div>
        <button type="button" onClick={onBack}>← Back to search</button>
      </header>

      {loading && <div className="collection-empty">Loading your collection…</div>}
      {!loading && items.length === 0 && <div className="collection-empty"><span>◇</span><h2>No saved sites yet</h2><p>Return to recommendations and select Save on a site to build your collection.</p><button type="button" onClick={onBack}>Explore recommendations</button></div>}

      <div className="collection-grid">
        {items.map((item, index) => {
          const image = buildPropertyImageUrl(item.site);
          return <article className="collection-card clickable" key={item.id} role="button" tabIndex={0} aria-label={`Open details for ${item.address}`} onClick={() => setSelected({ item, index })} onKeyDown={(event) => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); setSelected({ item, index }); } }}>
            <div className="collection-image-wrap">{image ? <img src={image} alt={`${item.address} preview`} /> : <span>SD</span>}<span className="saved-pill">Saved</span></div>
            <div className="collection-card-body">
              <p className="collection-rid">RID {item.rid}</p><h2>{item.address}</h2>
              <div className="collection-stats"><div><span>Opportunity</span><strong>{formatScore(item.site.agent_opportunity_score ?? item.site.strategy_score)}</strong></div><div><span>Estimated total cost</span><strong>{formatMoney(item.site.estimated_total_project_cost)}</strong></div></div>
              <div className="collection-tags"><span>{zoningLabel(item)}</span><span>{lotSizeLabel(item)}</span></div>
              <footer><small>Saved {new Date(item.created_at).toLocaleDateString()}</small><span className="view-detail">View details →</span><button type="button" onClick={(event) => { event.stopPropagation(); onRemove(item); }}>Remove</button></footer>
            </div>
          </article>;
        })}
      </div>
      <CollectionDetailDialog item={selected?.item ?? null} onClose={() => setSelected(null)} onAISummary={() => selected && onAISummary(selected.item, selected.index)} aiSummaryState={selected ? getAISummaryState(selected.item) : undefined} />
    </section>
  );
}
