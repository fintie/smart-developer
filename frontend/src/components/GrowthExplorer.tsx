const growthAreas = [
  { suburb: "Parramatta", region: "Western Sydney", growth: 8.7, rent: 6.2, population: 3.1, confidence: "High", x: 48, y: 49 },
  { suburb: "Liverpool", region: "South West Sydney", growth: 8.1, rent: 7.4, population: 3.4, confidence: "High", x: 35, y: 68 },
  { suburb: "Ryde", region: "Northern Sydney", growth: 7.8, rent: 5.6, population: 1.6, confidence: "High", x: 60, y: 39 },
  { suburb: "Penrith", region: "Outer West Sydney", growth: 7.4, rent: 6.9, population: 2.8, confidence: "Medium", x: 18, y: 45 },
  { suburb: "Epping", region: "Northern Sydney", growth: 6.2, rent: 5.1, population: 2.1, confidence: "High", x: 55, y: 29 },
  { suburb: "Meadowbank", region: "Northern Sydney", growth: 4.9, rent: 6.5, population: 2.8, confidence: "Medium", x: 64, y: 46 },
];

export function GrowthExplorer() {
  return (
    <section className="growth-page" aria-labelledby="growth-title">
      <header className="section-intro">
        <div><p className="eyebrow">Regional intelligence</p><h1 id="growth-title">Growth Map</h1><p>Compare modelled price, rental and population momentum across Sydney opportunity areas.</p></div>
        <span className="demo-data-badge">Illustrative market signals</span>
      </header>
      <div className="growth-filters" aria-label="Growth map filters">
        <label>Region<select defaultValue="Sydney"><option>Sydney</option></select></label>
        <label>Property type<select defaultValue="All dwellings"><option>All dwellings</option><option>Houses</option><option>Units</option></select></label>
        <label>Primary signal<select defaultValue="12-month growth"><option>12-month growth</option><option>Rental growth</option><option>Population growth</option></select></label>
        <label>Confidence<select defaultValue="Medium & high"><option>Medium & high</option><option>High only</option></select></label>
      </div>
      <div className="growth-layout">
        <div className="growth-map-card">
          <div className="growth-map-grid" aria-label="Illustrative Sydney growth map">
            <span className="map-watermark">SYDNEY</span>
            {growthAreas.map((area) => <button key={area.suburb} type="button" className={area.confidence === "High" ? "growth-bubble high" : "growth-bubble"} style={{ left: `${area.x}%`, top: `${area.y}%`, width: `${42 + area.growth * 3}px`, height: `${42 + area.growth * 3}px` }} aria-label={`${area.suburb}, ${area.growth}% annual growth`}><strong>{area.growth}%</strong><span>{area.suburb}</span></button>)}
          </div>
          <div className="growth-legend"><span><i className="legend-high" />High confidence</span><span><i />Medium confidence</span><small>Bubble size represents modelled annual price growth</small></div>
        </div>
        <aside className="growth-ranking">
          <div><p className="eyebrow">Ranked areas</p><h2>Strongest signals</h2></div>
          {growthAreas.map((area, index) => <article key={area.suburb}><span className="growth-rank">{index + 1}</span><div><strong>{area.suburb}</strong><small>{area.region}</small></div><div><strong>+{area.growth}%</strong><small>price</small></div><div><strong>+{area.rent}%</strong><small>rent</small></div></article>)}
        </aside>
      </div>
      <p className="source-note growth-note">Illustrative modelled figures for product demonstration. Confirm current market data before making an investment decision.</p>
    </section>
  );
}
