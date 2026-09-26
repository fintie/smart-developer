import { useEffect, useMemo, useState } from "react";

type Listing = {
  id: string;
  address: string;
  suburb: string;
  image: string;
  price: number;
  priceGuide: string;
  valuation: number;
  valuationLow: number;
  valuationHigh: number;
  rent: number;
  beds: number;
  baths: number;
  cars: number;
  land: number;
  type: string;
  listed: string;
  description: string;
  zoning: string;
  risk: string;
  history: Array<{ date: string; event: string; price: string }>;
};

const listings: Listing[] = [
  { id: "ryde", address: "18 Banksia Street, Ryde NSW 2112", suburb: "Ryde", image: "/listing-ryde.png", price: 2180000, priceGuide: "$2.10m–$2.30m", valuation: 2240000, valuationLow: 2110000, valuationHigh: 2390000, rent: 1180, beds: 4, baths: 2, cars: 2, land: 612, type: "House", listed: "Listed 6 days ago", description: "A modern family home on a level parcel, close to transport, schools and village amenities. The regular lot shape and established residential context support a clear long-term hold profile.", zoning: "R2 Low Density Residential", risk: "No major flood, bushfire or heritage flag in the current screening dataset.", history: [{ date: "20 Sep 2026", event: "Listed for sale", price: "Guide $2.10m–$2.30m" }, { date: "14 Mar 2018", event: "Sold", price: "$1,420,000" }] },
  { id: "epping", address: "42 Orchard Road, Epping NSW 2121", suburb: "Epping", image: "/listing-epping.png", price: 1960000, priceGuide: "Auction guide $1.90m", valuation: 2015000, valuationLow: 1880000, valuationHigh: 2150000, rent: 1050, beds: 4, baths: 3, cars: 1, land: 438, type: "Townhouse", listed: "Auction in 9 days", description: "Contemporary accommodation with strong rail access and practical family amenity. Rental demand is supported by nearby education, retail and employment connections.", zoning: "R3 Medium Density Residential", risk: "Low constraint screen; verify strata, easements and site-specific controls.", history: [{ date: "18 Sep 2026", event: "Listed for auction", price: "Guide $1.90m" }, { date: "11 Jul 2020", event: "Sold", price: "$1,510,000" }] },
  { id: "meadowbank", address: "7/61 Park Avenue, Meadowbank NSW 2114", suburb: "Meadowbank", image: "/listing-meadowbank.png", price: 925000, priceGuide: "$900,000–$950,000", valuation: 938000, valuationLow: 880000, valuationHigh: 995000, rent: 760, beds: 2, baths: 2, cars: 1, land: 0, type: "Apartment", listed: "Listed yesterday", description: "Well-connected apartment within walking distance of rail, ferry, parks and local shops. A lower-maintenance option with a comparatively strong indicative rental yield.", zoning: "R4 High Density Residential", risk: "Review strata records, building condition and flood context before purchase.", history: [{ date: "25 Sep 2026", event: "Listed for sale", price: "$900,000–$950,000" }, { date: "02 Feb 2021", event: "Sold", price: "$785,000" }] },
];

const extraInfo = {
  ryde: { council: "City of Ryde", built: "2016 (indicative)", floor: "238 m²", frontage: "15.2 m", rates: "$520/qtr", water: "$285/qtr", schools: ["Ryde Public School · 0.8 km", "Ryde Secondary College · 1.6 km"], transport: ["Bus to CBD · 280 m", "West Ryde Station · 2.1 km"], amenities: ["Top Ryde City · 0.9 km", "Ryde Park · 0.6 km"], median: "$2.32m", growth: "+7.8%", population: "+1.6% p.a." },
  epping: { council: "City of Parramatta", built: "2021 (indicative)", floor: "206 m²", frontage: "9.6 m", rates: "$410/qtr", water: "$250/qtr", schools: ["Epping Public School · 0.7 km", "Cheltenham Girls High · 2.0 km"], transport: ["Epping Station · 0.9 km", "Metro & rail interchange · 0.9 km"], amenities: ["Epping town centre · 0.8 km", "Boronia Park · 0.5 km"], median: "$2.10m", growth: "+6.2%", population: "+2.1% p.a." },
  meadowbank: { council: "City of Ryde", built: "2018 (indicative)", floor: "98 m²", frontage: "Strata property", rates: "$335/qtr", water: "$205/qtr", schools: ["Meadowbank Public School · 0.6 km", "Marsden High School · 1.7 km"], transport: ["Meadowbank Station · 0.5 km", "Meadowbank Ferry · 0.8 km"], amenities: ["Village Plaza · 0.7 km", "Parramatta River walk · 0.3 km"], median: "$815k", growth: "+4.9%", population: "+2.8% p.a." },
} as const;

const money = (value: number) => new Intl.NumberFormat("en-AU", { style: "currency", currency: "AUD", maximumFractionDigits: 0 }).format(value);

function nswStampDuty(value: number) {
  if (value <= 17000) return value * 0.0125;
  if (value <= 36000) return 212 + (value - 17000) * 0.015;
  if (value <= 97000) return 497 + (value - 36000) * 0.0175;
  if (value <= 364000) return 1564 + (value - 97000) * 0.035;
  if (value <= 1212000) return 10909 + (value - 364000) * 0.045;
  return 49069 + (value - 1212000) * 0.055;
}

export function MarketIntelligence({ onOpenOpportunity }: { onOpenOpportunity: () => void }) {
  const [selected, setSelected] = useState<Listing>(listings[0]);
  const [detailOpen, setDetailOpen] = useState(false);
  const [depositPct, setDepositPct] = useState(20);
  const [rate, setRate] = useState(6.15);
  const [term, setTerm] = useState(30);
  const [weeklyRent, setWeeklyRent] = useState(selected.rent);
  const [monthlyCosts, setMonthlyCosts] = useState(850);

  useEffect(() => {
    const syncFromHash = () => {
      const match = window.location.hash.match(/^#property\/(.+)$/);
      const listing = match ? listings.find((item) => item.id === match[1]) : undefined;
      if (listing) {
        setSelected(listing);
        setWeeklyRent(listing.rent);
        setDetailOpen(true);
      } else if (!window.location.hash.startsWith("#property/")) {
        setDetailOpen(false);
      }
    };
    syncFromHash();
    window.addEventListener("hashchange", syncFromHash);
    return () => window.removeEventListener("hashchange", syncFromHash);
  }, []);

  function choose(listing: Listing) {
    setSelected(listing);
    setWeeklyRent(listing.rent);
    setDetailOpen(true);
    window.history.pushState(null, "", `#property/${listing.id}`);
    window.setTimeout(() => window.scrollTo({ top: 0, behavior: "smooth" }), 0);
  }

  const finance = useMemo(() => {
    const deposit = selected.price * depositPct / 100;
    const loan = selected.price - deposit;
    const monthlyRate = rate / 100 / 12;
    const payments = term * 12;
    const repayment = monthlyRate === 0 ? loan / payments : loan * monthlyRate * (1 + monthlyRate) ** payments / ((1 + monthlyRate) ** payments - 1);
    const duty = nswStampDuty(selected.price);
    const conveyancing = 3500;
    const inspections = 900;
    const upfront = deposit + duty + conveyancing + inspections;
    const rentMonthly = weeklyRent * 52 / 12;
    return { deposit, loan, repayment, duty, upfront, rentMonthly, cashFlow: rentMonthly - repayment - monthlyCosts };
  }, [depositPct, monthlyCosts, rate, selected, term, weeklyRent]);

  const details = extraInfo[selected.id as keyof typeof extraInfo];

  if (detailOpen) {
    return (
      <section className="property-detail-page" aria-label={`Property details for ${selected.address}`}>
        <header className="detail-page-topbar"><button type="button" onClick={() => { window.history.pushState(null, "", window.location.pathname); setDetailOpen(false); }}>← Back to listings</button><div><button type="button">♡ Save</button><button type="button" onClick={onOpenOpportunity}>Create report</button></div></header>
        <div className="detail-page-hero"><img src={selected.image} alt={`Illustrative exterior for ${selected.address}`} /><span>Illustrative listing image</span><div><p>{selected.listed} · For sale</p><h1>{selected.address}</h1><div><span>{selected.beds} bedrooms</span><span>{selected.baths} bathrooms</span><span>{selected.cars} car</span><span>{selected.land ? `${selected.land} m² land` : selected.type}</span></div></div></div>

        <div className="detail-page-layout">
          <main className="detail-page-content">
            <section className="detail-price-grid"><article><span>Current price guide</span><strong>{selected.priceGuide}</strong><small>{selected.listed}</small></article><article><span>Smart Developer estimate</span><strong>{money(selected.valuation)}</strong><small>{money(selected.valuationLow)}–{money(selected.valuationHigh)} · modelled estimate</small></article><article><span>Estimated rent</span><strong>{money(selected.rent)}/week</strong><small>{((selected.rent * 52 / selected.price) * 100).toFixed(2)}% indicative gross yield</small></article></section>
            <section className="detail-card"><p className="eyebrow">Property overview</p><h2>About this property</h2><p className="detail-copy">{selected.description}</p><div className="property-attributes"><div><span>Property type</span><strong>{selected.type}</strong></div><div><span>Land area</span><strong>{selected.land ? `${selected.land} m²` : "Strata"}</strong></div><div><span>Internal area</span><strong>{details.floor}</strong></div><div><span>Year built</span><strong>{details.built}</strong></div><div><span>Frontage</span><strong>{details.frontage}</strong></div><div><span>Local council</span><strong>{details.council}</strong></div></div></section>
            <section className="detail-card"><p className="eyebrow">Area intelligence</p><h2>Market and neighbourhood</h2><div className="suburb-metrics"><div><span>Suburb median</span><strong>{details.median}</strong></div><div><span>12-month change</span><strong>{details.growth}</strong></div><div><span>Population trend</span><strong>{details.population}</strong></div></div><div className="nearby-grid"><article><h3>Schools</h3>{details.schools.map(item => <p key={item}>{item}</p>)}</article><article><h3>Transport</h3>{details.transport.map(item => <p key={item}>{item}</p>)}</article><article><h3>Lifestyle</h3>{details.amenities.map(item => <p key={item}>{item}</p>)}</article></div><small className="source-note">Illustrative proximity and market figures pending connection to licensed/current data sources.</small></section>
            <section className="detail-card"><p className="eyebrow">Planning & due diligence</p><h2>Land, planning and risk</h2><div className="planning-grid"><article><span>Zoning</span><strong>{selected.zoning}</strong><p>Verify permissible uses, minimum lot size, height and floor-space controls against the current LEP.</p></article><article><span>Preliminary risk screen</span><strong>Professional verification required</strong><p>{selected.risk}</p></article></div></section>
            <section className="detail-card listing-history"><p className="eyebrow">Property timeline</p><h2>Listing & sales history</h2>{selected.history.map((item) => <div key={item.date + item.event}><time>{item.date}</time><span>{item.event}</span><strong>{item.price}</strong></div>)}</section>
            <section className="detail-card"><p className="eyebrow">Indicative ownership costs</p><h2>Regular property costs</h2><div className="property-attributes"><div><span>Council rates</span><strong>{details.rates}</strong></div><div><span>Water service</span><strong>{details.water}</strong></div><div><span>Insurance allowance</span><strong>$180/month</strong></div><div><span>Property management</span><strong>6.5% of rent</strong></div></div></section>
          </main>

          <aside className="finance-calculator detail-finance">
            <div><p className="eyebrow">Finance scenario</p><h2>Cash required & monthly position</h2><p>Adjust assumptions and see the result immediately.</p></div>
            <label>Purchase price<input type="number" value={selected.price} readOnly /></label>
            <div className="finance-two"><label>Deposit<input type="number" min="5" max="80" value={depositPct} onChange={(e) => setDepositPct(Number(e.target.value))} /><span>%</span></label><label>Interest rate<input type="number" min="0" max="20" step="0.05" value={rate} onChange={(e) => setRate(Number(e.target.value))} /><span>%</span></label></div>
            <div className="finance-two"><label>Loan term<input type="number" min="1" max="40" value={term} onChange={(e) => setTerm(Number(e.target.value))} /><span>years</span></label><label>Weekly rent<input type="number" value={weeklyRent} onChange={(e) => setWeeklyRent(Number(e.target.value))} /><span>AUD</span></label></div>
            <label>Other monthly holding costs<input type="number" value={monthlyCosts} onChange={(e) => setMonthlyCosts(Number(e.target.value))} /></label>
            <div className="upfront-summary"><span>Estimated cash required</span><strong>{money(finance.upfront)}</strong><div><span>Deposit <b>{money(finance.deposit)}</b></span><span>NSW transfer duty <b>{money(finance.duty)}</b></span><span>Legal & inspections <b>{money(4400)}</b></span></div></div>
            <div className={finance.cashFlow >= 0 ? "cashflow-summary positive" : "cashflow-summary negative"}><span>Estimated monthly cash flow</span><strong>{finance.cashFlow >= 0 ? "+" : "−"}{money(Math.abs(finance.cashFlow))}</strong><small>Rent {money(finance.rentMonthly)} − loan {money(finance.repayment)} − other costs {money(monthlyCosts)}</small></div>
            <p className="calculator-note">Indicative principal-and-interest scenario only. Confirm duty, lending and costs with qualified advisers.</p>
            <button className="report-cta" type="button" onClick={onOpenOpportunity}>Continue to live analysis & report</button>
          </aside>
        </div>
      </section>
    );
  }

  return (
    <section className="market-intelligence" aria-labelledby="market-heading">
      <header className="market-heading">
        <div><p className="eyebrow">Property market workspace</p><h2 id="market-heading">Active listing research</h2><p>Compare asking prices, modelled value, rent and funding in one place.</p></div>
        <span className="demo-data-badge">Illustrative listings · connect licensed feed for live data</span>
      </header>

      <div className="listing-grid">
        {listings.map((listing) => <article className={selected.id === listing.id ? "listing-card selected" : "listing-card"} key={listing.id}>
          <button type="button" onClick={() => choose(listing)} aria-label={`Analyse ${listing.address}`}>
            <div className="listing-image"><img src={listing.image} alt={`Illustrative exterior for ${listing.type} in ${listing.suburb}`} /><span>For sale</span><strong>{listing.priceGuide}</strong></div>
            <div className="listing-card-body"><p>{listing.listed}</p><h3>{listing.address}</h3><div className="listing-facts"><span>{listing.beds} bed</span><span>{listing.baths} bath</span><span>{listing.cars} car</span><span>{listing.type}</span></div><div className="listing-card-value"><span>Modelled value</span><strong>{money(listing.valuation)}</strong></div></div>
          </button>
        </article>)}
      </div>
    </section>
  );
}
