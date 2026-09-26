import { useMemo, useState } from "react";

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

const money = (value: number) => new Intl.NumberFormat("en-AU", { style: "currency", currency: "AUD", maximumFractionDigits: 0 }).format(value);

function nswStampDuty(value: number) {
  if (value <= 17000) return value * 0.0125;
  if (value <= 36000) return 212 + (value - 17000) * 0.015;
  if (value <= 97000) return 497 + (value - 36000) * 0.0175;
  if (value <= 364000) return 1564 + (value - 97000) * 0.035;
  if (value <= 1212000) return 10909 + (value - 364000) * 0.045;
  return 49069 + (value - 1212000) * 0.055;
}

export function MarketIntelligence() {
  const [selected, setSelected] = useState<Listing>(listings[0]);
  const [depositPct, setDepositPct] = useState(20);
  const [rate, setRate] = useState(6.15);
  const [term, setTerm] = useState(30);
  const [weeklyRent, setWeeklyRent] = useState(selected.rent);
  const [monthlyCosts, setMonthlyCosts] = useState(850);

  function choose(listing: Listing) {
    setSelected(listing);
    setWeeklyRent(listing.rent);
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

      <div className="property-analysis">
        <section className="property-profile">
          <div className="profile-image"><img src={selected.image} alt={`Illustrative exterior for ${selected.address}`} /><span>Illustrative image</span></div>
          <div className="profile-title"><div><p className="eyebrow">Selected property</p><h2>{selected.address}</h2><p>{selected.beds} bedrooms · {selected.baths} bathrooms · {selected.cars} car · {selected.land ? `${selected.land} m² land` : selected.type}</p></div><button type="button">Save to collection</button></div>
          <p className="property-description">{selected.description}</p>
          <div className="valuation-strip"><div><span>Current price guide</span><strong>{selected.priceGuide}</strong><small>{selected.listed}</small></div><div><span>Smart Developer estimate</span><strong>{money(selected.valuation)}</strong><small>{money(selected.valuationLow)}–{money(selected.valuationHigh)} · modelled, not PropTrack</small></div><div><span>Estimated rent</span><strong>{money(selected.rent)}/week</strong><small>{((selected.rent * 52 / selected.price) * 100).toFixed(2)}% indicative gross yield</small></div></div>
          <div className="property-context"><article><span>Zoning & planning</span><strong>{selected.zoning}</strong><p>Verify current LEP, overlays and permissible uses with council.</p></article><article><span>Risk screen</span><strong>Preliminary only</strong><p>{selected.risk}</p></article></div>
          <div className="listing-history"><h3>Listing & sales history</h3>{selected.history.map((item) => <div key={item.date + item.event}><time>{item.date}</time><span>{item.event}</span><strong>{item.price}</strong></div>)}</div>
        </section>

        <aside className="finance-calculator">
          <div><p className="eyebrow">Finance scenario</p><h2>Cash required & monthly position</h2><p>Adjust the loan assumptions to see the result immediately.</p></div>
          <label>Purchase price<input type="number" value={selected.price} readOnly /></label>
          <div className="finance-two"><label>Deposit<input type="number" min="5" max="80" value={depositPct} onChange={(e) => setDepositPct(Number(e.target.value))} /><span>%</span></label><label>Interest rate<input type="number" min="0" max="20" step="0.05" value={rate} onChange={(e) => setRate(Number(e.target.value))} /><span>%</span></label></div>
          <div className="finance-two"><label>Loan term<input type="number" min="1" max="40" value={term} onChange={(e) => setTerm(Number(e.target.value))} /><span>years</span></label><label>Weekly rent<input type="number" value={weeklyRent} onChange={(e) => setWeeklyRent(Number(e.target.value))} /><span>AUD</span></label></div>
          <label>Other monthly holding costs<input type="number" value={monthlyCosts} onChange={(e) => setMonthlyCosts(Number(e.target.value))} /></label>
          <div className="upfront-summary"><span>Estimated cash required</span><strong>{money(finance.upfront)}</strong><div><span>Deposit <b>{money(finance.deposit)}</b></span><span>NSW transfer duty <b>{money(finance.duty)}</b></span><span>Legal & inspections <b>{money(4400)}</b></span></div></div>
          <div className={finance.cashFlow >= 0 ? "cashflow-summary positive" : "cashflow-summary negative"}><span>Estimated monthly cash flow</span><strong>{finance.cashFlow >= 0 ? "+" : "−"}{money(Math.abs(finance.cashFlow))}</strong><small>Rent {money(finance.rentMonthly)} − loan {money(finance.repayment)} − other costs {money(monthlyCosts)}</small></div>
          <p className="calculator-note">Indicative principal-and-interest scenario only. Duty estimate excludes concessions and special cases. Confirm with a broker, accountant and solicitor.</p>
          <button className="report-cta" type="button" onClick={() => document.getElementById("search")?.scrollIntoView({ behavior: "smooth" })}>Continue to live analysis & report</button>
        </aside>
      </div>

      <div className="market-tools">
        <article><span>Growth explorer</span><h3>Compare suburb price, rent and population signals</h3><p>Map modelled growth with confidence gates before moving from region to property.</p><button type="button" disabled>Growth map · next phase</button></article>
        <article><span>Weekly market report</span><h3>Review listing gaps and market movements</h3><p>Turn shortlist observations into a shareable report for brokers and advisers.</p><button type="button" disabled>Weekly report · next phase</button></article>
      </div>
    </section>
  );
}
