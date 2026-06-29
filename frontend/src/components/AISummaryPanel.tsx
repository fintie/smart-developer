import type { AIPropertySummaryResponse } from "../api";

export function AISummaryPanel({ data }: { data: AIPropertySummaryResponse }) {
  return (
    <section className="ai-property-card">
      <div className="ai-property-header">
        <div>
          <span className="section-kicker">AI property summary</span>
          <h4>{data.summary.headline}</h4>
        </div>
        <span className="ai-source">
          {data.source === "google_gemini"
            ? data.model
            : "local structured summary"}
        </span>
      </div>

      <div className="ai-value-panel">
        <span>Estimated value</span>
        <strong>{data.summary.value_estimate.label}</strong>
        {data.summary.value_estimate.range_label && (
          <p>{data.summary.value_estimate.range_label}</p>
        )}
        <small>
          Confidence: {data.summary.value_estimate.confidence ?? "N/A"}
        </small>
      </div>

      <div className="ai-summary-grid">
        <div>
          <h5>Basic information</h5>
          <ul>
            {data.summary.basic_info.map((item, itemIndex) => (
              <li key={itemIndex}>{item}</li>
            ))}
          </ul>
        </div>
        <div>
          <h5>User requirement fit</h5>
          <p>{data.summary.requirement_match}</p>
          <p>{data.summary.value_estimate.explanation}</p>
        </div>
        <div>
          <h5>Opportunity</h5>
          <ul>
            {data.summary.opportunity_notes.map((item, itemIndex) => (
              <li key={itemIndex}>{item}</li>
            ))}
          </ul>
        </div>
        <div>
          <h5>Risks</h5>
          <ul>
            {data.summary.risk_notes.map((item, itemIndex) => (
              <li key={itemIndex}>{item}</li>
            ))}
          </ul>
        </div>
      </div>

      <p className="ai-disclaimer">{data.summary.disclaimer}</p>

      <section className="ai-suggestion-card">
        <div className="ai-suggestion-header">
          <div>
            <span className="section-kicker">AI suggestion</span>
            <h5>{data.ai_suggestion.headline}</h5>
          </div>
        </div>

        <p>{data.ai_suggestion.suggestion}</p>

        {data.ai_suggestion.next_steps.length > 0 && (
          <div className="ai-suggestion-section">
            <strong>Suggested next steps</strong>
            <ul>
              {data.ai_suggestion.next_steps.map((step, stepIndex) => (
                <li key={stepIndex}>{step}</li>
              ))}
            </ul>
          </div>
        )}

        {data.external_sources.length > 0 && (
          <details className="ai-source-list">
            <summary>
              External sources ({data.external_sources.length})
            </summary>
            <div>
              {data.external_sources.map((source, sourceIndex) => (
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
  );
}

export type AISummaryState = {
  loading: boolean;
  data?: AIPropertySummaryResponse;
  error?: string;
};
