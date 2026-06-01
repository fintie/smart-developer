export const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8002";

export type SearchPayload = {
  strategy: string;
  query_text: string;
  top_k: number;
  recall_k: number;
  locality?: string | null;
  address_contains?: string | null;
  ranking_profile?: "balanced" | "policy_upside" | "budget_sensitive" | "high_value";
  with_explanations: boolean;
  use_template_explanations: boolean;
  log_request: boolean;
  debug: boolean;
  user_id: string;
  session_id: string;
};

export type SiteResult = {
  RID?: string | number;
  address?: string;
  base_site_address?: string;

  latitude?: number;
  longitude?: number;
  geometry_type?: string;
  geocode_source?: string;
  geocode_confidence?: number;

  primary_zoning_code?: string;
  primary_zoning_class?: string;
  zoning_band?: string;
  lot_size_band?: string;
  lot_size_proxy_sqm?: number;
  constraint_severity_band?: string;
  station_distance_band?: string;
  distance_to_station_m?: number;
  within_800m_catchment?: boolean;
  heritage_flag?: boolean;
  flood_flag?: boolean;
  bushfire_flag?: boolean;

  top_strategy?: string;
  top_strategy_score?: number;
  strategy_score?: number;

  agent_opportunity_score?: number;
  agent_rank_position?: number;
  ranking_profile?: string;

  policy_upside_score?: number;
  policy_signal_band?: string;
  policy_matched_rules?: string[];
  policy_matched_policies?: string[];
  policy_matched_policy_names?: string[];
  policy_explanation?: string;
  policy_evidence_count?: number;
  policy_evidence?: Array<{
    policy_id?: string;
    policy_name?: string;
    source_url?: string;
    snippet?: string;
    relevance_score?: number;
  }>;

  locality?: string;
  locality_median_sale_price?: number;
  locality_sales_count?: number;
  locality_price_confidence?: string;

  ml_estimated_market_value?: number;
  ml_value_lower_bound?: number;
  ml_value_upper_bound?: number;
  ml_value_error_pct?: number;
  ml_value_confidence?: string;
  ml_value_model?: string;

  estimated_acquisition_cost?: number;
  estimated_acquisition_cost_source?: string;
  gross_floor_area_proxy_sqm?: number;
  base_construction_cost?: number;
  estimated_development_cost?: number;
  estimated_soft_cost?: number;
  estimated_contingency?: number;
  estimated_total_project_cost?: number;

  cost_band?: string;
  cost_risk_score?: number;
  cost_efficiency_score?: number;
  value_potential_score?: number;
  value_potential_band?: string;
  cost_value_explanation?: string;

  fast_explanation?: string;
  explanation?: string;
  agent_pitch?: string;
};

export type SearchResponse = {
  request_id: string;
  results: SiteResult[];
  metadata?: Record<string, unknown>;
  logging?: Record<string, unknown>;
  service?: Record<string, unknown>;
  feedback_prompt?: {
    enabled: boolean;
    type: string;
    title: string;
    scale: {
      min: number;
      max: number;
      labels: Record<string, string>;
    };
    submit_endpoint: string;
  };
};

export type FeedbackPayload = {
  request_id: string;
  event_type: string;
  rid?: string | number | null;
  rank_position?: number | null;
  event_value?: Record<string, unknown> | null;
  user_note?: string | null;
  user_id?: string;
  session_id?: string;
};

export type RecommendationFeedbackPayload = {
  request_id: string;
  rating: number;
  user_note?: string | null;
  user_id?: string;
  session_id?: string;
};

export type ReportPayload = {
  request_id: string;
  explanation_mode: string;
  output_markdown: boolean;
  output_pdf: boolean;
  audience: string;
  title: string;
};

export async function searchSites(payload: SearchPayload): Promise<SearchResponse> {
  const response = await fetch(`${API_BASE_URL}/api/search`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    throw new Error(await response.text());
  }

  return response.json();
}

export async function sendFeedback(payload: FeedbackPayload) {
  const response = await fetch(`${API_BASE_URL}/api/feedback`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    throw new Error(await response.text());
  }

  return response.json();
}

export async function sendRecommendationFeedback(
  payload: RecommendationFeedbackPayload,
) {
  const response = await fetch(`${API_BASE_URL}/api/recommendation-feedback`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    throw new Error(await response.text());
  }

  return response.json();
}

export async function createReport(payload: ReportPayload) {
  const response = await fetch(`${API_BASE_URL}/api/reports`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    throw new Error(await response.text());
  }

  return response.json();
}

export async function exportReportPdf(payload: {
  strategy: string;
  query_text: string;
  results: SiteResult[];
  title?: string;
  audience?: string;
  output_format?: "pdf" | "markdown";
  max_rows?: number;
}) {
  const response = await fetch(`${API_BASE_URL}/api/export-report`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      title: "Smart Developer Site Recommendation Report",
      audience: "developer / real estate agent",
      output_format: "pdf",
      max_rows: 5,
      ...payload,
    }),
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `Failed to export report: ${response.status}`);
  }

  const blob = await response.blob();
  const url = window.URL.createObjectURL(blob);

  const link = document.createElement("a");
  link.href = url;
  link.download = "smart_developer_report.pdf";
  document.body.appendChild(link);
  link.click();
  link.remove();

  window.URL.revokeObjectURL(url);
}
