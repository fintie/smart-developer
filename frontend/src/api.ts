export const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8002";

export type SearchPayload = {
  strategy: string;
  query_text: string;
  top_k: number;
  recall_k: number;
  locality?: string | null;
  address_contains?: string | null;
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
  retrieval_similarity?: number;
  fusion_score?: number;
  dcn_prob?: number;
  dcn_rank_score?: number;
  fast_explanation?: string;
  explanation?: string;
};

export type SearchResponse = {
  request_id: string;
  results: SiteResult[];
  metadata?: Record<string, unknown>;
  logging?: Record<string, unknown>;
  service?: Record<string, unknown>;
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