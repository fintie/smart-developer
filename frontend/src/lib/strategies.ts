export type RankingProfile =
  | "balanced"
  | "policy_upside"
  | "budget_sensitive"
  | "high_value";

export type Strategy = {
  value: string;
  label: string;
  query: string;
};

export const STRATEGIES: Strategy[] = [
  {
    value: "single_dwelling_rebuild",
    label: "Single dwelling rebuild",
    query:
      "I want a site for detached house redevelopment on standard residential land, with low planning constraints and a suitable lot size.",
  },
  {
    value: "low_rise_apartment",
    label: "Low-rise apartment",
    query:
      "I want a site for low-rise apartment redevelopment near a train station, with high development zoning, a large site, and limited planning constraints.",
  },
  {
    value: "dual_occupancy",
    label: "Dual occupancy",
    query:
      "I want a residential site suitable for dual occupancy, with appropriate zoning, a suitable lot size, and low planning constraints.",
  },
  {
    value: "granny_flat",
    label: "Granny flat",
    query:
      "I want a residential site suitable for a granny flat or secondary dwelling, with low constraints and a practical lot size.",
  },
];

export const RANKING_PROFILES: Array<{
  value: RankingProfile;
  label: string;
  description: string;
}> = [
  {
    value: "balanced",
    label: "Balanced",
    description: "Balances strategy fit, policy upside, value, and cost.",
  },
  {
    value: "policy_upside",
    label: "Policy Upside",
    description: "Prioritises sites with stronger planning-policy signals.",
  },
  {
    value: "budget_sensitive",
    label: "Budget Sensitive",
    description: "Prioritises lower-cost and more cost-efficient opportunities.",
  },
  {
    value: "high_value",
    label: "High Value",
    description: "Prioritises sites with stronger market and redevelopment value signals.",
  },
];
