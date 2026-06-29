import { RANKING_PROFILES } from "./strategies";

export function formatNumber(value: unknown, digits = 1) {
  if (typeof value !== "number" || Number.isNaN(value)) return "N/A";
  return value.toFixed(digits);
}

export function formatMoney(value?: number | null) {
  if (typeof value !== "number" || Number.isNaN(value)) return "N/A";

  return new Intl.NumberFormat("en-AU", {
    style: "currency",
    currency: "AUD",
    maximumFractionDigits: 0,
  }).format(value);
}

export function formatScore(value?: number | null) {
  if (typeof value !== "number" || Number.isNaN(value)) return "N/A";
  return value.toFixed(1);
}

export function formatDistance(value: unknown) {
  if (typeof value !== "number" || Number.isNaN(value)) return "N/A";
  return `${Math.round(value)} m`;
}

export function formatProfileLabel(value?: string | null) {
  if (!value) return "N/A";
  return (
    RANKING_PROFILES.find((profile) => profile.value === value)?.label ??
    value.replaceAll("_", " ")
  );
}

export function splitAssessmentText(value: string) {
  return value
    .split(/(?<=[.!?])\s+/)
    .map((sentence) => sentence.trim())
    .filter(Boolean);
}
