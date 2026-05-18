#!/usr/bin/env bash
set -e

API_BASE_URL="${API_BASE_URL:-http://localhost:8002}"

LOCALITY="${1:-WAITARA}"
STRATEGY="${2:-low_rise_apartment}"

if [ "$STRATEGY" = "low_rise_apartment" ]; then
  QUERY="I want a site for low-rise apartment redevelopment near a train station, with high development zoning, a large site, and limited planning constraints."
else
  QUERY="I want a site for detached house redevelopment on standard residential land, with low planning constraints and a suitable lot size."
fi

echo "Testing locality: $LOCALITY"
echo "Strategy: $STRATEGY"
echo ""

TMP_FILE="$(mktemp)"

curl -s -X POST "$API_BASE_URL/api/search" \
  -H "Content-Type: application/json" \
  -d "{
    \"strategy\": \"$STRATEGY\",
    \"query_text\": \"$QUERY\",
    \"top_k\": 5,
    \"recall_k\": 1000,
    \"locality\": \"$LOCALITY\",
    \"with_explanations\": false,
    \"use_template_explanations\": true,
    \"log_request\": false,
    \"debug\": true
  }" > "$TMP_FILE"

python - "$TMP_FILE" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
raw = path.read_text()

if not raw.strip():
    print("Empty response from API.")
    sys.exit(1)

try:
    data = json.loads(raw)
except json.JSONDecodeError:
    print("Response is not valid JSON:")
    print(raw[:2000])
    sys.exit(1)

print("request_id:", data.get("request_id"))
print("result_count:", len(data.get("results", [])))

print("\nmetadata:")
print(json.dumps(data.get("metadata", {}), indent=2))

print("\nrequest:")
print(json.dumps(data.get("request", {}), indent=2))

print("\nTop results:")
for i, r in enumerate(data.get("results", []), 1):
    print(f"{i}. RID={r.get('RID')} | {r.get('base_site_address') or r.get('address')}")
    print(f"   zoning={r.get('primary_zoning_code')} score={r.get('strategy_score')} dcn={r.get('dcn_prob')} final={r.get('dcn_rank_score') or r.get('fusion_score')}")
PY

rm -f "$TMP_FILE"
