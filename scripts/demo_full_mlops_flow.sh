#!/usr/bin/env bash
set -euo pipefail

# =========================
# Smart Developer Full MLOps Demo
# =========================

API_BASE_URL="${API_BASE_URL:-http://localhost:8001}"

STRATEGY="single_dwelling_rebuild"
QUERY_TEXT="I want a site for detached house redevelopment on standard residential land, with low planning constraints and a suitable lot size."

TOP_K=3
RECALL_K=1000

USER_ID="demo_user"
SESSION_ID="full_mlops_demo_house"

EVENT_TYPE="save"
USER_NOTE="Full MLOps demo: user saved the top house redevelopment site."

REPORT_TITLE="Smart Developer House Redevelopment Report"
REPORT_AUDIENCE="developer"

echo "=========================================="
echo "Smart Developer Full MLOps Demo"
echo "API: $API_BASE_URL"
echo "=========================================="
echo

echo "1. Health check"
curl -s "$API_BASE_URL/health" | python -m json.tool
echo

echo "2. Retrieval request"
RETRIEVAL_RESPONSE=$(
  curl -s -X POST "$API_BASE_URL/retrieve-sites" \
    -H "Content-Type: application/json" \
    -d "{
      \"strategy\": \"$STRATEGY\",
      \"query_text\": \"$QUERY_TEXT\",
      \"top_k\": $TOP_K,
      \"recall_k\": $RECALL_K,
      \"with_explanations\": false,
      \"use_template_explanations\": true,
      \"user_id\": \"$USER_ID\",
      \"session_id\": \"$SESSION_ID\",
      \"log_request\": true
    }"
)

echo "$RETRIEVAL_RESPONSE" | python -m json.tool
echo

REQUEST_ID=$(echo "$RETRIEVAL_RESPONSE" | python -c 'import sys,json; print(json.load(sys.stdin)["request_id"])')
TOP_RID=$(echo "$RETRIEVAL_RESPONSE" | python -c 'import sys,json; print(json.load(sys.stdin)["results"][0]["RID"])')
TOP_SITE=$(echo "$RETRIEVAL_RESPONSE" | python -c 'import sys,json; print(json.load(sys.stdin)["results"][0]["base_site_address"])')
LATENCY_MS=$(echo "$RETRIEVAL_RESPONSE" | python -c 'import sys,json; print(json.load(sys.stdin)["metadata"]["latency_ms"])')

echo "Retrieved request_id: $REQUEST_ID"
echo "Top RID:              $TOP_RID"
echo "Top site:             $TOP_SITE"
echo "Retrieval latency:    ${LATENCY_MS} ms"
echo

echo "3. Feedback event"
FEEDBACK_RESPONSE=$(
  curl -s -X POST "$API_BASE_URL/feedback" \
    -H "Content-Type: application/json" \
    -d "{
      \"request_id\": \"$REQUEST_ID\",
      \"rid\": \"$TOP_RID\",
      \"rank_position\": 1,
      \"event_type\": \"$EVENT_TYPE\",
      \"event_value\": {
        \"source\": \"demo_script\",
        \"demo\": true
      },
      \"user_note\": \"$USER_NOTE\",
      \"user_id\": \"$USER_ID\",
      \"session_id\": \"$SESSION_ID\"
    }"
)

echo "$FEEDBACK_RESPONSE" | python -m json.tool
echo

echo "4. Report job"
REPORT_RESPONSE=$(
  curl -s -X POST "$API_BASE_URL/report-jobs" \
    -H "Content-Type: application/json" \
    -d "{
      \"request_id\": \"$REQUEST_ID\",
      \"explanation_mode\": \"template\",
      \"output_markdown\": true,
      \"output_pdf\": true,
      \"audience\": \"$REPORT_AUDIENCE\",
      \"title\": \"$REPORT_TITLE\"
    }"
)

echo "$REPORT_RESPONSE" | python -m json.tool
echo

REPORT_ID=$(echo "$REPORT_RESPONSE" | python -c 'import sys,json; print(json.load(sys.stdin)["report_id"])')
PDF_PATH=$(echo "$REPORT_RESPONSE" | python -c 'import sys,json; print(json.load(sys.stdin).get("pdf_path"))')

echo "Generated report_id: $REPORT_ID"
echo "PDF path:            $PDF_PATH"
echo

echo "5. Report status"
curl -s "$API_BASE_URL/report-jobs/$REPORT_ID" | python -m json.tool
echo

echo "=========================================="
echo "Demo complete."
echo "Useful SQL checks:"
echo
echo "SELECT request_id, strategy, latency_ms, result_count"
echo "FROM retrieval_requests"
echo "ORDER BY created_at DESC"
echo "LIMIT 5;"
echo
echo "SELECT request_id, rid, rank_position, base_site_address, strategy_score"
echo "FROM retrieval_results"
echo "WHERE request_id = '$REQUEST_ID'"
echo "ORDER BY rank_position;"
echo
echo "SELECT feedback_id, request_id, rid, event_type, user_note"
echo "FROM user_feedback"
echo "ORDER BY created_at DESC"
echo "LIMIT 5;"
echo
echo "SELECT report_id, request_id, status, output_pdf_path, latency_ms"
echo "FROM report_jobs"
echo "ORDER BY created_at DESC"
echo "LIMIT 5;"
echo "=========================================="

if [ "$PDF_PATH" != "None" ] && [ -n "$PDF_PATH" ]; then
  echo
  echo "To open PDF:"
  echo "open $PDF_PATH"
fi