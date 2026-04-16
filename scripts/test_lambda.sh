#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# scripts/test_lambda.sh
# Invokes the Lambda function directly from the CLI and shows the response.
# Useful for verifying deployment without waiting for the EventBridge schedule.
#
# Usage:
#   bash scripts/test_lambda.sh
#   bash scripts/test_lambda.sh forecast_only    # default
#   bash scripts/test_lambda.sh full             # full retrain (slow!)
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail

AWS_REGION="${AWS_REGION:-us-east-1}"
LAMBDA_FUNCTION="stock-forecasting-pipeline"
MODE="${1:-forecast_only}"

PAYLOAD="{\"mode\": \"${MODE}\", \"benchmark\": false}"

echo ""
echo "Invoking Lambda: ${LAMBDA_FUNCTION}"
echo "Payload : ${PAYLOAD}"
echo "This may take 1-3 minutes for forecast_only mode..."
echo ""

# Invoke and capture response
aws lambda invoke \
    --function-name "${LAMBDA_FUNCTION}" \
    --region "${AWS_REGION}" \
    --payload "$(echo ${PAYLOAD} | base64)" \
    --cli-binary-format raw-in-base64-out \
    --log-type Tail \
    response.json 2>&1 | grep -v "^LogResult" || true

echo "── Response ─────────────────────────────────────────"
cat response.json | python3 -m json.tool 2>/dev/null || cat response.json
echo ""
echo "── Status ───────────────────────────────────────────"

# Parse success field from response body
SUCCESS=$(cat response.json | python3 -c "
import json, sys
try:
    outer = json.load(sys.stdin)
    body  = json.loads(outer.get('body', '{}'))
    print('true' if body.get('success') else 'false')
except:
    print('unknown')
" 2>/dev/null || echo "unknown")

if [ "$SUCCESS" = "true" ]; then
    echo "✅ Lambda invocation SUCCEEDED"
else
    echo "❌ Lambda invocation FAILED — check response above"
fi

rm -f response.json
echo ""
