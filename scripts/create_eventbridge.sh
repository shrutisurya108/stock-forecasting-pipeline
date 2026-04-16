#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# scripts/create_eventbridge.sh
# Creates an EventBridge rule that triggers the Lambda function at 01:00 UTC
# every day. Run this AFTER deploy_lambda.sh.
#
# Usage:
#   bash scripts/create_eventbridge.sh
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail

AWS_REGION="${AWS_REGION:-us-east-1}"
RULE_NAME="stock-forecasting-daily"
LAMBDA_FUNCTION="stock-forecasting-pipeline"

echo ""
echo "═══════════════════════════════════════════════════════"
echo "  Creating EventBridge Schedule Rule"
echo "  Rule  : ${RULE_NAME}"
echo "  Cron  : 0 1 * * ? *  (01:00 UTC daily)"
echo "  Target: Lambda → ${LAMBDA_FUNCTION}"
echo "═══════════════════════════════════════════════════════"
echo ""

# ── Step 1: Get Lambda ARN ────────────────────────────────────────────────────
if [ -f ".lambda_arn" ]; then
    LAMBDA_ARN=$(cat .lambda_arn)
    echo "[1/4] Using Lambda ARN from .lambda_arn: ${LAMBDA_ARN}"
else
    LAMBDA_ARN=$(aws lambda get-function \
        --function-name "${LAMBDA_FUNCTION}" \
        --region "${AWS_REGION}" \
        --query "Configuration.FunctionArn" \
        --output text)
    echo "[1/4] Retrieved Lambda ARN: ${LAMBDA_ARN}"
fi

# ── Step 2: Create EventBridge rule ──────────────────────────────────────────
echo ""
echo "[2/4] Creating EventBridge rule..."
RULE_ARN=$(aws events put-rule \
    --name "${RULE_NAME}" \
    --schedule-expression "cron(0 1 * * ? *)" \
    --state ENABLED \
    --description "Daily stock forecast pipeline trigger at 01:00 UTC" \
    --region "${AWS_REGION}" \
    --query "RuleArn" \
    --output text)
echo "      ✅ Rule created: ${RULE_ARN}"

# ── Step 3: Grant EventBridge permission to invoke Lambda ─────────────────────
echo ""
echo "[3/4] Granting EventBridge permission to invoke Lambda..."
# Remove existing permission first (ignore error if it doesn't exist)
aws lambda remove-permission \
    --function-name "${LAMBDA_FUNCTION}" \
    --statement-id "EventBridgeDailyTrigger" \
    --region "${AWS_REGION}" 2>/dev/null || true

aws lambda add-permission \
    --function-name "${LAMBDA_FUNCTION}" \
    --statement-id "EventBridgeDailyTrigger" \
    --action "lambda:InvokeFunction" \
    --principal "events.amazonaws.com" \
    --source-arn "${RULE_ARN}" \
    --region "${AWS_REGION}" \
    --output json > /dev/null
echo "      ✅ Permission granted"

# ── Step 4: Add Lambda as EventBridge target ──────────────────────────────────
echo ""
echo "[4/4] Adding Lambda as EventBridge target..."

# The event payload sent to Lambda on each trigger
EVENT_PAYLOAD='{
  "mode": "forecast_only",
  "benchmark": false
}'

aws events put-targets \
    --rule "${RULE_NAME}" \
    --targets "[{
        \"Id\": \"StockForecastingLambda\",
        \"Arn\": \"${LAMBDA_ARN}\",
        \"Input\": $(echo ${EVENT_PAYLOAD} | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))')
    }]" \
    --region "${AWS_REGION}" \
    --output json > /dev/null
echo "      ✅ Lambda added as EventBridge target"

echo ""
echo "═══════════════════════════════════════════════════════"
echo "  ✅ EventBridge rule created!"
echo ""
echo "  Schedule: Every day at 01:00 UTC"
echo "  Mode    : forecast_only"
echo "  Lambda  : ${LAMBDA_FUNCTION}"
echo ""
echo "  To verify: AWS Console → EventBridge → Rules → ${RULE_NAME}"
echo "  To test now: bash scripts/test_lambda.sh"
echo "═══════════════════════════════════════════════════════"
echo ""
