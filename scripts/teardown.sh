#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# scripts/teardown.sh
# Cleanly removes all AWS resources created by this project.
# Run this when you're done with the project to ensure zero ongoing cost.
#
# Removes:
#   - EventBridge rule and targets
#   - Lambda function
#   - ECR repository and all images
#   - S3 bucket contents (optionally the bucket itself)
#   - IAM role and policies
#
# Usage:
#   bash scripts/teardown.sh
# ──────────────────────────────────────────────────────────────────────────────

set -uo pipefail   # don't use -e — we want to continue even if one delete fails

AWS_REGION="${AWS_REGION:-us-east-1}"
RULE_NAME="stock-forecasting-daily"
LAMBDA_FUNCTION="stock-forecasting-pipeline"
ECR_REPO="stock-forecasting-pipeline"
S3_BUCKET="${S3_BUCKET_NAME:-stock-forecasting-pipeline}"
IAM_ROLE="stock-forecasting-lambda-role"

echo ""
echo "═══════════════════════════════════════════════════════"
echo "  Stock Forecasting Pipeline — Teardown"
echo "  This will remove all AWS resources."
echo "═══════════════════════════════════════════════════════"
echo ""
read -p "Are you sure? Type 'yes' to confirm: " CONFIRM
if [ "$CONFIRM" != "yes" ]; then
    echo "Aborted."
    exit 0
fi
echo ""

# ── 1. Delete EventBridge rule ────────────────────────────────────────────────
echo "[1/6] Removing EventBridge rule..."
aws events remove-targets \
    --rule "${RULE_NAME}" \
    --ids "StockForecastingLambda" \
    --region "${AWS_REGION}" 2>/dev/null && echo "      Targets removed" || echo "      No targets found"

aws events delete-rule \
    --name "${RULE_NAME}" \
    --region "${AWS_REGION}" 2>/dev/null && echo "      ✅ Rule deleted" || echo "      Rule not found"

# ── 2. Delete Lambda function ─────────────────────────────────────────────────
echo ""
echo "[2/6] Deleting Lambda function..."
aws lambda delete-function \
    --function-name "${LAMBDA_FUNCTION}" \
    --region "${AWS_REGION}" 2>/dev/null && echo "      ✅ Lambda deleted" || echo "      Function not found"

# ── 3. Delete ECR repository ──────────────────────────────────────────────────
echo ""
echo "[3/6] Deleting ECR repository (all images)..."
aws ecr delete-repository \
    --repository-name "${ECR_REPO}" \
    --region "${AWS_REGION}" \
    --force 2>/dev/null && echo "      ✅ ECR repository deleted" || echo "      Repository not found"

# ── 4. Empty and optionally delete S3 bucket ─────────────────────────────────
echo ""
echo "[4/6] Emptying S3 bucket: ${S3_BUCKET}..."
aws s3 rm "s3://${S3_BUCKET}" --recursive 2>/dev/null && \
    echo "      ✅ Bucket emptied" || echo "      Bucket not found or already empty"

read -p "      Delete the S3 bucket entirely? (yes/no): " DELETE_BUCKET
if [ "$DELETE_BUCKET" = "yes" ]; then
    aws s3 rb "s3://${S3_BUCKET}" 2>/dev/null && \
        echo "      ✅ Bucket deleted" || echo "      Could not delete bucket"
else
    echo "      Bucket kept (contents removed)"
fi

# ── 5. Detach and delete IAM role ─────────────────────────────────────────────
echo ""
echo "[5/6] Removing IAM role..."
aws iam detach-role-policy \
    --role-name "${IAM_ROLE}" \
    --policy-arn "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole" \
    2>/dev/null || true
aws iam detach-role-policy \
    --role-name "${IAM_ROLE}" \
    --policy-arn "arn:aws:iam::aws:policy/AmazonS3FullAccess" \
    2>/dev/null || true
aws iam delete-role \
    --role-name "${IAM_ROLE}" \
    2>/dev/null && echo "      ✅ IAM role deleted" || echo "      Role not found"

# ── 6. GitHub Actions ─────────────────────────────────────────────────────────
echo ""
echo "[6/6] GitHub Actions"
echo "      To stop the daily workflow:"
echo "      → GitHub → Actions → Daily Retrain Pipeline → ⋯ → Disable workflow"

echo ""
echo "═══════════════════════════════════════════════════════"
echo "  ✅ AWS teardown complete — zero ongoing cost"
echo ""
echo "  Local project files are untouched."
echo "  Streamlit Cloud: go to share.streamlit.io → delete app"
echo "═══════════════════════════════════════════════════════"
echo ""
