#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# scripts/create_ecr.sh
# Creates the ECR repository for the Lambda container image.
# Run this ONCE before running deploy_lambda.sh.
#
# Usage:
#   bash scripts/create_ecr.sh
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail

AWS_REGION="${AWS_REGION:-us-east-1}"
ECR_REPO="stock-forecasting-pipeline"

echo ""
echo "Creating ECR repository: ${ECR_REPO}"

# Check if repo already exists
if aws ecr describe-repositories \
    --repository-names "${ECR_REPO}" \
    --region "${AWS_REGION}" > /dev/null 2>&1; then
    echo "✅ Repository already exists — nothing to do"
else
    aws ecr create-repository \
        --repository-name "${ECR_REPO}" \
        --region "${AWS_REGION}" \
        --image-scanning-configuration scanOnPush=true \
        --output json > /dev/null
    echo "✅ ECR repository created: ${ECR_REPO}"
fi

AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
echo ""
echo "Repository URI:"
echo "  ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO}"
echo ""
echo "Next step: bash scripts/deploy_lambda.sh"
