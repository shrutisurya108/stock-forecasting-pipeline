#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# scripts/deploy_lambda.sh
# Builds the Lambda container image, pushes it to ECR, and deploys the
# Lambda function. Run this once to deploy, and again whenever you update code.
#
# Prerequisites:
#   - AWS CLI installed and configured (aws configure)
#   - Docker Desktop running
#   - ECR repository already created (scripts/create_ecr.sh)
#
# Usage:
#   bash scripts/deploy_lambda.sh
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail   # exit on error, undefined var, or pipe failure

# ── Configuration (edit these if needed) ─────────────────────────────────────
AWS_REGION="${AWS_REGION:-us-east-1}"
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_REPO="stock-forecasting-pipeline"
LAMBDA_FUNCTION="stock-forecasting-pipeline"
IMAGE_TAG="latest"
ECR_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO}:${IMAGE_TAG}"
LAMBDA_MEMORY=1024      # MB — enough for PyTorch inference
LAMBDA_TIMEOUT=900      # seconds (15 min — Lambda maximum)

echo ""
echo "═══════════════════════════════════════════════════════"
echo "  Stock Forecasting Pipeline — Lambda Deployment"
echo "═══════════════════════════════════════════════════════"
echo "  Account : ${AWS_ACCOUNT_ID}"
echo "  Region  : ${AWS_REGION}"
echo "  ECR URI : ${ECR_URI}"
echo "  Function: ${LAMBDA_FUNCTION}"
echo "═══════════════════════════════════════════════════════"
echo ""

# ── Step 1: Authenticate Docker with ECR ─────────────────────────────────────
echo "[1/5] Authenticating Docker with ECR..."
aws ecr get-login-password --region "${AWS_REGION}" | \
    docker login --username AWS --password-stdin \
    "${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
echo "      ✅ Docker authenticated with ECR"

# ── Step 2: Build Docker image ────────────────────────────────────────────────
echo ""
echo "[2/5] Building Docker image (this takes 3-5 min on first run)..."
docker build \
    --platform linux/amd64 \
    --tag "${ECR_URI}" \
    --tag "${ECR_REPO}:${IMAGE_TAG}" \
    .
echo "      ✅ Image built: ${ECR_URI}"

# ── Step 3: Push image to ECR ────────────────────────────────────────────────
echo ""
echo "[3/5] Pushing image to ECR..."
docker push "${ECR_URI}"
echo "      ✅ Image pushed to ECR"

# ── Step 4: Create or update Lambda function ──────────────────────────────────
echo ""
echo "[4/5] Deploying Lambda function..."

# Check if Lambda function already exists
if aws lambda get-function --function-name "${LAMBDA_FUNCTION}" \
    --region "${AWS_REGION}" > /dev/null 2>&1; then

    echo "      Function exists — updating code..."
    aws lambda update-function-code \
        --function-name "${LAMBDA_FUNCTION}" \
        --image-uri "${ECR_URI}" \
        --region "${AWS_REGION}" \
        --output json > /dev/null

    echo "      Waiting for update to complete..."
    aws lambda wait function-updated \
        --function-name "${LAMBDA_FUNCTION}" \
        --region "${AWS_REGION}"

    # Update configuration (memory, timeout, env vars)
    aws lambda update-function-configuration \
        --function-name "${LAMBDA_FUNCTION}" \
        --memory-size "${LAMBDA_MEMORY}" \
        --timeout "${LAMBDA_TIMEOUT}" \
        --environment "Variables={
            AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID},
            AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY},
            AWS_REGION=${AWS_REGION},
            S3_BUCKET_NAME=${S3_BUCKET_NAME:-stock-forecasting-pipeline},
            LOG_LEVEL=INFO
        }" \
        --region "${AWS_REGION}" \
        --output json > /dev/null

    echo "      ✅ Lambda function updated"

else
    echo "      Function does not exist — creating..."

    # Get or create execution role
    ROLE_ARN=$(aws iam get-role \
        --role-name "stock-forecasting-lambda-role" \
        --query "Role.Arn" \
        --output text 2>/dev/null || echo "")

    if [ -z "${ROLE_ARN}" ]; then
        echo "      Creating IAM execution role..."
        ROLE_ARN=$(aws iam create-role \
            --role-name "stock-forecasting-lambda-role" \
            --assume-role-policy-document '{
                "Version": "2012-10-17",
                "Statement": [{
                    "Effect": "Allow",
                    "Principal": {"Service": "lambda.amazonaws.com"},
                    "Action": "sts:AssumeRole"
                }]
            }' \
            --query "Role.Arn" \
            --output text)

        # Attach required policies
        aws iam attach-role-policy \
            --role-name "stock-forecasting-lambda-role" \
            --policy-arn "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
        aws iam attach-role-policy \
            --role-name "stock-forecasting-lambda-role" \
            --policy-arn "arn:aws:iam::aws:policy/AmazonS3FullAccess"

        echo "      ✅ IAM role created: ${ROLE_ARN}"
        echo "      Waiting 10s for role to propagate..."
        sleep 10
    fi

    aws lambda create-function \
        --function-name "${LAMBDA_FUNCTION}" \
        --package-type Image \
        --code "ImageUri=${ECR_URI}" \
        --role "${ROLE_ARN}" \
        --memory-size "${LAMBDA_MEMORY}" \
        --timeout "${LAMBDA_TIMEOUT}" \
        --environment "Variables={
            AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID},
            AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY},
            AWS_REGION=${AWS_REGION},
            S3_BUCKET_NAME=${S3_BUCKET_NAME:-stock-forecasting-pipeline},
            LOG_LEVEL=INFO
        }" \
        --region "${AWS_REGION}" \
        --output json > /dev/null

    echo "      Waiting for function to become active..."
    aws lambda wait function-active \
        --function-name "${LAMBDA_FUNCTION}" \
        --region "${AWS_REGION}"

    echo "      ✅ Lambda function created"
fi

# ── Step 5: Verify deployment ─────────────────────────────────────────────────
echo ""
echo "[5/5] Verifying deployment..."
FUNCTION_ARN=$(aws lambda get-function \
    --function-name "${LAMBDA_FUNCTION}" \
    --region "${AWS_REGION}" \
    --query "Configuration.FunctionArn" \
    --output text)

STATE=$(aws lambda get-function \
    --function-name "${LAMBDA_FUNCTION}" \
    --region "${AWS_REGION}" \
    --query "Configuration.State" \
    --output text)

echo "      ARN   : ${FUNCTION_ARN}"
echo "      State : ${STATE}"
echo "      ✅ Deployment verified"

echo ""
echo "═══════════════════════════════════════════════════════"
echo "  ✅ Lambda deployment complete!"
echo ""
echo "  Next steps:"
echo "  1. Run: bash scripts/create_eventbridge.sh"
echo "     (sets up daily 01:00 UTC trigger)"
echo ""
echo "  2. Test manually from AWS Console:"
echo "     Lambda → ${LAMBDA_FUNCTION} → Test"
echo "     Payload: {\"mode\": \"forecast_only\", \"benchmark\": false}"
echo "═══════════════════════════════════════════════════════"
echo ""

# Save function ARN for use by create_eventbridge.sh
echo "${FUNCTION_ARN}" > .lambda_arn
