# ──────────────────────────────────────────────────────────────────────────────
# Dockerfile
# Lambda container image for the Stock Forecasting Pipeline.
#
# Why container image instead of zip deployment:
#   - Zip packages are limited to 250 MB unzipped.
#   - PyTorch alone is ~800 MB. Container images support up to 10 GB.
#   - Container images are the AWS-recommended approach for ML workloads.
#
# Base image: AWS Lambda Python 3.11 (official, pre-wired for Lambda runtime)
# ──────────────────────────────────────────────────────────────────────────────

FROM public.ecr.aws/lambda/python:3.11

# ── System dependencies ───────────────────────────────────────────────────────
# Prophet requires gcc for compiling Stan; torch needs libgomp
RUN yum install -y gcc gcc-c++ make && yum clean all

# ── Copy requirements first (layer caching — only reinstalls on changes) ──────
COPY requirements.txt ${LAMBDA_TASK_ROOT}/

# ── Install Python dependencies ───────────────────────────────────────────────
# --no-cache-dir keeps image size smaller
# torch CPU-only keeps the image under ~3 GB
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r ${LAMBDA_TASK_ROOT}/requirements.txt

# ── Copy project source code ──────────────────────────────────────────────────
COPY config/        ${LAMBDA_TASK_ROOT}/config/
COPY data/          ${LAMBDA_TASK_ROOT}/data/
COPY models/        ${LAMBDA_TASK_ROOT}/models/
COPY training/      ${LAMBDA_TASK_ROOT}/training/
COPY forecasting/   ${LAMBDA_TASK_ROOT}/forecasting/
COPY pipeline/      ${LAMBDA_TASK_ROOT}/pipeline/
COPY storage/       ${LAMBDA_TASK_ROOT}/storage/
COPY scripts/       ${LAMBDA_TASK_ROOT}/scripts/

# ── Create required runtime directories ──────────────────────────────────────
# Lambda has a writable /tmp (512 MB). We redirect data/raw, predictions,
# logs, and saved models there so writes succeed in the read-only container.
RUN mkdir -p /tmp/data/raw /tmp/predictions/models /tmp/logs

# ── Environment variable defaults ─────────────────────────────────────────────
# Overridden at deploy time by deploy_lambda.sh and at runtime by Lambda config
ENV LOG_LEVEL="INFO" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# ── Lambda handler ────────────────────────────────────────────────────────────
# Points to pipeline/lambda_handler.py → handler()
CMD ["pipeline.lambda_handler.handler"]
