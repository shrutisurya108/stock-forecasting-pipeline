# ──────────────────────────────────────────────────────────────────────────────
# Dockerfile — Lambda container image for Stock Forecasting Pipeline
# Base: AWS Lambda Python 3.11 (Amazon Linux 2, GCC 7.3)
#
# Strategy: install EVERYTHING with --only-binary :all: so nothing ever
# compiles from source. Each package pinned to last version with a
# pre-built cp311 manylinux wheel compatible with numpy 1.26.4.
# ──────────────────────────────────────────────────────────────────────────────

FROM public.ecr.aws/lambda/python:3.11

# ── Upgrade pip only ──────────────────────────────────────────────────────────
RUN pip install --no-cache-dir --upgrade pip

# ── Step 1: Core numerics — all pre-built wheels ──────────────────────────────
RUN pip install --no-cache-dir --only-binary :all: \
        "numpy==1.26.4" \
        "scipy==1.13.1" \
        "pandas==2.2.3" \
        "pyarrow==16.1.0"

# ── Step 2: PyTorch CPU (pre-built wheel from pytorch index) ──────────────────
RUN pip install --no-cache-dir --only-binary :all: \
        "torch==2.6.0+cpu" \
        --index-url https://download.pytorch.org/whl/cpu

# ── Step 3: ML packages — pre-built wheels ────────────────────────────────────
RUN pip install --no-cache-dir --only-binary :all: \
        "scikit-learn==1.4.2" \
        "statsmodels==0.14.2" \
        "contourpy==1.2.1" \
        "matplotlib==3.8.4"

# ── Step 4: pmdarima — has cp311 manylinux wheel ─────────────────────────────
RUN pip install --no-cache-dir --only-binary :all: \
        "pmdarima==2.0.4"

# ── Step 5: Prophet and its dependencies ─────────────────────────────────────
# prophet itself is pure Python but pulls matplotlib (pinned above).
# Pre-install all its compiled deps with pinned wheels first.
RUN pip install --no-cache-dir --only-binary :all: \
        "Pillow==10.4.0" \
        "kiwisolver==1.4.7" \
        "cycler==0.12.1" \
        "fonttools==4.55.3" \
        "pyparsing==3.2.3" \
        "python-dateutil==2.9.0" && \
    pip install --no-cache-dir \
        "prophet==1.1.6" \
        --no-build-isolation

# ── Step 6: Pure-Python packages (no compilation) ────────────────────────────
RUN pip install --no-cache-dir \
        "boto3>=1.35.0" \
        "yfinance>=0.2.54" \
        "python-dotenv>=1.0.1" \
        "requests>=2.32.0"

# ── Copy project source ───────────────────────────────────────────────────────
COPY config/        ${LAMBDA_TASK_ROOT}/config/
COPY data/          ${LAMBDA_TASK_ROOT}/data/
COPY models/        ${LAMBDA_TASK_ROOT}/models/
COPY training/      ${LAMBDA_TASK_ROOT}/training/
COPY forecasting/   ${LAMBDA_TASK_ROOT}/forecasting/
COPY pipeline/      ${LAMBDA_TASK_ROOT}/pipeline/
COPY storage/       ${LAMBDA_TASK_ROOT}/storage/
COPY scripts/       ${LAMBDA_TASK_ROOT}/scripts/

# ── Runtime directories in /tmp ───────────────────────────────────────────────
RUN mkdir -p /tmp/data/raw /tmp/predictions/models /tmp/logs

# ── Environment defaults ──────────────────────────────────────────────────────
ENV LOG_LEVEL="INFO" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# ── Lambda handler entry point ────────────────────────────────────────────────
CMD ["pipeline.lambda_handler.handler"]
