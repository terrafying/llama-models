FROM python:3.13-alpine

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apk add --no-cache \
    build-base \
    curl \
    git \
    ffmpeg \
    python3-dev \
    py3-pip \
    gcc \
    musl-dev \
    linux-headers

# Set working directory
WORKDIR /app

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies using uv
RUN uv pip install -r requirements.txt

# Copy project files
COPY . .

# Install project in editable mode
RUN uv pip install -e .

# Expose port for web interface
EXPOSE 7860

# Set default command
CMD ["python", "-m", "ragtime_llm.web_interface"] 