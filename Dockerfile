# ---- Base image ----
FROM python:3.11-slim

# ---- System deps ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && \
    rm -rf /var/lib/apt/lists/*

# ---- Workdir ----
WORKDIR /app

# ---- Copy project ----
COPY pyproject.toml uv.lock ./
COPY src ./src
COPY README.md ./
COPY .env.example ./

# ---- Install uv ----
RUN pip install --no-cache-dir uv

# ---- Create venv + install deps ----
RUN uv venv /opt/venv && \
    . /opt/venv/bin/activate && \
    uv pip install -e .

# ---- Ensure venv is active ----
ENV PATH="/opt/venv/bin:$PATH"

# ---- Entrypoint ----
ENTRYPOINT ["python", "-m", "ai_agents_01_call_llm.main"]
