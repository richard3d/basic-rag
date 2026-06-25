FROM ghcr.io/astral-sh/uv:python3.14-alpine

WORKDIR /app

# Bring the toml and lockfile so we can 
# detect drift when running uv sync --frozen
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY src/ .

CMD ["uv", "run", "python", "-u", "main.py"]
