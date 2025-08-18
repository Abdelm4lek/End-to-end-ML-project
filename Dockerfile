FROM python:3.11-slim AS base

# add full build toolchain, then remove it later if you want
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential         \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# API image (matches Dockerfile.api behavior)
FROM base AS api
WORKDIR /app
COPY . .
ENV PYTHONPATH=/app/src
EXPOSE 8000
CMD ["uvicorn", "src.mlProject.api.main:app", "--host", "0.0.0.0", "--port", "8000"]


# Web/Streamlit image (matches Dockerfile.web behavior)
FROM base AS streamlit
WORKDIR /app
COPY . .
ENV PYTHONPATH=/app/src
EXPOSE 8501
CMD ["streamlit", "run", "web_app.py", "--server.port=8501", "--server.address=0.0.0.0"]