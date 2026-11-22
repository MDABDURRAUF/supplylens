# SupplyLens dashboard
# Build: docker build -t supplylens .
# Run:   docker run --rm -p 8501:8501 supplylens
# Then open http://localhost:8501
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[app]"
COPY app.py ./
COPY examples ./examples
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
