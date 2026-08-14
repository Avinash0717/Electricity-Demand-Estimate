# --- VoltCast: Electricity Demand Forecasting API ---
FROM python:3.12-slim

WORKDIR /app

# Install dependencies first (better Docker layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code + model artifacts
COPY main.py .
COPY models/rf_model.pkl ./models/rf_model.pkl
COPY model_metadata.json .
COPY static/ ./static/

# Render (and most PaaS providers) inject $PORT at runtime.
# Default to 8000 for local `docker run`.
ENV PORT=8000
EXPOSE 8000

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
