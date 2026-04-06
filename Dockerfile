FROM python:3.11-slim

WORKDIR /app

# Copy requirements first (maximize layer caching)
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy all source code
COPY app/ ./app/

# inference.py must be accessible from container root (/app/inference.py)
COPY inference.py .

COPY openenv.yaml .

# Environment
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

EXPOSE 7860

# IMPORTANT: single worker — environment is stateful, multiple workers would break sessions
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
