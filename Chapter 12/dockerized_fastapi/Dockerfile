FROM python:3.12-slim
# Environment settings
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
# Create and switch to /app working directory
WORKDIR /app
# Update OS packages and install minimal required tools
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends gcc && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
# Copy only the needed files into the container
COPY . /app
# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
# Ensure log directory exists if used by loguru
RUN mkdir -p /app/logs
# Expose FastAPI port
EXPOSE 8000
# Run the app using Uvicorn (ASGI server)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]