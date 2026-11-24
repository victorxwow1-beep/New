FROM python:3.13-slim
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg fonts-dejavu-core && rm -rf /var/lib/apt/lists/*
WORKDIR /studio
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py engine.py ./
COPY static ./static
RUN useradd --create-home studio && mkdir -p /data && chown -R studio:studio /data /studio
USER studio
ENV ZENVYRA_DATA_DIR=/data
EXPOSE 8000
CMD ["sh","-c","exec python -m uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
