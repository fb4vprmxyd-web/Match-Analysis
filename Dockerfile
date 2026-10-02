# One image, three uses in docker-compose.yml - all write to whatever's
# mounted at APP_DATA_DIR, none of them know whether that's a local volume
# or the droplet's real storage:
#   - `app`:       one-shot/manual (`docker compose run --rm app python3 run.py`)
#   - `scheduler`: this CMD overridden to scheduler.py, left running
#                  (restart: unless-stopped) so the Mon/Thu pulls happen
#                  on their own once deployed - see scheduler.py.
#   - `webapp`:    this CMD overridden to gunicorn, left running and
#                  exposed - the actual dashboard a coach's assistant signs
#                  into. See webapp.py and auth.py.

FROM python:3.12-slim

# freetype/fontconfig for matplotlib's font rendering; libgl1 for PyMuPDF.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libfreetype6 fontconfig libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Data lives on the mounted volume, not the image - a rebuild of the image
# never touches what's already been pulled or built. See envconfig.py.
ENV APP_DATA_DIR=/data
VOLUME ["/data"]

CMD ["python3", "run.py"]
