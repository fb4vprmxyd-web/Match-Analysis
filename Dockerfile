# Runs the pipeline (run.py, build_*.py) - not a web server. There isn't
# one yet; the dashboard/report files this produces are served some other
# way for now (see README). This image just does the pull-and-build work
# and writes its output onto whatever's mounted at APP_DATA_DIR, so it's
# invoked on demand (`docker compose run app python3 run.py`), not left
# running - matching how run.py already describes its own usage: "run it
# whenever you know data has landed, nothing runs on a timer."

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
