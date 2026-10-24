# Off-Ball Value dashboard: Streamlit on port 8501.
#
#   docker build -t off-ball-value . && docker run -p 8501:8501 --env-file .env off-ball-value
#
# The synthetic league is git-ignored (~550 MB). When it is in the build context it is
# copied in unchanged (the exact matches the numbers in the docs come from); otherwise it
# is regenerated from its fixed seed. Model credentials are environment variables at run
# time (see .env.example) and never part of the image.
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
RUN test -f data/synthetic/league_players.csv || python scripts/generate_synthetic_league.py
# Score the league at build time, so a container that scaled to zero starts in seconds.
RUN python scripts/precompute_scores.py

RUN useradd --create-home app && chown -R app /app
USER app

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", \
     "--server.headless=true", "--browser.gatherUsageStats=false"]
