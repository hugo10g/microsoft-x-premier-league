"""
Score every synthetic match once and save the tables next to them (data/synthetic/scored.pkl, xt.pkl).

    python scripts/precompute_scores.py

The dashboard loads this file when it matches the games and pipeline version, so a
fresh start (a new container, a cleared cache) skips the minutes-long scoring.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core import pipeline  # noqa: E402

if __name__ == "__main__":
    games = pipeline.available_games()
    if not games:
        sys.exit("No games in data/synthetic: run scripts/generate_synthetic_league.py first.")
    signature = pipeline.games_signature(games)
    xt = pipeline.learn_xt(games)
    pipeline.save_scored(signature, xt, pipeline.XT_FILE)
    pipeline.save_scored(signature, pipeline.score_all(xt, games, progress=lambda i, n: print(f"Scored {i}/{n}")))
    print(f"Saved {pipeline.SCORED_FILE} and {pipeline.XT_FILE.name} for {len(games)} games, "
          f"pipeline version {pipeline.VERSION}")
