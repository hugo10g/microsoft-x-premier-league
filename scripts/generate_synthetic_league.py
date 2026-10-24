"""
Generate the synthetic league the dashboard runs on: four fictional clubs, a full
round robin (six matches), written to data/synthetic/ in Metrica's CSV layout.

Everything is fictional and reproducible from the seed. The players' hidden
attributes go to data/synthetic/league_players.csv, the ground truth the
off-ball metrics can be checked against.

Run:  python scripts/generate_synthetic_league.py [--half-minutes 45] [--seed 7]
"""

import argparse
import itertools
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.synthetic import make_squad, value_players, write_match  # noqa: E402

OUTPUT = Path(__file__).resolve().parent.parent / "data" / "synthetic"
CLUBS = [("Harbourside FC", "4-3-3"), ("Northvale Athletic", "4-4-2"),
         ("Redmoor Rovers", "3-5-2"), ("Ashcombe City", "4-3-3")]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--half-minutes", type=float, default=45.0)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    used: set = set()
    squads = {club: make_squad(club, formation, rng, used) for club, formation in CLUBS}
    value_players([p for squad in squads.values() for p in squad], rng)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([asdict(p) | {"position": p.position} for squad in squads.values() for p in squad]) \
        .to_csv(OUTPUT / "league_players.csv", index=False)

    for matchday, (home, away) in enumerate(itertools.combinations(squads, 2), 1):
        folder = OUTPUT / f"Matchday_{matchday}_{home.split()[0]}_v_{away.split()[0]}"
        started = time.time()
        result = write_match(folder, squads[home], squads[away], seed=args.seed * 100 + matchday,
                             half_minutes=args.half_minutes)
        print(f"{folder.name}: {result['home']} {result['score']} {result['away']}, "
              f"{result['events']} events ({time.time() - started:.0f} s)")


if __name__ == "__main__":
    main()
