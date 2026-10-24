"""
Score every synthetic match and write the Composite Off-Ball Index.

    python scripts/composite_index.py [--config config/composite_index.json] [--out outputs/composite_index]

Writes rankings.csv (every player, ranked or not, with each metric's raw value and
score) and polygon.json (radar data for the ranked players), then prints the top five
per position and any pair of metric scores correlated beyond |rho| 0.7.
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core import composite, pipeline, roster  # noqa: E402

SEASON = "synthetic"        # the league is one season; real data would carry its own column


def league() -> tuple[list, dict, pd.DataFrame]:
    """(games, core.pipeline.score_all tables, every name's season totals as the app builds them)."""
    games = pipeline.available_games()
    if not games:
        sys.exit("No games in data/synthetic: run scripts/generate_synthetic_league.py first.")
    data = pipeline.score_all(pipeline.learn_xt(games), games)
    names = pd.concat([pipeline.load_roster(g).assign(uid=lambda t, n=g.name: n + ":" + t["player"])
                       for g in games], ignore_index=True).rename(columns={"club": "real_team"})
    names = names[["uid", "name", "real_team", "position", "detailed_position", "market_value_m"]]
    totals = pipeline.player_totals(data, roster.static_timeline(data["profiles"], names, key="uid"), on="uid")
    return games, data, pipeline.off_ball_value(totals).assign(season=SEASON)   # adds decoy_p90, lane_cut_p90


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    parser.add_argument("--config", default=composite.CONFIG_PATH)
    parser.add_argument("--out", default="outputs/composite_index")
    args = parser.parse_args()
    config = composite.load_config(args.config)
    scored = composite.composite(league()[2], config)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    metrics = [m for c in config["categories"].values() for m in c["metrics"]]
    columns = ["name", "real_team", "season", "broad_position", "minutes", "eligible", "coverage", "index",
               *config["categories"], *metrics, *(f"{m}_score" for m in metrics)]
    scored.sort_values("index", ascending=False)[columns].to_csv(out / "rankings.csv", index=False)
    (out / "polygon.json").write_text(json.dumps(composite.polygon(scored, config), indent=1))

    ranked = scored[scored["eligible"]]
    print(f"{len(ranked)} ranked, {(~scored['eligible']).sum()} not (under {config['min_minutes']} min "
          f"or excluded position); {ranked['index'].isna().sum()} ranked below coverage {config['min_coverage']}.")
    for position, group in ranked.groupby("broad_position"):
        top = group.nlargest(5, "index")[["name", "real_team", "minutes", "index", *config["categories"]]]
        print(f"\n{position} (n={len(group)})\n{top.round(1).to_string(index=False)}")
    rho = ranked[[f"{m}_score" for m in metrics]].corr("spearman")
    pairs = [(a, b, rho.loc[a, b]) for i, a in enumerate(rho) for b in rho.columns[i + 1:] if abs(rho.loc[a, b]) > 0.7]
    print("\nCollinear score pairs (|rho| > 0.7):", *(f"  {a} ~ {b}: {r:.2f}" for a, b, r in pairs) or ["  none"],
          sep="\n")
    print(f"\nWrote {out / 'rankings.csv'} and {out / 'polygon.json'}")


if __name__ == "__main__":
    main()
