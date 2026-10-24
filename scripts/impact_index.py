"""
Fit the Off-Ball Impact Index's weights on every synthetic match and score the players.

    python scripts/impact_index.py [--block 5] [--out outputs/impact_index]

Writes weights.csv (each metric's weight in shots per SD, with its bootstrap interval),
rankings.csv and polygon.json, and prints the weights and the top five per position.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core import composite, impact  # noqa: E402
from scripts.composite_index import league  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    parser.add_argument("--block", type=float, default=5.0, help="window length in minutes")
    parser.add_argument("--out", default="outputs/impact_index")
    args = parser.parse_args()
    games, data, pool = league()
    blocks = impact.team_blocks(data, games, args.block)
    weights = impact.fit_weights(blocks)
    config = impact.index_config(weights["weight"])
    scored = composite.composite(pool, config)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    weights.to_csv(out / "weights.csv")
    columns = ["name", "real_team", "broad_position", "minutes", "eligible", "coverage", "index", *impact.PILLARS,
               *impact.METRICS, *(f"{m}_score" for m in impact.METRICS)]
    scored.sort_values("index", ascending=False)[columns].to_csv(out / "rankings.csv", index=False)
    (out / "polygon.json").write_text(json.dumps(composite.polygon(scored, config), indent=1))

    print(f"{len(blocks)} team windows of {args.block:g} min, {len(games)} matches. Weights (shots per SD):")
    print(weights.round(3).to_string())
    print("Pillar weights:", {c["label"]: round(c["weight"], 3) for c in config["categories"].values()})
    ranked = scored[scored["eligible"]]
    for position, group in ranked.groupby("broad_position"):
        top = group.nlargest(5, "index")[["name", "real_team", "minutes", "index", *impact.PILLARS]]
        print(f"\n{position} (n={len(group)})\n{top.round(1).to_string(index=False)}")
    print(f"\nWrote {out}/weights.csv, rankings.csv and polygon.json")


if __name__ == "__main__":
    main()
