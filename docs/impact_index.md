# Off-Ball Impact Index

The five tactical metrics, combined into one 0–100 score per player. Each metric is weighted by how much it moves the team's shot balance. The Off-Ball Index weights its categories equally; this one lets the matches set the weights.

- Code: `core/impact.py`. It reuses `core/composite.py` to score the players.
- Run: `python scripts/impact_index.py`. This writes `outputs/impact_index/weights.csv`, `rankings.csv` and `polygon.json`.
- In the app: the Off-Ball Index page → **Off-Ball Impact**. It is also on the Value Finder axis and in the AI Scout.
- Tests: `tests/test_impact.py`

## Pillars

| Pillar | Metric(s) | Team total per window, normalised by |
|---|---|---|
| Pressure | `pressure_intensity` | minutes the opponent has the ball |
| Receipts in space | `receipts_in_space_pct` | share of the team's receptions |
| Defensive cover | `def_responsibility_p90`, negated (less is better) | minutes the opponent has the ball |
| OBSO | `obso_p90` | minutes of the team's own possession |
| Run types | `runs_in_behind_p90`, `pulling_wide_p90`, `dropping_deep_p90` | minutes of the team's own possession |

## Method

1. Cut every match into 5-minute windows, per team: 236 windows across 6 matches. Windows where either side had the ball for under 30 s are dropped.
2. Standardise each metric's team total in each window.
3. Remove possession share from the metrics and from the outcome, so a team isn't credited just for having the ball.
4. Regress the window's shot difference (shots for − shots against) on the metrics by non-negative least squares.
   - A weight is the extra shots one standard deviation of that work goes with.
   - A metric with no detectable benefit gets 0 rather than a negative weight, so being good at something never lowers a player's score.
5. Bootstrap whole matches (500 resamples) for a 95% interval on each weight.
6. A pillar's weight is the sum of its metrics' weights.
7. Score players as in the Off-Ball Index:
   - percentiles within position, shrunk towards 50 on small samples;
   - inside a pillar, its metrics count equally, so the radar shows the player's work on each pillar;
   - the fitted pillar weights decide how much each pillar counts in the index.

**Why shots, not xT.** Defensive responsibility is built from the xT that opponents' passes gain. Fitting it to xT would be circular. Shots are the nearest outcome that none of the five metrics is built from.

**Why team windows, not players.** A shot balance belongs to the whole team. The fit says how much each pillar matters; players are then scored on their own work on each pillar.

## Results on the synthetic league

| Metric | Weight (shots per SD) | 95% interval | Above 0 in |
|---|---|---|---|
| OBSO | 0.44 | 0.35–0.59 | 100% |
| Defensive cover | 0.32 | 0.18–0.49 | 100% |
| Pressure intensity | 0.08 | 0–0.22 | 84% |
| Receipts in space | 0 | 0–0.18 | 40% |
| Run types (all three) | 0 | 0–0.08 | 9–36% |

That makes the pillar weights about **52% OBSO, 38% Defensive cover, 9% Pressure, and 0 for Receipts in space and Run types**.

Checked against the hidden traits (Spearman, within position, n = 40):

| Score | Correlation |
|---|---|
| Impact index vs the Off-Ball Index | 0.61 |
| Impact index vs the mean of work rate, pressing and awareness | 0.13 |
| Pressure pillar vs pressing | 0.73 |
| Run types pillar vs work rate | 0.45 |
| OBSO pillar | no trait |
| Defensive cover pillar | no trait |

## How to read it

The two indexes answer different questions:
- The **Off-Ball Index** asks *how much off-ball work does a player do*. It recovers the hidden talents.
- The **Impact Index** asks *which work goes with out-shooting the opponent*.

In this league the answer is mostly being free in dangerous space (OBSO) and not letting the ball into your zone (defensive cover). Neither is driven by a player's hidden talent. Both depend largely on role, team and game state.

## Limitations

- **Closeness to the outcome.**
  - OBSO and defensive cover sit one step from a shot, so they are bound to explain shots best.
  - Pressing and runs pay off indirectly: a press wins the ball, and a run opens space for someone else. A 5-minute shot balance can miss that, so they can look worthless here even when they matter.
- **Association, not causation.** A team that is winning may sit back, press less and still out-shoot the opponent late on. Possession share is controlled for; game state is not.
- **Sample size.** Six matches and 236 windows. The order of the weights is fairly stable (OBSO and cover are positive in every resample), but the exact numbers are not.
- **Team-level weights.** Every player is weighted the same way. Position-specific weights would need many more matches.
- **Synthetic data.** The weights describe how this simulator turns off-ball work into shots. On real tracking data they would need refitting, and could differ a lot.
