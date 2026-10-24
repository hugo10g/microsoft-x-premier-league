# Off-ball metric definitions

Every threshold lives in `core/config.py`. Each metric is computed from 25 Hz tracking (`positions (T, P, 2)` in metres), the event file and the simulated head-orientation feed. Each event is credited to the name a tracked player carried at that minute.

## Four pillars

| Pillar | Column | Definition |
|---|---|---|
| Scanning | `scan_rate` | Head checks per second in the 5 s before a completed reception. See the scanning rule below. |
| Space | `decoy_p90` | xT-weighted pitch control that teammates gain from an off-ball run, per 90. See the run rules below. |
| Pressing | `lane_cut_p90` | Seconds of threat-weighted passing lanes cut per 90. Counted on every frame the player is within 15 m of the ball while the opponent has it: the share of the carrier's passing lanes inside their cover shadow, projected 0.5 s ahead. |
| Transition | `transition_score`, `median_latency` | Time from an open-play turnover to a 1 s sprint above 20 km/h, for players within 30 m of the ball, over a 10 s horizon. The score rewards fast sprints (falling to 1/e at 2 s) and penalises walking. Latency is the median over the turnovers where the player did sprint. |

**Scanning rule.**
- A head check is a turn more than 60° away from the ball.
- The head must turn at a peak speed of at least 90°/s.
- The check lasts 0.1–2 s.
- Receptions with too little head data are left out rather than counted as zero.

**Decoy run rules.** A run must:
- be at least 15 km/h, for at least 1 s, over at least 8 m;
- happen while the player's team has the ball and the player doesn't;
- end level with or ahead of the ball, because a run behind the ball is the defensive line moving;
- start more than 3 s after the team won the ball, because earlier runs are transitions.

**How a run is valued.**
- Markers are the defenders who had the runner as their nearest attacker for at least 60% of the run and moved with them.
- The counterfactual puts each marker back where they would have been had they shifted with the rest of the block.
- The value is clamped at 0, so a run is never penalised.

## Five tactical metrics

| Metric | Column | Definition |
|---|---|---|
| Pressure Intensity | `pressure_intensity` | Spells of at least 0.4 s within 10 m of a controlled carrier, closing faster than 5 km/h. Metres closed, weighted by proximity, per 90. |
| Press and win | `press_wins`, `press_wins_p90`, `press_win_pct` | A pressure spell is *won* when, between the spell's start and 3 s after it ends, the presser's team wins the ball by a tackle or an interception (a `RECOVERY` event; keeper saves excluded), or the pressed team passes it out of play (a forced error; a shot off target doesn't count). `won_how` says which and `regained_by` who won it. Each player is credited once per win. Count, per 90 (in the Off-Ball Index), and share of the player's pressures. |
| Receipts in Space | `receipts_in_space_pct` | Share of receptions with no opponent within 5 m when the pass arrives |
| Defensive Responsibility | `def_responsibility_p90` | xT gained by opponents' passes ending in the player's zone (their slot in the team's shape), shared by distance, per 90. Lower is better. |
| OBSO | `obso_p90` | Every 0.2 s in possession: the off-ball attacker's xT × how free they are × how reachable they are from the ball. Per 90. |
| Run types | `runs_in_behind_p90`, `pulling_wide_p90`, `dropping_deep_p90` | Each run, measured against the team's own movement, classified as in behind (ends past the last defender), pulling wide or dropping deep. Per 90. Describes style, not quality. |

## Composites

| Score | What it is | Method |
|---|---|---|
| **Off-Ball Index** (headline) | Ten metrics in five equally weighted categories, percentiles within position. In match views it is computed within that match. | [composite_index.md](composite_index.md) |
| Pillar score (`off_ball_value`) | Mean percentile of the four pillars among the players shown; a building block | — |
| Off-Ball Impact Index | The five tactical metrics, weighted by their association with shot difference | [impact_index.md](impact_index.md) |
