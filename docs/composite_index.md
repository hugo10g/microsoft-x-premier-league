# Composite Off-Ball Index

One 0–100 number per player, plus five category subscores for a radar ("attribute polygon").
It is built only from metrics the pipeline already produces. It adds no new measurement.

- Code: `core/composite.py`
- Weights: `config/composite_index.json`
- Run: `python scripts/composite_index.py`. This writes `outputs/composite_index/rankings.csv` and `polygon.json`.
- Tests: `tests/test_composite.py`

## Metrics used

| Category | Metric | Direction | Sample for shrinkage |
|---|---|---|---|
| Spacing | `pitch_control_pct`: share of the pitch the player controls in possession | higher | minutes |
| Spacing | `receipts_in_space_pct`: receptions with no opponent within 5 m | higher | receptions |
| Movement | `decoy_p90`: xT created by off-ball runs per 90 | higher | minutes |
| Movement | `transition_score`: speed of the sprint reaction to turnovers | higher | transitions |
| Creation | `obso_p90`: off-ball scoring opportunity per 90 | higher | minutes |
| Scanning | `scan_rate`: head scans per second before receiving | higher | receptions |
| Defence | `lane_cut_p90`: seconds spent cutting passing lanes per 90 | higher | minutes |
| Defence | `pressure_intensity`: metres closed on the carrier per 90, weighted by proximity | higher | minutes |
| Defence | `press_wins_p90`: pressures per 90 after which the player's team won the ball within 3 s (tackle, interception, or a pass forced out of play) | higher | minutes |
| Defence | `def_responsibility_p90`: xT conceded through the player's zone per 90 | **lower** (flipped) | minutes |

**Left out on purpose:**
- Near-duplicates of a metric already used: `median_latency`, `sprint_rate`, `receipt_space`, `press_closing_kmh`, `mean_lane_cut` and `pre_reception_scans`.
- Run types (`runs_in_behind_p90`, `pulling_wide_p90`, `dropping_deep_p90`): they describe style, and more is not better.
- `retention_rate`: it measures what happens on the ball.
- `off_ball_value`: it is already a composite.
- Raw counts: they serve as the sample sizes instead.

## Method

1. **Eligibility.**
   - Outfielders with at least `min_minutes` (90) are ranked.
   - Everyone else stays in the output with `eligible = False` and NaN scores.
   - Ineligible players never affect anyone else's percentile.
2. **Normalisation.**
   - Each metric gets a mid-rank percentile `P = (rank − 0.5) / n × 100` within its (season, broad position) group.
   - Metrics where lower is better are ranked in reverse, so 100 is always best.
   - Percentiles are used instead of z-scores because the metrics are skewed and bounded. Ranks also make the units irrelevant.
3. **Low samples.** Each score is shrunk towards 50:
   `score = 50 + (P − 50) · n / (n + k)`
   - `n` is the player's sample (minutes, receptions or transitions).
   - `k` is that metric's `min_sample`.
   - A player whose sample equals `k` moves halfway to 50. A large sample keeps its percentile.
4. **Subscores.** `sub_c = Σ w_m · score_m / Σ w_m`, summed over the metrics in category c that the player has.
5. **Index.** `index = Σ W_c · sub_c / Σ W_c`, summed over the categories the player has.
   - 50 means the middle of their position group. The index is a weighted average of percentiles, not a percentile itself.
6. **Missing values.**
   - A NaN metric drops out of both sums, and the weights are renormalised.
   - `coverage` is the share of the total weight the player actually has.
   - Below `min_coverage` (0.5), the index is NaN.

The formula is a plain weighted sum, so any score can be traced back by hand from `rankings.csv`, which holds every raw value and every metric score. The index does not use PCA. With 40 ranked players, PCA loadings would be unstable, and they would be harder to explain.

## Tuning the weights

Edit `config/composite_index.json`:
- category `weight`;
- metric `weight`;
- `higher_is_better`;
- `sample` / `min_sample`;
- `min_minutes`, `min_coverage` and `group_by`.

All weights default to 1. With the current layout, that means each **category** counts equally, so a metric in a one-metric category (OBSO, scan rate) carries more weight than each of the three defence metrics.

To weight every metric equally instead, set each category's weight to its number of metrics. Negative weights are rejected. Pass a different file with `--config`.

## Results on the synthetic league

- 40 players ranked: 15 defenders, 15 midfielders and 10 forwards.
- Checked against the hidden traits (Spearman, ranked within position, n = 40):

| Score | Tracks |
|---|---|
| Index | 0.54 with the mean of work rate, pressing and awareness |
| Movement | 0.67 with work rate |
| Scanning | 0.88 with awareness |
| Defence | 0.64 with pressing |
| Spacing | none (|ρ| ≤ 0.25 with every trait) |
| Creation | weak (|ρ| ≤ 0.43 with every trait) |

## Limitations and flags

- **Multicollinearity.**
  - `pressure_intensity` and `press_wins_p90` have ρ = 0.75, and lane cuts overlap both (0.73, 0.72), so pressing volume dominates Defence. Lowering those weights in the config is a reasonable fix.
  - Press-and-win enters Defence as pressures won per 90 (`press_wins_p90`), not the win *rate* (`press_win_pct`). The rate has no link to the hidden pressing trait here (ρ −0.41; the count 0.45): in the simulator whether a press ends in a turnover depends on the carrier and the passing lane, not the presser. With the rate in Defence, Defence tracked pressing at 0.31 and the index 0.52; with pressures won per 90, 0.64 and 0.54. The count overlaps pressure intensity (ρ 0.74), so pressing volume now carries three of Defence's four metrics. The rate is still shown in the leaderboards.
  - The script prints every pair of scores with |ρ| > 0.7.
- **Sample size.**
  - There are only 6 matches, and each player has at most about 280 minutes.
  - With groups of 10–15 players, one rank moves a percentile by 7–10 points.
  - Shrinkage reduces this but does not fix it. The index orders players roughly; it is not a precise measurement.
- **Position bias.**
  - Ranking within broad position removes the gap between roles (forwards get more OBSO; defenders cut fewer lanes). It cannot separate a full-back from a centre-back, or a winger from a striker.
  - Once there is enough data, `detailed_position` could replace `broad_position` in `group_by`.
- **Season.** The synthetic league is one season, so the script sets `season = "synthetic"`. Real data with a season column would be grouped by it automatically.
- **Needs clarification:**
  - `def_responsibility_p90` depends on the team: a player behind a weak press concedes more through their zone through no fault of their own.
  - `pitch_control_pct` rewards area, which may partly reflect standing far from everyone rather than finding space.
  - Neither Spacing nor Creation matches any hidden trait. The simulator has no "positioning" trait, so those scores come from role and team effects. Read them as context, not as skill, until they are validated on real data.
- **CFI-style.** The index follows the general idea of a composite index (clear direction for each metric, documented normalisation, transparent weighted sum). It does not copy the formula of any specific published index.
