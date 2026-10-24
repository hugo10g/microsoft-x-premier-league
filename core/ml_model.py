"""
Machine learning on the aggregated off-ball metrics: a market-value model and role clusters.

    train_valuation_model   gradient-boosted trees predict a player's market value from
                            their off-ball work (and, by default, their position, since
                            prices differ by position). Predictions are out-of-fold: each
                            player is priced by a model that never saw them, so "predicted
                            vs actual" is an honest comparison, not memorised prices.
    price_steps             how one player's price is built: a typical player in their
                            position, then each of their off-ball numbers in turn

On the synthetic league, off-ball metrics alone explain almost none of the price
(cross-validated R^2 around 0): value is driven mostly by finishing, passing, speed
and position, which these metrics don't measure. With position the model explains a
little (R^2 ~0.1). The gap between price and off-ball contribution is the point.
    cluster_player_roles    K-means on the standardised metrics groups players by movement
                            signature; each cluster is named after the archetype it leads on
                            (one-to-one, by Hungarian assignment, even when there are fewer
                            clusters than archetypes).

Input is one row per player, e.g. core.pipeline.player_totals passed through
core.pipeline.off_ball_value, filtered to the pool you want to compare. Features
missing from it (or entirely blank, e.g. Scanning before head-orientation data
exists) are left out rather than breaking either model.

The model is scikit-learn's HistGradientBoostingRegressor (histogram-based gradient
boosting, the same algorithm as LightGBM). One engine on every machine means a
laptop and the deployed app always show the same prices.
"""

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

# Feature column -> display name. Transition Latency is "lower is better".
FEATURES = {
    "decoy_p90": "Decoy Run Value /90",
    "lane_cut_p90": "Pressing Lane Cuts /90",
    "median_latency": "Transition Latency",
    "pitch_control_pct": "Pitch Control %",
    "scan_rate": "Scanning rate (scans/s before receiving)",
}
# Short display names (chart axes) for the same features.
SHORT_NAMES = {"Decoy Run Value /90": "Decoy runs", "Pressing Lane Cuts /90": "Lane cuts",
               "Transition Latency": "Transition latency", "Pitch Control %": "Pitch control",
               "Scanning rate (scans/s before receiving)": "Scanning"}
# Role names for a cluster that stands out on each feature (latency: by being *low*).
ROLES = {
    "decoy_p90": "Space creator",
    "lane_cut_p90": "Lane-cutting presser",
    "median_latency": "Transition sprinter",
    "pitch_control_pct": "Space occupier",
    "scan_rate": "Scanner",
}
LOWER_IS_BETTER = {"median_latency"}


ENGINE = "scikit-learn HistGradientBoosting"


def _regressor(seed: int) -> HistGradientBoostingRegressor:
    """Small, regularised trees: there are only a few dozen players to learn from."""
    return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.03, max_leaf_nodes=7,
                                         min_samples_leaf=5, l2_regularization=1.0, random_state=seed)


def present_features(df: pd.DataFrame) -> list[str]:
    """FEATURES columns in `df` with at least one value, in FEATURES order."""
    return [f for f in FEATURES if f in df and df[f].notna().any()]


def design_matrix(df: pd.DataFrame, include_position: bool = True) -> pd.DataFrame:
    """The off-ball features present, plus one-hot broad position when include_position."""
    X = df[present_features(df)].astype(float).rename(columns=FEATURES)
    if include_position and "broad_position" in df:
        X = X.join(pd.get_dummies(df["broad_position"], prefix="Position", dtype=float))
    return X


def train_valuation_model(global_df: pd.DataFrame, target: str = "market_value_m", folds: int = 5,
                          seed: int = 0, include_position: bool = True):
    """Predict market value from off-ball metrics. Returns (model, df with 'Predicted_Value').

    The model learns log(value): prices are heavily right-skewed, so

        ln(price) = c + Σ_t learning_rate · tree_t(x),   price = e^ln(price)

    with x the off-ball features and one-hot position. 'Predicted_Value' (€m) is
    out-of-fold and 'Price_Fold' says which fold's model priced each player. The returned
    model is refitted on every player, for scoring new ones. Cross-validated diagnostics
    are in df.attrs["valuation"]: engine, features, r2 (in log space, against always
    predicting the average), mae_m (€m), the feature importances, the fold models and
    the tree settings (trees, learning_rate, leaves).
    """
    df = global_df.copy()
    X = design_matrix(df, include_position)
    y = np.log(df[target].astype(float))
    oof, fold_of, fold_models = np.full(len(df), np.nan), np.zeros(len(df), dtype=int), []
    splits = KFold(n_splits=min(folds, len(df)), shuffle=True, random_state=seed)
    for fold, (train, test) in enumerate(splits.split(X)):
        model = _regressor(seed)
        model.fit(X.iloc[train], y.iloc[train])
        oof[test], fold_of[test] = model.predict(X.iloc[test]), fold
        fold_models.append(model)
    model = _regressor(seed)
    model.fit(X, y)

    df["Predicted_Value"] = np.round(np.exp(oof), 1)
    df["Price_Fold"] = fold_of
    importances = getattr(model, "feature_importances_", None)
    df.attrs["valuation"] = {
        "engine": ENGINE,
        "r2": float(r2_score(y, oof)),
        "mae_m": float(mean_absolute_error(df[target], df["Predicted_Value"])),
        "features": list(X.columns),
        "importance": dict(zip(X.columns, importances.tolist())) if importances is not None else None,
        "fold_models": fold_models, "trees": int(model.n_iter_), "learning_rate": model.learning_rate,
        "leaves": model.max_leaf_nodes,
    }
    return model, df


def price_steps(model, X: pd.DataFrame, row: int) -> list[tuple[str, float]]:
    """How `model` prices player `row` of the design matrix X, as (label, price €m) steps.

    The first step prices a typical player in the same position (every off-ball feature
    at the pool's median); each next one switches one feature to the player's own value,
    in X's column order, and the last is the player's price. The steps add up exactly,
    but the share each feature gets depends on that order: trees combine features.
    """
    player = X.iloc[[row]]
    position = [c for c in X.columns if c.startswith("Position_")]
    current = player.copy()
    off_ball = [c for c in X.columns if c not in position]
    current[off_ball] = X[off_ball].median().to_numpy()
    price = lambda x: float(np.exp(model.predict(x))[0])
    steps = [("Typical " + next((c.removeprefix("Position_") for c in position if player[c].iloc[0] == 1), "player"),
              price(current))]
    for column in off_ball:
        current[column] = player[column].to_numpy()
        steps.append((column, price(current)))
    return steps


def role_names(centres: pd.DataFrame) -> dict[int, str]:
    """Cluster id -> archetype name, one-to-one, from standardised cluster centres.

    `centres` has one row per cluster and one column per feature (lower-is-better
    already flipped). The rectangular Hungarian assignment gives each cluster the
    archetype it leads on most, with no name used twice: with fewer clusters than
    archetypes some archetypes go unused; with more, the extra clusters are named
    after their strongest feature plus a number ("Scanner 6").
    """
    roles = [f for f in centres.columns if f in ROLES]
    rows, cols = linear_sum_assignment(-centres[roles].to_numpy())
    label = {int(centres.index[r]): ROLES[roles[c]] for r, c in zip(rows, cols)}
    for r, cluster in enumerate(centres.index):
        if int(cluster) not in label:
            label[int(cluster)] = f"{ROLES[centres.loc[cluster, roles].idxmax()]} {r + 1}"
    return label


def cluster_player_roles(global_df: pd.DataFrame, k: int = 4, seed: int = 0) -> pd.DataFrame:
    """Group players by off-ball signature. Adds 'Cluster_ID' and a readable 'Cluster_Role'.

    Missing values (no sprint, so no Transition Latency) are filled with the median
    before standardising. Clusters are named by `role_names`: each after the feature
    it leads the other clusters on most (for latency, by being fastest), assigned
    one-to-one so no two clusters share a name.
    """
    df = global_df.copy()
    features = present_features(df)
    X = df[features].astype(float)
    X = X.fillna(X.median())
    scaled = StandardScaler().fit_transform(X)
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(scaled)
    df["Cluster_ID"] = kmeans.labels_

    centres = pd.DataFrame(kmeans.cluster_centers_, columns=features)
    for feature in LOWER_IS_BETTER & set(features):
        centres[feature] = -centres[feature]
    df["Cluster_Role"] = df["Cluster_ID"].map(role_names(centres))
    df.attrs["clusters"] = (X.assign(Cluster_Role=df["Cluster_Role"]).groupby("Cluster_Role").mean()
                            .rename(columns=FEATURES).round(3))
    return df
