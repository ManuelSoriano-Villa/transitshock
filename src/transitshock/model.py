"""Train and evaluate surrogate models that predict single-stop removal damage.

The question: can cheap features (features.py) stand in for the exhaustive
simulation (simulate.single_removal_damage) when ranking which stops matter most?
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from transitshock.features import FEATURES

TARGET = "efficiency_loss"
TOP_FRACTION = 0.05  # "critical" = the top 5% most damaging stops

MODELS = {
    "ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
    "random_forest": RandomForestRegressor(n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=0),
    "gradient_boosting": HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=0),
}


def top_k_overlap(y_true: np.ndarray, y_pred: np.ndarray, fraction: float = TOP_FRACTION) -> float:
    """Share of the truly most-damaging stops that the prediction also puts in its top group."""
    k = max(1, int(round(fraction * len(y_true))))
    true_top = set(np.argsort(-np.asarray(y_true))[:k])
    pred_top = set(np.argsort(-np.asarray(y_pred))[:k])
    return len(true_top & pred_top) / k


def score(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """R² (exact values), Spearman ρ (ranking), and top-5% overlap (finding the critical stops)."""
    return {
        "r2": r2_score(y_true, y_pred),
        "spearman": spearmanr(y_true, y_pred).statistic,
        "top5_overlap": top_k_overlap(y_true, y_pred),
    }


def spatial_groups(df: pd.DataFrame, n_groups: int = 10, seed: int = 0) -> np.ndarray:
    """Cluster stops by location so cross-validation can hold out whole neighborhoods.

    Neighboring stops have nearly identical features and damage. With a random split,
    a stop's neighbor in the training set makes the test look easier than it is.
    """
    return KMeans(n_clusters=n_groups, n_init=10, random_state=seed).fit_predict(df[["lat", "lon"]])


def out_of_fold_predictions(
    df: pd.DataFrame,
    model,
    target: str = TARGET,
    groups: np.ndarray | None = None,
    n_splits: int = 5,
    log_target: bool = True,
    features: list[str] = FEATURES,
) -> np.ndarray:
    """Predict every stop with a model that never saw it in training (k-fold CV).

    log_target: train on log(damage). Damage is heavily skewed (a few stops matter
    a lot), and the log keeps those few from dominating the fit.
    features: which columns to train on (pass a subset to test a feature's value).
    """
    X, y = df[features].to_numpy(), df[target].to_numpy()
    splitter = GroupKFold(n_splits) if groups is not None else KFold(n_splits, shuffle=True, random_state=0)
    preds = np.empty(len(df))
    for train, test in splitter.split(X, y, groups):
        m = clone(model)
        y_train = np.log(y[train]) if log_target else y[train]
        m.fit(X[train], y_train)
        preds[test] = np.exp(m.predict(X[test])) if log_target else m.predict(X[test])
    return preds


def compare(df: pd.DataFrame, target: str = TARGET, groups: np.ndarray | None = None) -> pd.DataFrame:
    """Score single-feature heuristics and each trained model on the same stops.

    Heuristics (raw betweenness, trips/day, ...) only produce a ranking, so their
    R² is left blank.
    """
    y = df[target].to_numpy()
    rows = {}
    for feature in ["betweenness", "trips_per_day", "n_routes", "harmonic_in", "bypass_gap_m", "stranded_share"]:
        s = score(y, df[feature].to_numpy())
        rows[f"heuristic: {feature}"] = {**s, "r2": np.nan}
    for name, model in MODELS.items():
        rows[f"model: {name}"] = score(y, out_of_fold_predictions(df, model, target, groups))
    return pd.DataFrame(rows).T


def fit(df: pd.DataFrame, model=None, target: str = TARGET):
    """Fit on all stops (log target) for final use; predict with np.exp(model.predict(X))."""
    model = clone(model if model is not None else MODELS["random_forest"])
    return model.fit(df[FEATURES].to_numpy(), np.log(df[target].to_numpy()))
