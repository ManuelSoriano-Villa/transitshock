"""Tests for features, single-stop damage labels, and model evaluation helpers."""
import networkx as nx
import numpy as np
import pandas as pd
import pytest

from transitshock.features import FEATURES, bypass_gap, node_features
from transitshock.model import out_of_fold_predictions, score, spatial_groups, top_k_overlap
from transitshock.simulate import single_removal_damage

# ~0.0009 degrees of latitude is ~100 m.
DEG_100M = 0.0009


@pytest.fixture
def line():
    """A -> B -> C -> D along one route, stops ~300 m apart, plus walk edges A<->W (W is 100 m from A)."""
    G = nx.DiGraph()
    for i, name in enumerate("ABCD"):
        G.add_node(name, lat=33.7 + i * 3 * DEG_100M, lon=-118.1, trips_per_day=10, n_routes=1, name=name)
    G.add_node("W", lat=33.7 - DEG_100M, lon=-118.1, trips_per_day=5, n_routes=1, name="W")
    for u, v in ["AB", "BC", "CD"]:
        G.add_edge(u, v, kind="transit", weight=60, routes="R1")
        G.add_edge(v, u, kind="transit", weight=60, routes="R1")
    G.add_edge("A", "W", kind="walk", weight=77, distance_m=100)
    G.add_edge("W", "A", kind="walk", weight=77, distance_m=100)
    return G


def test_node_features_columns(line):
    df = node_features(line)
    assert set(FEATURES) <= set(df.columns)
    assert not df[FEATURES].isna().any().any()


def test_local_counts(line):
    df = node_features(line)
    assert df.loc["B", "transit_in"] == 2 and df.loc["B", "transit_out"] == 2
    assert df.loc["A", "walk_degree"] == 1
    assert df.loc["W", "is_terminus"] == 1
    assert df.loc["B", "is_terminus"] == 0


def test_nearest_stop_distance(line):
    df = node_features(line)
    assert df.loc["W", "nearest_stop_m"] == pytest.approx(100, abs=2)


def test_bypass_gap_is_distance_between_neighbors(line):
    df = node_features(line)
    # B's neighbors are A and C (~600 m apart); C's predecessor/successor sets include B and D.
    assert df.loc["B", "bypass_gap_m"] == pytest.approx(600, abs=5)
    assert df.loc["W", "bypass_gap_m"] == 0  # no transit edges


def test_middle_stop_has_highest_betweenness(line):
    df = node_features(line)
    assert df["betweenness"].idxmax() in {"A", "B", "C"}
    assert df.loc["D", "betweenness"] == 0


def test_single_removal_damage(line):
    damage = single_removal_damage(line, workers=1)
    n = line.number_of_nodes()
    # Every removal loses at least the stop's own pairs.
    assert (damage["reachable_loss"] >= 2 / n - 1e-9).all()
    # Removing B cuts the line in two; removing the dead-end D only loses its own pairs.
    assert damage.loc["B", "reachable_loss"] > damage.loc["D", "reachable_loss"]
    assert damage.loc["D", "reachable_loss"] == pytest.approx(2 / n)


def test_single_removal_parallel_matches_serial(line):
    serial = single_removal_damage(line, workers=1)
    parallel = single_removal_damage(line, workers=2)
    pd.testing.assert_frame_equal(serial, parallel)


def test_top_k_overlap():
    y = np.arange(100)
    assert top_k_overlap(y, y) == 1.0
    assert top_k_overlap(y, -y) == 0.0


def test_score_perfect_prediction():
    y = np.linspace(0.1, 1, 50)
    s = score(y, y)
    assert s["r2"] == pytest.approx(1) and s["spearman"] == pytest.approx(1)


def test_out_of_fold_predictions_learn_simple_relationship():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.random((200, len(FEATURES))), columns=FEATURES)
    df["lat"], df["lon"] = rng.random(200), rng.random(200)
    df["efficiency_loss"] = np.exp(3 * df["betweenness"]) / 100  # damage driven by one feature
    from transitshock.model import MODELS
    preds = out_of_fold_predictions(df, MODELS["gradient_boosting"])
    assert score(df["efficiency_loss"], preds)["spearman"] > 0.9
    grouped = out_of_fold_predictions(df, MODELS["ridge"], groups=spatial_groups(df, n_groups=5))
    assert len(grouped) == 200
