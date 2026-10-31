"""Tests for metrics and simulations on small graphs with known answers."""
import networkx as nx
import numpy as np
import pytest

from transitshock.metrics import evaluate, to_matrix
from transitshock.simulate import attack_curve, rank_nodes, route_edges, route_outages, sample_sources


def two_way(G, u, v, weight, route="R1", kind="transit"):
    for a, b in [(u, v), (v, u)]:
        G.add_edge(a, b, weight=weight, kind=kind, routes=route, trips=10)


@pytest.fixture
def line():
    """A - B - C, both directions, 60 s per hop."""
    G = nx.DiGraph()
    for node, trips in [("A", 10), ("B", 50), ("C", 10)]:
        G.add_node(node, trips_per_day=trips)
    two_way(G, "A", "B", 60)
    two_way(G, "B", "C", 60)
    return G


@pytest.fixture
def hub_and_spokes():
    """Hub H connected to 4 leaves; leaf L1 also walks to L2."""
    G = nx.DiGraph()
    G.add_node("H", trips_per_day=5)
    for leaf in ["L1", "L2", "L3", "L4"]:
        G.add_node(leaf, trips_per_day=100 if leaf == "L1" else 1)
        two_way(G, "H", leaf, 60, route=f"R{leaf[-1]}")
    two_way(G, "L1", "L2", 100, kind="walk")
    return G


def test_intact_line_metrics(line):
    _, M = to_matrix(line)
    m = evaluate(M)
    # Ordered pairs: 4 at 60 s, 2 at 120 s.
    assert m["efficiency"] == pytest.approx((4 / 60 + 2 / 120) / 6)
    assert m["reachable_share"] == 1.0
    assert m["mean_time_min"] == pytest.approx((4 * 1 + 2 * 2) / 6)
    assert m["lscc_share"] == 1.0


def test_removing_middle_disconnects_line(line):
    nodes, M = to_matrix(line)
    removed = np.array([n == "B" for n in nodes])
    m = evaluate(M, removed=removed)
    assert m["reachable_share"] == 0
    assert m["efficiency"] == 0
    assert m["lscc_share"] == pytest.approx(1 / 3)


def test_removing_end_counts_against_original_pairs(line):
    nodes, M = to_matrix(line)
    m = evaluate(M, removed=np.array([n == "C" for n in nodes]))
    assert m["reachable_share"] == pytest.approx(2 / 6)  # only A<->B survive


def test_directed_edges_respected():
    G = nx.DiGraph()
    G.add_edge("A", "B", weight=60)
    _, M = to_matrix(G)
    assert evaluate(M)["reachable_share"] == pytest.approx(1 / 2)


def test_sampled_sources_match_full_on_symmetric_graph(hub_and_spokes):
    _, M = to_matrix(hub_and_spokes)
    full = evaluate(M)
    all_sources = evaluate(M, sources=np.arange(M.shape[0]))
    assert all_sources == pytest.approx(full)


def test_sample_sources():
    assert sample_sources(100, None) is None
    assert sample_sources(100, 500) is None
    s = sample_sources(100, 10, seed=1)
    assert len(s) == 10 and len(set(s)) == 10


def test_rankings(hub_and_spokes):
    assert rank_nodes(hub_and_spokes, "degree")[0] == "H"
    assert rank_nodes(hub_and_spokes, "betweenness")[0] == "H"
    assert rank_nodes(hub_and_spokes, "trips_per_day")[0] == "L1"
    assert sorted(rank_nodes(hub_and_spokes, "random", seed=3)) == sorted(hub_and_spokes)
    with pytest.raises(ValueError):
        rank_nodes(hub_and_spokes, "nope")


def test_attack_curve_starts_at_baseline_and_degrades(hub_and_spokes):
    curve = attack_curve(hub_and_spokes, "degree", [0, 0.2, 0.4])
    assert curve.loc[0, "efficiency_rel"] == pytest.approx(1)
    assert curve["removed"].tolist() == [0, 1, 2]
    assert curve["efficiency_rel"].is_monotonic_decreasing
    # Removing the hub leaves only L1 <-> L2 connected.
    assert curve.loc[1, "reachable_share"] == pytest.approx(2 / 20)


def test_adaptive_matches_static_on_first_step(hub_and_spokes):
    static = attack_curve(hub_and_spokes, "betweenness", [0, 0.2])
    adaptive = attack_curve(hub_and_spokes, "betweenness", [0, 0.2], adaptive=True)
    assert adaptive["efficiency"].tolist() == pytest.approx(static["efficiency"].tolist())


def test_route_edges_only_exclusive(hub_and_spokes):
    G = hub_and_spokes
    G.edges["H", "L3"]["routes"] = "R3,R4"
    exclusive = route_edges(G)
    assert ("H", "L3") not in exclusive.get("R3", [])
    assert set(exclusive["R1"]) == {("H", "L1"), ("L1", "H")}


def test_route_outage_keeps_walk_access(hub_and_spokes):
    outages = route_outages(hub_and_spokes).set_index("route")
    # Without R1, L1 is still reachable on foot via L2; without R3, L3 is cut off.
    assert outages.loc["R1", "reachable_share"] == 1.0
    assert outages.loc["R3", "reachable_share"] < 1.0
    assert outages.loc["R1", "added_min"] > 0
