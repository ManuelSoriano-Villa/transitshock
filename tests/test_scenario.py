"""Tests for what-if scenarios on a small hand-built network."""
import networkx as nx
import pytest

from transitshock.scenario import baseline, impact


def two_way(G, u, v, weight, route, kind="transit"):
    for a, b in [(u, v), (v, u)]:
        G.add_edge(a, b, weight=weight, kind=kind, routes=route)


@pytest.fixture
def network():
    """Line A-B-C-D on route R1 (60 s hops), plus a slow detour B-E-C on route R2 (120 s hops)."""
    G = nx.DiGraph()
    for i, n in enumerate("ABCDE"):
        G.add_node(n, name=f"{n} St", lat=33.7 + i * 0.01, lon=-118.1)
    for u, v in ["AB", "BC", "CD"]:
        two_way(G, u, v, 60, "R1")
    two_way(G, "B", "E", 120, "R2")
    two_way(G, "E", "C", 120, "R2")
    return G


def test_no_change_scenario(network):
    summary, per_stop = impact(network, baseline(network))
    assert summary["efficiency_change"] == pytest.approx(0)
    assert summary["pairs_lost"] == pytest.approx(0)
    assert (per_stop["status"] == "unaffected").all()


def test_closing_end_stop_only_closes_it(network):
    summary, per_stop = impact(network, baseline(network), stops=["D"])
    assert per_stop.loc["D", "status"] == "closed"
    assert per_stop.loc["A", "lost_destinations"] == 0  # D no longer counts as a destination
    assert summary["stops_cut_off"] == 0


def test_closing_cut_vertex_cuts_off_stops(network):
    # C is the only way to D, so closing C cuts D off from everyone.
    summary, per_stop = impact(network, baseline(network), stops=["C"])
    assert per_stop.loc["D", "status"] == "cut off"
    assert per_stop.loc["D", "lost_destinations"] == 3  # A, B, E
    assert summary["pairs_lost"] > 0


def test_route_outage_forces_detour(network):
    # Without R1, B->C must go B-E-C (240 s instead of 60 s), so A and B get slower.
    base = baseline(network)
    summary, per_stop = impact(network, base, routes=["R1"])
    assert per_stop.loc["B", "added_min"] > 0
    assert summary["efficiency_change"] < 0
    # A and D lose their only links (A-B and C-D are R1-only), so they're cut off.
    assert per_stop.loc["A", "status"] == "cut off"


def test_unknown_route_is_ignored(network):
    summary, _ = impact(network, baseline(network), routes=["nope"])
    assert summary["efficiency_change"] == pytest.approx(0)
