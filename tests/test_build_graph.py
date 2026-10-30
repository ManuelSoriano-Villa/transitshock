"""Tests for build_graph using small hand-built feeds."""
import numpy as np
import pandas as pd
import pytest

from transitshock.build_graph import build_graph, graph_summary, load_graph, save_graph, walking_pairs
from transitshock.load_gtfs import clean_stop_times, clean_stops

# ~0.0009 degrees of latitude is ~100 m.
DEG_100M = 0.0009


def make_feed(stops, routes, trips, stop_times):
    """Build a load_feed-style dict from lists of tuples."""
    stops = pd.DataFrame(stops, columns=["stop_id", "stop_name", "stop_lat", "stop_lon", "location_type", "parent_station"])
    return {
        "stops": clean_stops(stops.astype(str)),
        "routes": pd.DataFrame(routes, columns=["route_id", "route_type"]),
        "trips": pd.DataFrame(trips, columns=["trip_id", "route_id"]),
        "stop_times": clean_stop_times(pd.DataFrame(
            [(t, a, a, s, q, "1", "") for t, s, q, a in stop_times],
            columns=["trip_id", "arrival_time", "departure_time", "stop_id", "stop_sequence",
                     "timepoint", "shape_dist_traveled"],
        )),
    }


@pytest.fixture
def bus_feed():
    # A -> B -> C along a line ~1 km apart, plus D ~100 m from C (walkable, never served).
    # E is ~100 m from A and served by a second route, so A <-> E gets walking edges.
    stops = [
        ("A", "A St", 33.70, -118.10, "0", ""),
        ("B", "B St", 33.71, -118.10, "0", ""),
        ("C", "C St", 33.72, -118.10, "0", ""),
        ("D", "D St", 33.72 + DEG_100M, -118.10, "0", ""),
        ("E", "E St", 33.70 + DEG_100M, -118.10, "0", ""),
        ("F", "F St", 33.69, -118.10, "0", ""),
    ]
    routes = [("R1", "3"), ("R2", "3")]
    trips = [("t1", "R1"), ("t2", "R1"), ("t3", "R1"), ("t4", "R2")]
    stop_times = [
        # trip, stop, seq, arrival
        ("t1", "A", 1, "08:00:00"), ("t1", "B", 2, "08:02:00"), ("t1", "C", 3, "08:05:00"),
        ("t2", "A", 1, "09:00:00"), ("t2", "B", 2, "09:04:00"), ("t2", "C", 3, "09:05:00"),
        ("t3", "A", 1, "10:00:00"), ("t3", "B", 2, "10:03:00"), ("t3", "C", 3, "10:03:00"),
        ("t4", "E", 1, "08:00:00"), ("t4", "F", 2, "08:03:00"),
    ]
    return make_feed(stops, routes, trips, stop_times)


@pytest.fixture
def rail_feed():
    # Station S1 has two platforms (P1a, P1b); station S2 has one (P2).
    stops = [
        ("S1", "One Station", 34.00, -118.20, "1", ""),
        ("P1a", "One Station", 34.00, -118.20, "0", "S1"),
        ("P1b", "One Station", 34.00, -118.20, "0", "S1"),
        ("S2", "Two Station", 34.02, -118.20, "1", ""),
        ("P2", "Two Station", 34.02, -118.20, "0", "S2"),
        ("X1", "One Station Entrance", 34.00, -118.20, "2", "S1"),
    ]
    routes = [("L", "0")]
    trips = [("r1", "L")]
    stop_times = [("r1", "P1a", 1, "07:00:00"), ("r1", "P1b", 2, "07:00:30"), ("r1", "P2", 3, "07:04:00")]
    return make_feed(stops, routes, trips, stop_times)


def test_transit_edges_use_median_travel_time(bus_feed):
    G = build_graph({"bus": bus_feed})
    # A->B took 120 s, 240 s, 180 s; median is 180 s.
    assert G.edges["bus:A", "bus:B"]["weight"] == 180
    assert G.edges["bus:A", "bus:B"]["trips"] == 3
    assert G.edges["bus:A", "bus:B"]["kind"] == "transit"


def test_transit_edges_are_directed(bus_feed):
    G = build_graph({"bus": bus_feed})
    assert G.has_edge("bus:A", "bus:B")
    assert not G.has_edge("bus:B", "bus:A")


def test_minimum_travel_time_floor(bus_feed):
    # B->C took 180 s, 60 s, and 0 s (same scheduled minute); median is 60 s.
    assert build_graph({"bus": bus_feed}, min_travel_s=15).edges["bus:B", "bus:C"]["weight"] == 60
    assert build_graph({"bus": bus_feed}, min_travel_s=90).edges["bus:B", "bus:C"]["weight"] == 90


def test_unserved_stops_are_not_nodes(bus_feed):
    G = build_graph({"bus": bus_feed})
    assert "bus:D" not in G


def test_walking_edges_connect_nearby_served_stops(bus_feed):
    G = build_graph({"bus": bus_feed}, walk_radius_m=200, walk_speed_m_s=1.0)
    edge = G.edges["bus:A", "bus:E"]
    assert edge["kind"] == "walk"
    assert edge["distance_m"] == pytest.approx(100, abs=2)
    assert edge["weight"] == pytest.approx(edge["distance_m"])  # 1 m/s
    assert G.has_edge("bus:E", "bus:A")


def test_no_walking_edges_beyond_radius(bus_feed):
    G = build_graph({"bus": bus_feed}, walk_radius_m=50)
    assert not G.has_edge("bus:A", "bus:E")


def test_walking_pairs_matches_brute_force():
    # 300 deterministic pseudo-random points in a ~2 km box.
    i = np.arange(300)
    nodes = pd.DataFrame({
        "lat": 33.7 + (i * 7919 % 1000) / 1000 * 0.02,
        "lon": -118.1 + (i * 104729 % 1000) / 1000 * 0.02,
    }, index=[f"n{k}" for k in i])
    fast = walking_pairs(nodes, radius_m=150)

    lat0 = nodes["lat"].mean()
    x = np.radians(nodes["lon"]) * np.cos(np.radians(lat0)) * 6_371_000
    y = np.radians(nodes["lat"]) * 6_371_000
    d = np.hypot(x.values[:, None] - x.values[None, :], y.values[:, None] - y.values[None, :])
    i, j = np.where((d <= 150) & ~np.eye(len(nodes), dtype=bool))
    assert len(fast) == len(i)
    assert set(zip(fast["u"], fast["v"])) == set(zip(nodes.index[i], nodes.index[j]))


def test_node_attributes(bus_feed):
    G = build_graph({"bus": bus_feed})
    a = G.nodes["bus:A"]
    assert a["name"] == "A St"
    assert a["trips_per_day"] == 3
    assert a["n_routes"] == 1
    assert a["modes"] == "bus"


def test_platforms_merge_into_station(rail_feed):
    G = build_graph({"rail": rail_feed})
    assert set(G.nodes) == {"rail:S1", "rail:S2"}  # no platforms, no entrance
    assert G.has_edge("rail:S1", "rail:S2")
    assert not G.has_edge("rail:S1", "rail:S1")  # P1a -> P1b collapses to a self-loop, dropped
    assert G.nodes["rail:S1"]["name"] == "One Station"
    assert G.nodes["rail:S1"]["modes"] == "rail"


def test_multiple_feeds_combine_without_collisions(bus_feed, rail_feed):
    G = build_graph({"bus": bus_feed, "rail": rail_feed})
    assert {n.split(":")[0] for n in G} == {"bus", "rail"}


def test_summary_and_round_trip(tmp_path, bus_feed):
    G = build_graph({"bus": bus_feed})
    summary = graph_summary(G)
    assert summary["nodes"] == 5
    path = tmp_path / "g.gpickle"
    save_graph(G, path)
    assert graph_summary(load_graph(path)).equals(summary)
