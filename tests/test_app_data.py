"""The CSV export used by the hosted demo must reproduce the graph exactly."""
import networkx as nx

from transitshock.app_data import read_graph_csv, write_graph_csv


def test_graph_csv_round_trip(tmp_path):
    G = nx.DiGraph(route_names={"lbt": {"1": "1 Del Amo Blvd", "01": "01 Leading Zero"}})
    G.add_node("lbt:0002", name="A St", lat=33.123456789012, lon=-118.1, trips_per_day=10, n_routes=2,
               routes="01,1", modes="bus", feed="lbt")
    G.add_node("lbt:0004", name="B St", lat=33.2, lon=-118.2, trips_per_day=3, n_routes=1,
               routes="1", modes="bus", feed="lbt")
    G.add_edge("lbt:0002", "lbt:0004", kind="transit", weight=61.5, trips=12, routes="01,1", modes="bus")
    G.add_edge("lbt:0004", "lbt:0002", kind="walk", weight=135.56337558632057, distance_m=176.2)

    write_graph_csv(G, tmp_path, "t")
    H = read_graph_csv(tmp_path, "t")

    assert dict(H.nodes(data=True)) == dict(G.nodes(data=True))
    assert {(u, v): d for u, v, d in H.edges(data=True)} == {(u, v): d for u, v, d in G.edges(data=True)}
    assert H.graph == G.graph
    assert H.nodes["lbt:0004"]["routes"] == "1"  # stays a string, not the number 1


def test_timetable_round_trip(tmp_path):
    import numpy as np
    import pandas as pd

    from transitshock.app_data import read_timetable, write_timetable
    from transitshock.timetable import timetable_from_events

    G = nx.DiGraph()
    for i, n in enumerate(["lbt:0002", "lbt:0004", "lbt:0006"]):
        G.add_node(n, lat=33.70 + i * 0.001, lon=-118.1)  # ~110 m apart: walkable
    events = pd.DataFrame({
        "trip": [0, 0, 1, 1], "stop": [0, 1, 1, 2], "seq": [1, 2, 1, 2],
        "arr_s": [28800, 28900, 29000, 29100], "dep_s": [28800, 28900, 29000, 29100], "route": ["01", "01", "7", "7"],
    }).astype({"trip": np.int32, "stop": np.int32, "seq": np.int32, "arr_s": np.int32, "dep_s": np.int32})
    tt = timetable_from_events(events, G)

    write_timetable(tt, tmp_path, "t")
    back = read_timetable(tmp_path, "t", G)
    for field in ["dep", "arr", "from_stop", "to_stop", "trip", "foot_ptr", "foot_to", "foot_time"]:
        np.testing.assert_array_equal(getattr(back, field), getattr(tt, field))
    assert back.events["route"].tolist() == ["01", "01", "7", "7"]  # leading zero kept
    assert len(tt.foot_to) > 0
