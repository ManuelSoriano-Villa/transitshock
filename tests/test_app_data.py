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
