"""Build a weighted, directed stop graph from one or more GTFS feeds.

Nodes are served stops (rail platforms merged into their parent station).
Edges are either:
- transit: consecutive stops on any trip, weighted by median scheduled travel time
- walk: stops within a short walking distance, weighted by walking time

Node IDs are prefixed with the feed name ("lbt:0002") so feeds can be combined
without stop_id collisions.
"""
import argparse
import pickle
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from transitshock.load_gtfs import load_feed

ROOT = Path(__file__).resolve().parents[2]
EARTH_RADIUS_M = 6_371_000
MIN_TRAVEL_S = 15  # floor for 0-second segments between distinct stops
WALK_RADIUS_M = 200
WALK_SPEED_M_S = 1.3  # about 4.7 km/h
RAIL_ROUTE_TYPES = {"0", "1", "2"}


def node_ids(stops: pd.DataFrame, feed: str) -> pd.Series:
    """Map stop_id -> node ID, collapsing platforms into their parent station."""
    is_platform = (stops["location_type"] == 0) & (stops["parent_station"] != "")
    node_stop = stops["parent_station"].where(is_platform, stops["stop_id"])
    return pd.Series((feed + ":" + node_stop).values, index=stops["stop_id"].values)


def trip_segments(data: dict[str, pd.DataFrame], feed: str) -> pd.DataFrame:
    """One row per consecutive stop pair on every trip: u, v, travel_s, route_id, mode."""
    to_node = node_ids(data["stops"], feed)
    st = data["stop_times"].assign(node=lambda df: df["stop_id"].map(to_node))
    nxt = st.groupby("trip_id")[["node", "arrival_s"]].shift(-1)

    seg = pd.DataFrame({
        "trip_id": st["trip_id"],
        "u": st["node"],
        "v": nxt["node"],
        "travel_s": nxt["arrival_s"] - st["departure_s"],
    }).dropna(subset=["v"])
    seg = seg[seg["u"] != seg["v"]]  # platforms of one station merged into a single node

    route_mode = data["routes"].assign(
        mode=lambda df: np.where(df["route_type"].isin(RAIL_ROUTE_TYPES), "rail", "bus")
    )[["route_id", "mode"]]
    return seg.merge(data["trips"][["trip_id", "route_id"]], on="trip_id").merge(route_mode, on="route_id")


def transit_edges(segments: pd.DataFrame, min_travel_s: float = MIN_TRAVEL_S) -> pd.DataFrame:
    """Collapse segments to one edge per (u, v) with median travel time and service counts."""
    join_unique = lambda s: ",".join(sorted(set(s)))
    edges = segments.groupby(["u", "v"]).agg(
        weight=("travel_s", "median"),
        trips=("trip_id", "size"),
        routes=("route_id", join_unique),
        modes=("mode", join_unique),
    ).reset_index()
    edges["weight"] = edges["weight"].clip(lower=min_travel_s)
    return edges


def node_table(data: dict[str, pd.DataFrame], segments: pd.DataFrame, feed: str) -> pd.DataFrame:
    """Attributes for every served node: name, coordinates, trips per day, routes, modes."""
    stops = data["stops"].assign(node=lambda df: df["stop_id"].map(node_ids(df, feed)))
    # Prefer the node's own row (the station) for name/coords; fall back to a platform.
    stops["is_node_row"] = (feed + ":" + stops["stop_id"]) == stops["node"]
    info = stops.sort_values("is_node_row", ascending=False).drop_duplicates("node").set_index("node")

    visits = data["stop_times"]["stop_id"].map(node_ids(data["stops"], feed)).value_counts()
    join_unique = lambda s: ",".join(sorted(set(s)))
    served = pd.concat([
        segments[["u", "route_id", "mode"]].rename(columns={"u": "node"}),
        segments[["v", "route_id", "mode"]].rename(columns={"v": "node"}),
    ]).groupby("node").agg(routes=("route_id", join_unique), modes=("mode", join_unique))

    nodes = served.join(info[["stop_name", "stop_lat", "stop_lon"]]).join(visits.rename("trips_per_day"))
    nodes["n_routes"] = nodes["routes"].str.count(",") + 1
    nodes["feed"] = feed
    return nodes.rename(columns={"stop_name": "name", "stop_lat": "lat", "stop_lon": "lon"})


def route_names(routes: pd.DataFrame) -> dict[str, str]:
    """route_id -> display name like "1 Del Amo Blvd" (falls back to whichever name exists)."""
    short = routes.get("route_short_name", pd.Series("", index=routes.index)).fillna("")
    long = routes.get("route_long_name", pd.Series("", index=routes.index)).fillna("")
    names = (short + " " + long).str.strip()
    return dict(zip(routes["route_id"], names.where(names != "", routes["route_id"])))


def walking_pairs(nodes: pd.DataFrame, radius_m: float = WALK_RADIUS_M) -> pd.DataFrame:
    """All ordered node pairs within `radius_m`, using a grid so we never compare all pairs.

    Coordinates are projected to local meters (equirectangular), which is accurate
    to well under a meter at these distances.
    """
    lat0 = np.radians(nodes["lat"].mean())
    pts = pd.DataFrame({
        "node": nodes.index,
        "x": np.radians(nodes["lon"].to_numpy()) * np.cos(lat0) * EARTH_RADIUS_M,
        "y": np.radians(nodes["lat"].to_numpy()) * EARTH_RADIUS_M,
    })
    pts["cx"] = np.floor(pts["x"] / radius_m).astype(int)
    pts["cy"] = np.floor(pts["y"] / radius_m).astype(int)

    # Any pair within radius_m lies in the same or an adjacent grid cell.
    pairs = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            shifted = pts.assign(cx=pts["cx"] + dx, cy=pts["cy"] + dy)
            m = shifted.merge(pts, on=["cx", "cy"], suffixes=("_u", "_v"))
            m = m[m["node_u"] != m["node_v"]]
            m["distance_m"] = np.hypot(m["x_u"] - m["x_v"], m["y_u"] - m["y_v"])
            pairs.append(m.loc[m["distance_m"] <= radius_m, ["node_u", "node_v", "distance_m"]])
    return pd.concat(pairs, ignore_index=True).rename(columns={"node_u": "u", "node_v": "v"})


def build_graph(
    feeds: dict[str, dict[str, pd.DataFrame]],
    walk_radius_m: float = WALK_RADIUS_M,
    walk_speed_m_s: float = WALK_SPEED_M_S,
    min_travel_s: float = MIN_TRAVEL_S,
) -> nx.DiGraph:
    """Build one directed graph from {feed_name: load_feed(...) output}."""
    G = nx.DiGraph(walk_radius_m=walk_radius_m, walk_speed_m_s=walk_speed_m_s, min_travel_s=min_travel_s)

    for feed, data in feeds.items():
        G.graph.setdefault("route_names", {})[feed] = route_names(data["routes"])
        segments = trip_segments(data, feed)
        for node, attrs in node_table(data, segments, feed).iterrows():
            G.add_node(node, **attrs.to_dict())
        for e in transit_edges(segments, min_travel_s).itertuples(index=False):
            G.add_edge(e.u, e.v, kind="transit", weight=float(e.weight),
                       trips=int(e.trips), routes=e.routes, modes=e.modes)

    # Walking transfers across all feeds; transit edges take priority where both exist.
    nodes = pd.DataFrame.from_dict(dict(G.nodes(data=True)), orient="index")
    for e in walking_pairs(nodes, walk_radius_m).itertuples(index=False):
        if not G.has_edge(e.u, e.v):
            G.add_edge(e.u, e.v, kind="walk", weight=e.distance_m / walk_speed_m_s,
                       distance_m=float(e.distance_m))
    return G


def graph_summary(G: nx.DiGraph) -> pd.Series:
    kinds = pd.Series([k for _, _, k in G.edges(data="kind")]).value_counts()
    weak = max(nx.weakly_connected_components(G), key=len)
    strong = max(nx.strongly_connected_components(G), key=len)
    return pd.Series({
        "nodes": G.number_of_nodes(),
        "transit edges": int(kinds.get("transit", 0)),
        "walk edges": int(kinds.get("walk", 0)),
        "weakly connected components": nx.number_weakly_connected_components(G),
        "largest weak component share": round(len(weak) / len(G), 3),
        "strongly connected components": nx.number_strongly_connected_components(G),
        "largest strong component share": round(len(strong) / len(G), 3),
        "mean out-degree": round(sum(d for _, d in G.out_degree()) / len(G), 2),
    }, dtype=object)


def save_graph(G: nx.DiGraph, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(G, f)


def load_graph(path: str | Path) -> nx.DiGraph:
    with open(path, "rb") as f:
        return pickle.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a stop graph from GTFS feeds in data/raw/.")
    parser.add_argument("feeds", nargs="+", help="feed folder names, e.g. lbt metro_bus")
    parser.add_argument("--date", required=True, help="service date, YYYYMMDD")
    parser.add_argument("--out", help="output path (default: data/processed/<feeds>_<date>.gpickle)")
    args = parser.parse_args()

    feeds = {name: load_feed(ROOT / "data" / "raw" / name, args.date) for name in args.feeds}
    G = build_graph(feeds)
    out = Path(args.out) if args.out else ROOT / "data" / "processed" / f"{'+'.join(args.feeds)}_{args.date}.gpickle"
    save_graph(G, out)
    print(graph_summary(G).to_string())
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
