"""What-if scenarios: close some stops and/or routes, see who is affected.

This is the engine behind the Streamlit demo. It reuses the Week 3 pieces
(`metrics.evaluate`, `simulate.route_edges`) and adds a per-stop breakdown
so the map can color each stop by how its trips changed.
"""
from dataclasses import dataclass

import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse.csgraph import dijkstra

from transitshock.metrics import evaluate, to_matrix
from transitshock.simulate import route_edges

CUT_OFF_SHARE = 0.10  # a stop is "cut off" if it can no longer reach 10%+ of what it could
SLOWER_MIN = 2.0      # a stop is "slower" if its trips take 2+ minutes longer on average


@dataclass
class Baseline:
    """Everything about the intact network that every scenario compares against."""
    nodes: list
    matrix: object
    dist: np.ndarray  # all-pairs travel time in seconds (inf = unreachable)
    metrics: dict


def baseline(G: nx.DiGraph) -> Baseline:
    """Compute once per graph (the app caches this)."""
    nodes, matrix = to_matrix(G)
    return Baseline(nodes, matrix, dijkstra(matrix, directed=True), evaluate(matrix))


def damaged_matrix(G: nx.DiGraph, base: Baseline, routes: list[str] = ()):
    """Adjacency matrix with the given routes' exclusive segments removed (same node order)."""
    if not routes:
        return base.matrix
    exclusive = route_edges(G)
    H = G.copy()
    for route in routes:
        H.remove_edges_from(exclusive.get(route, []))
    return to_matrix(H, base.nodes)[1]


def impact(G: nx.DiGraph, base: Baseline, stops: list = (), routes: list[str] = ()) -> tuple[dict, pd.DataFrame]:
    """Close `stops` (node IDs) and `routes` (route IDs) and measure the result.

    Returns:
        summary: network metrics before/after (same definitions as metrics.evaluate)
        per_stop: one row per stop with
            lost_destinations  stops it could reach before but not now
            added_min          average extra minutes to destinations still reachable
            status             "closed", "cut off", "slower", or "unaffected"
    """
    index = {node: i for i, node in enumerate(base.nodes)}
    removed = np.zeros(len(base.nodes), dtype=bool)
    removed[[index[s] for s in stops]] = True

    matrix = damaged_matrix(G, base, list(routes))
    after = evaluate(matrix, removed=removed)

    # Per-stop view: all-pairs distances in the damaged network (closed stops become unreachable).
    keep = np.flatnonzero(~removed)
    dist = np.full_like(base.dist, np.inf)
    dist[np.ix_(keep, keep)] = dijkstra(matrix[keep][:, keep], directed=True)

    open_dest = ~removed  # destinations that still exist
    could = np.isfinite(base.dist) & open_dest
    can = np.isfinite(dist) & open_dest
    np.fill_diagonal(could, False)
    np.fill_diagonal(can, False)

    lost = (could & ~can).sum(axis=1)
    both = could & can
    delta = np.where(both, dist - base.dist, 0).sum(axis=1)
    added_min = np.divide(delta, both.sum(axis=1), out=np.zeros(len(delta)), where=both.sum(axis=1) > 0) / 60

    per_stop = pd.DataFrame({
        "name": [G.nodes[n].get("name", n) for n in base.nodes],
        "lat": [G.nodes[n]["lat"] for n in base.nodes],
        "lon": [G.nodes[n]["lon"] for n in base.nodes],
        "lost_destinations": lost,
        "added_min": added_min,
    }, index=pd.Index(base.nodes, name="node"))
    lost_share = lost / np.maximum(could.sum(axis=1), 1)
    per_stop["status"] = np.select(
        [removed, lost_share >= CUT_OFF_SHARE, added_min >= SLOWER_MIN],
        ["closed", "cut off", "slower"],
        default="unaffected",
    )

    summary = {
        "efficiency_change": after["efficiency"] / base.metrics["efficiency"] - 1,
        "pairs_lost": base.metrics["reachable_share"] - after["reachable_share"],
        "mean_time_before_min": base.metrics["mean_time_min"],
        "mean_time_after_min": after["mean_time_min"],
        "stops_cut_off": int((per_stop["status"] == "cut off").sum()),
        "stops_slower": int((per_stop["status"] == "slower").sum()),
    }
    return summary, per_stop
