"""Network health metrics for a (possibly damaged) stop graph.

Shortest paths run in scipy's compiled Dijkstra rather than networkx, which is
what makes evaluating hundreds of damaged graphs practical.

All metrics are measured against the ORIGINAL node set: a removed stop counts
as unreachable, so removing stops always lowers efficiency and reachability.
"""
import networkx as nx
import numpy as np
from scipy.sparse import csr_array
from scipy.sparse.csgraph import connected_components, dijkstra


def to_matrix(G: nx.DiGraph, nodes: list | None = None, weight: str = "weight") -> tuple[list, csr_array]:
    """Weighted adjacency matrix with a fixed node order (so masks line up across graphs)."""
    nodes = list(G.nodes) if nodes is None else nodes
    return nodes, nx.to_scipy_sparse_array(G, nodelist=nodes, weight=weight, format="csr")


def evaluate(
    matrix: csr_array,
    removed: np.ndarray | None = None,
    sources: np.ndarray | None = None,
) -> dict[str, float]:
    """Health of the graph after removing nodes.

    matrix:  weighted adjacency (seconds), from `to_matrix`
    removed: boolean mask of removed nodes (None = nothing removed)
    sources: node indices to run Dijkstra from (None = all). Sampling sources
             estimates the same quantities faster on large graphs.

    Returns:
        efficiency       mean of 1 / travel_time over all original ordered pairs (unreachable = 0)
        reachable_share  share of original ordered pairs that can still reach each other
        mean_time_min    mean travel time over the pairs that are still reachable
        lscc_share       largest strongly connected component / original node count
    """
    n = matrix.shape[0]
    removed = np.zeros(n, dtype=bool) if removed is None else removed
    sources = np.arange(n) if sources is None else np.asarray(sources)
    pairs = len(sources) * (n - 1)

    keep = np.flatnonzero(~removed)
    sub = matrix[keep][:, keep]
    live_sources = sources[~removed[sources]]
    sub_sources = np.searchsorted(keep, live_sources)

    if len(sub_sources):
        dist = dijkstra(sub, directed=True, indices=sub_sources)
        dist[np.arange(len(sub_sources)), sub_sources] = np.inf  # ignore each source's distance to itself
        reach = dist[np.isfinite(dist)]
    else:
        reach = np.array([])

    _, labels = connected_components(sub, directed=True, connection="strong")
    largest = np.bincount(labels).max() if len(keep) else 0

    return {
        "efficiency": float((1 / reach).sum() / pairs),
        "reachable_share": len(reach) / pairs,
        "mean_time_min": float(reach.mean() / 60) if len(reach) else float("nan"),
        "lscc_share": float(largest / n),
    }
