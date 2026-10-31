"""Failure and attack simulations on the stop graph.

- Node attacks: remove the top-k stops by some ranking (or at random) and
  measure what's left, giving a robustness curve per strategy.
- Route outages: stop running one route and measure the damage.
- Single-stop removal: remove each stop on its own and record the damage
  (the "ground truth" labels the Week 4 model learns to predict).
"""
from concurrent.futures import ProcessPoolExecutor

import networkx as nx
import numpy as np
import pandas as pd

from transitshock.metrics import evaluate, to_matrix

STRATEGIES = ["random", "degree", "trips_per_day", "betweenness"]


def rank_nodes(G: nx.DiGraph, strategy: str, seed: int = 0) -> list:
    """Nodes ordered most-important first (ties broken by node ID for reproducibility)."""
    if strategy == "random":
        nodes = sorted(G.nodes)
        return list(np.random.default_rng(seed).permutation(nodes))
    if strategy == "degree":
        score = dict(G.degree())
    elif strategy == "trips_per_day":
        score = dict(G.nodes(data="trips_per_day"))
    elif strategy == "betweenness":
        score = nx.betweenness_centrality(G, weight="weight")
    else:
        raise ValueError(f"Unknown strategy {strategy!r}; choose from {STRATEGIES}")
    return sorted(G.nodes, key=lambda n: (-score[n], n))


def sample_sources(n: int, max_sources: int | None, seed: int = 0) -> np.ndarray | None:
    """All nodes for small graphs; a fixed random sample of node indices for large ones."""
    if max_sources is None or n <= max_sources:
        return None
    return np.sort(np.random.default_rng(seed).choice(n, size=max_sources, replace=False))


def attack_curve(
    G: nx.DiGraph,
    strategy: str,
    fractions: list[float],
    adaptive: bool = False,
    seed: int = 0,
    max_sources: int | None = 2000,
) -> pd.DataFrame:
    """Remove the top `fraction` of stops by `strategy` and measure the network at each step.

    Static (default): rank once on the intact graph.
    Adaptive: re-rank on the damaged graph before each step, so the next removals
    target whatever has become critical. Slower, and usually more damaging.

    Metrics are reported raw and relative to the intact graph ("*_rel").
    """
    nodes, matrix = to_matrix(G)
    index = {node: i for i, node in enumerate(nodes)}
    n = len(nodes)
    sources = sample_sources(n, max_sources, seed)
    baseline = evaluate(matrix, sources=sources)

    order = rank_nodes(G, strategy, seed)
    damaged = G.copy()
    removed_nodes: list = []
    rows = []
    for fraction in sorted(fractions):
        k = int(round(fraction * n))
        if adaptive:
            while len(removed_nodes) < k:
                # Re-rank on the damaged graph and remove the rest of this step's batch.
                batch = rank_nodes(damaged, strategy, seed)[: k - len(removed_nodes)]
                damaged.remove_nodes_from(batch)
                removed_nodes += batch
        else:
            removed_nodes = order[:k]

        mask = np.zeros(n, dtype=bool)
        mask[[index[v] for v in removed_nodes]] = True
        result = evaluate(matrix, removed=mask, sources=sources)
        rows.append({"strategy": strategy, "adaptive": adaptive, "seed": seed,
                     "fraction": fraction, "removed": k, **result})

    curve = pd.DataFrame(rows)
    for metric in ["efficiency", "reachable_share", "lscc_share"]:
        curve[f"{metric}_rel"] = curve[metric] / baseline[metric]
    return curve


def random_curves(G: nx.DiGraph, fractions: list[float], seeds: range = range(5), **kwargs) -> pd.DataFrame:
    """Several random-failure curves (one per seed) for averaging."""
    return pd.concat([attack_curve(G, "random", fractions, seed=s, **kwargs) for s in seeds], ignore_index=True)


def route_edges(G: nx.DiGraph) -> dict[str, list[tuple]]:
    """Transit edges served ONLY by each route (edges shared with other routes keep running)."""
    exclusive: dict[str, list[tuple]] = {}
    for u, v, d in G.edges(data=True):
        if d["kind"] == "transit" and "," not in d["routes"]:
            exclusive.setdefault(d["routes"], []).append((u, v))
    return exclusive


def route_outages(G: nx.DiGraph, max_sources: int | None = 2000, seed: int = 0) -> pd.DataFrame:
    """Damage from shutting down each route, one at a time.

    Stops stay in the graph (riders can still walk or take other routes);
    only the edges that route alone provides disappear.
    """
    nodes, matrix = to_matrix(G)
    sources = sample_sources(len(nodes), max_sources, seed)
    baseline = evaluate(matrix, sources=sources)

    rows = []
    for route, edges in sorted(route_edges(G).items()):
        damaged = G.copy()
        damaged.remove_edges_from(edges)
        _, damaged_matrix = to_matrix(damaged, nodes)
        result = evaluate(damaged_matrix, sources=sources)
        rows.append({
            "route": route,
            "edges_removed": len(edges),
            "efficiency_rel": result["efficiency"] / baseline["efficiency"],
            "reachable_share": result["reachable_share"],
            "mean_time_min": result["mean_time_min"],
            "added_min": result["mean_time_min"] - baseline["mean_time_min"],
        })
    return pd.DataFrame(rows).sort_values("efficiency_rel").reset_index(drop=True)


# Worker processes get the matrix once (via the initializer) instead of once per task.
_WORKER: dict = {}


def _init_worker(matrix, sources) -> None:
    _WORKER["matrix"], _WORKER["sources"] = matrix, sources


def _remove_one(i: int) -> dict[str, float]:
    matrix = _WORKER["matrix"]
    removed = np.zeros(matrix.shape[0], dtype=bool)
    removed[i] = True
    return evaluate(matrix, removed=removed, sources=_WORKER["sources"])


def single_removal_damage(
    G: nx.DiGraph,
    max_sources: int | None = None,
    workers: int | None = None,
    seed: int = 0,
) -> pd.DataFrame:
    """Damage from removing each stop on its own, one row per node.

    This is the exhaustive (slow) computation: one full network evaluation per stop.
    Runs in parallel across `workers` processes (None = all CPU cores, 1 = no parallelism).

    Columns:
        efficiency_loss  1 - efficiency / intact efficiency
        reachable_loss   share of original stop pairs that can no longer connect
        added_min        change in mean travel time over still-reachable pairs
    Note: removing a stop always loses its own pairs (2 / n of all pairs), so
    reachable_loss is never below 2 / n, and efficiency_loss is never 0.
    """
    nodes, matrix = to_matrix(G)
    sources = sample_sources(len(nodes), max_sources, seed)
    _init_worker(matrix, sources)
    baseline = evaluate(matrix, sources=sources)

    if workers == 1:
        results = [_remove_one(i) for i in range(len(nodes))]
    else:
        with ProcessPoolExecutor(workers, initializer=_init_worker, initargs=(matrix, sources)) as pool:
            results = list(pool.map(_remove_one, range(len(nodes)), chunksize=16))

    df = pd.DataFrame(results, index=pd.Index(nodes, name="node"))
    return pd.DataFrame({
        "efficiency_loss": 1 - df["efficiency"] / baseline["efficiency"],
        "reachable_loss": baseline["reachable_share"] - df["reachable_share"],
        "added_min": df["mean_time_min"] - baseline["mean_time_min"],
    })
