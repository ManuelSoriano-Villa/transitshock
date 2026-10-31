"""Per-stop features for predicting how much damage removing a stop causes.

Every feature here is cheap compared with the exhaustive simulation
(`simulate.single_removal_damage`), which is the point of the surrogate model.

Run as a script to build the training dataset (features + labels) for a graph:
    python -m transitshock.features data/processed/lbt_20261104.gpickle --date 20261104
"""
import argparse
import resource
import time
from collections import defaultdict
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

from transitshock.build_graph import EARTH_RADIUS_M, ROOT, load_graph
from transitshock.metrics import to_matrix
from transitshock.simulate import single_removal_damage, timetable_removal_damage

FEATURES = [
    # service
    "trips_per_day", "n_routes",
    # local structure
    "transit_in", "transit_out", "walk_degree", "is_terminus",
    # chain structure: can riders get around this stop if it closes?
    "nearest_stop_m", "bypass_gap_m", "stranded_share",
    # global structure
    "betweenness", "harmonic_in", "harmonic_out", "pagerank",
    # geography
    "dist_center_km",
]


# Candidate extra features for the timetable labels (tt_efficiency_loss). Tested in notebook 06:
# they did NOT improve the model, so the final model uses FEATURES only.
TT_FEATURES = ["am_departures", "tt_harmonic_in", "tt_harmonic_out"]


def timetable_features(tt, start: str = "07:00", end: str = "09:00", every_min: int = 15) -> pd.DataFrame:
    """Per-stop features from the timetable, one CSA baseline (about 0.2 s for LBT).

    am_departures:   buses leaving the stop in [start, end), i.e. how often riders get a chance to board
    tt_harmonic_out: average of 1 / door-to-door minutes from this stop to every other (waits included)
    tt_harmonic_in:  the same, to this stop from every other
    """
    from transitshock import csa

    to_s = lambda hhmm: int(hhmm[:2]) * 3600 + int(hhmm[3:5]) * 60
    ev = tt.events
    in_window = ev[(ev["dep_s"] >= to_s(start)) & (ev["dep_s"] < to_s(end))]
    departures = in_window.groupby("stop").size().reindex(range(tt.n_stops), fill_value=0)

    sources, t0s = csa.departure_grid(np.arange(tt.n_stops), start, end, every_min)
    with np.errstate(divide="ignore"):
        inv = 1 / csa.travel_times(tt, sources, t0s)
    inv[np.arange(len(inv)), sources] = np.nan  # a stop's trip to itself doesn't count
    inv = np.nan_to_num(inv, nan=0.0, posinf=0.0)
    per_source = pd.DataFrame(inv).groupby(sources).mean()
    return pd.DataFrame({
        "am_departures": departures.to_numpy(),
        "tt_harmonic_out": per_source.mean(axis=1).to_numpy(),
        "tt_harmonic_in": per_source.mean(axis=0).to_numpy(),
    }, index=pd.Index(tt.stops, name="node"))


def project_m(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Lat/lon -> local x/y meters (equirectangular around the mean latitude)."""
    lat0 = np.radians(np.mean(lat))
    return np.column_stack([
        np.radians(lon) * np.cos(lat0) * EARTH_RADIUS_M,
        np.radians(lat) * EARTH_RADIUS_M,
    ])


def bypass_gap(G: nx.DiGraph, xy: dict) -> dict:
    """Shortest straight-line distance from any stop before this one to any stop after it.

    If a stop closes, a rider on that route would have to walk from the previous
    stop to the next one. A small gap means an easy walk around it; a large gap
    means the closure cuts the route. Stops with no transit predecessor or
    successor (route ends) get 0.
    """
    gaps = {}
    for node in G:
        preds = [p for p in G.predecessors(node) if G.edges[p, node]["kind"] == "transit"]
        succs = [s for s in G.successors(node) if G.edges[node, s]["kind"] == "transit"]
        if not preds or not succs:
            gaps[node] = 0.0
            continue
        # Skip pairs where the bus comes from and returns to the same stop (two-way segments).
        pairs = [(p, s) for p in preds for s in succs if p != s]
        if not pairs:
            gaps[node] = 0.0
            continue
        gaps[node] = float(min(np.hypot(*(xy[p] - xy[s])) for p, s in pairs))
    return gaps


def stranded_share(G: nx.DiGraph, n_roots: int | None = 200, seed: int = 0) -> dict:
    """Estimated share of stop pairs a closure disconnects, beyond the stop's own pairs.

    Uses dominator trees (a classic compiler algorithm). From a starting stop r,
    stop v *dominates* stop x if every path r -> x passes through v. So if v
    closes, a rider starting at r loses exactly the stops v dominates. Summing
    that over every starting stop gives the number of (start, destination) pairs
    the closure disconnects, without running any removals.

    n_roots: how many starting stops to sample (None = all of them, which gives the
    exact value and matches `simulate.single_removal_damage`'s reachable_loss minus
    the 2 / n every removal loses). Sampling 200 is about 10x faster and still
    correlates ~0.9 with the exact value on LBT.
    """
    nodes = list(G)
    n = len(nodes)
    if n_roots is None or n_roots >= n:
        roots = nodes
    else:
        roots = list(np.random.default_rng(seed).choice(nodes, size=n_roots, replace=False))

    stranded = dict.fromkeys(nodes, 0)
    for root in roots:
        idom = nx.immediate_dominators(G, root)  # each reachable stop -> its closest dominator
        children = defaultdict(list)
        for stop, parent in idom.items():
            if stop != root:
                children[parent].append(stop)

        # Breadth-first order, so every stop comes after its dominator...
        order = [root]
        for stop in order:
            order.extend(children[stop])
        # ...then walk it backwards to count each stop's dominator subtree from the leaves up.
        subtree = {}
        for stop in reversed(order):
            subtree[stop] = 1 + sum(subtree[c] for c in children[stop])
        for stop in order[1:]:
            stranded[stop] += subtree[stop] - 1  # stops cut off from root, not counting itself

    return {v: stranded[v] / (len(roots) * (n - 1)) for v in nodes}


def node_features(G: nx.DiGraph) -> pd.DataFrame:
    """One row per node with every column in FEATURES (plus lat/lon/name for plotting).

    Harmonic closeness uses an all-pairs distance matrix, so memory grows with n²;
    fine for LBT (1.9K stops), too big for the full LA graph without sampling.
    """
    nodes, matrix = to_matrix(G)
    n = len(nodes)
    attrs = pd.DataFrame.from_dict(dict(G.nodes(data=True)), orient="index").loc[nodes]
    xy_arr = project_m(attrs["lat"].to_numpy(), attrs["lon"].to_numpy())
    xy = dict(zip(nodes, xy_arr))

    # Harmonic closeness: average of 1 / travel time to (out) or from (in) every other stop.
    dist = dijkstra(matrix, directed=True)
    with np.errstate(divide="ignore"):
        inv = 1 / dist
    np.fill_diagonal(inv, 0)
    inv[~np.isfinite(inv)] = 0

    # Distance to the nearest other stop (k=2 because the nearest point is the stop itself).
    nearest, _ = cKDTree(xy_arr).query(xy_arr, k=2)

    kind_count = lambda edges, kind: sum(1 for *_, k in edges if k == kind)
    df = pd.DataFrame(index=pd.Index(nodes, name="node"))
    df["name"] = attrs["name"]
    df["lat"], df["lon"] = attrs["lat"], attrs["lon"]
    df["trips_per_day"] = attrs["trips_per_day"].astype(float)
    df["n_routes"] = attrs["n_routes"].astype(float)
    df["transit_in"] = [kind_count(G.in_edges(v, data="kind"), "transit") for v in nodes]
    df["transit_out"] = [kind_count(G.out_edges(v, data="kind"), "transit") for v in nodes]
    df["walk_degree"] = [kind_count(G.out_edges(v, data="kind"), "walk") for v in nodes]
    df["is_terminus"] = ((df["transit_in"] == 0) | (df["transit_out"] == 0)).astype(int)
    df["nearest_stop_m"] = nearest[:, 1]
    df["bypass_gap_m"] = pd.Series(bypass_gap(G, xy))
    df["stranded_share"] = pd.Series(stranded_share(G))
    df["betweenness"] = pd.Series(nx.betweenness_centrality(G, weight="weight"))
    df["harmonic_out"] = inv.sum(axis=1) / (n - 1)
    df["harmonic_in"] = inv.sum(axis=0) / (n - 1)
    # PageRank treats weights as link strength, so use the unweighted graph (travel time isn't strength).
    df["pagerank"] = pd.Series(nx.pagerank(G, weight=None))
    center = xy_arr.mean(axis=0)
    df["dist_center_km"] = np.hypot(*(xy_arr - center).T) / 1000
    return df


def cpu_seconds() -> float:
    """CPU time used by this process plus finished child processes (the parallel workers)."""
    me, kids = resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN)
    return me.ru_utime + me.ru_stime + kids.ru_utime + kids.ru_stime


def build_dataset(G: nx.DiGraph, workers: int | None = None, tt=None) -> tuple[pd.DataFrame, dict[str, float]]:
    """Features + exhaustive labels for every stop, and how long each part took.

    Pass `tt` (timetable.build_timetable output) to also compute timetable labels
    (tt_efficiency_loss etc.: single-stop blockage damage with waiting counted).

    Wall time is what you wait; CPU time is the total work, i.e. roughly what it
    would take on a single core.
    """
    wall0, cpu0 = time.perf_counter(), cpu_seconds()
    features = node_features(G)
    wall1, cpu1 = time.perf_counter(), cpu_seconds()
    labels = single_removal_damage(G, workers=workers)
    wall2, cpu2 = time.perf_counter(), cpu_seconds()
    timings = {
        "features_wall_s": wall1 - wall0, "features_cpu_s": cpu1 - cpu0,
        "labels_wall_s": wall2 - wall1, "labels_cpu_s": cpu2 - cpu1,
    }
    dataset = features.join(labels)
    if tt is not None:
        dataset = dataset.join(timetable_features(tt)).join(timetable_removal_damage(tt))
        wall3, cpu3 = time.perf_counter(), cpu_seconds()
        timings.update({"tt_labels_wall_s": wall3 - wall2, "tt_labels_cpu_s": cpu3 - cpu2})
    return dataset, timings


def main() -> None:
    parser = argparse.ArgumentParser(description="Build features + single-stop damage labels for a graph.")
    parser.add_argument("graph", help="path to a .gpickle from build_graph")
    parser.add_argument("--workers", type=int, default=None, help="processes for the simulation (default: all cores)")
    parser.add_argument("--date", help="YYYYMMDD: also compute timetable labels from data/raw/<feed> (feed = graph name prefix)")
    args = parser.parse_args()

    path = Path(args.graph)
    G = load_graph(path)
    tt = None
    if args.date:
        from transitshock.load_gtfs import load_feed
        from transitshock.timetable import build_timetable
        feed = path.stem.split("_")[0]
        tt = build_timetable(load_feed(ROOT / "data" / "raw" / feed, args.date), feed, G)
    dataset, timings = build_dataset(G, workers=args.workers, tt=tt)
    out = path.with_name(path.stem + "_dataset.csv")
    dataset.to_csv(out)
    pd.Series(timings).to_csv(path.with_name(path.stem + "_timings.csv"), header=["seconds"])
    print(f"{len(dataset)} stops")
    print(pd.Series(timings).round(1).to_string())
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
