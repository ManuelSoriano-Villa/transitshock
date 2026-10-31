"""What-if scenarios: close some stops and/or routes, see who is affected.

This is the engine behind the Streamlit demo. Two versions:
- `impact`: on the static graph (median travel times, no waiting). Reuses the
  Week 3 pieces (`metrics.evaluate`, `simulate.route_edges`).
- `timetable_impact`: on the timetable with the Connection Scan Algorithm, so
  waiting for buses and transfers counts, and frequency cuts and delays can be
  modeled too.
Both add a per-stop breakdown so the map can color each stop by how its trips changed.
"""
from dataclasses import dataclass

import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse.csgraph import dijkstra

from transitshock import csa
from transitshock.metrics import evaluate, to_matrix
from transitshock.simulate import route_edges
from transitshock.timetable import Timetable, disrupt

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
    per_stop["status"] = stop_status(removed, lost_share, added_min)

    summary = {
        "efficiency_change": after["efficiency"] / base.metrics["efficiency"] - 1,
        "pairs_lost": base.metrics["reachable_share"] - after["reachable_share"],
        "mean_time_before_min": base.metrics["mean_time_min"],
        "mean_time_after_min": after["mean_time_min"],
        "stops_cut_off": int((per_stop["status"] == "cut off").sum()),
        "stops_slower": int((per_stop["status"] == "slower").sum()),
    }
    return summary, per_stop


def stop_status(removed: np.ndarray, lost_share: np.ndarray, added_min: np.ndarray) -> np.ndarray:
    return np.select(
        [removed, lost_share >= CUT_OFF_SHARE, added_min >= SLOWER_MIN],
        ["closed", "cut off", "slower"],
        default="unaffected",
    )


@dataclass
class TimetableBaseline:
    """The intact timetable's door-to-door times for one departure window."""
    tt: Timetable
    sources: np.ndarray  # one entry per query (every stop x every departure time)
    t0s: np.ndarray
    minutes: np.ndarray  # (queries, stops), NaN = unreachable
    metrics: dict
    window_s: tuple[int, int] = (0, 0)  # [start, end) in seconds after midnight


def timetable_baseline(tt: Timetable, start: str = "07:00", end: str = "09:00", every_min: int = 15) -> TimetableBaseline:
    """Route from every stop at every departure time in [start, end). Compute once per window."""
    sources, t0s = csa.departure_grid(np.arange(tt.n_stops), start, end, every_min)
    minutes = csa.travel_times(tt, sources, t0s)
    to_s = lambda hhmm: int(hhmm[:2]) * 3600 + int(hhmm[3:5]) * 60
    return TimetableBaseline(tt, sources, t0s, minutes, csa.summarize(minutes, sources, tt.n_stops),
                             window_s=(to_s(start), to_s(end)))


def timetable_impact(
    base: TimetableBaseline,
    stops: list = (),
    canceled_routes: list[str] = (),
    reduced_routes: list[str] = (),
    delays_min: dict[str, float] | None = None,
    blocked_stops: list = (),
) -> tuple[dict, pd.DataFrame]:
    """Like `impact`, but on the timetable: waits and transfers count.

    stops are *closed* (buses skip them); blocked_stops can't be passed at all
    (the static graph's notion of removing a stop). Per-stop numbers are averaged
    over the window's departure times.
    """
    tt = base.tt
    disrupted = disrupt(tt, stops, canceled_routes, reduced_routes, delays_min, blocked_stops)
    after = csa.travel_times(disrupted, base.sources, base.t0s)

    removed = np.zeros(tt.n_stops, dtype=bool)
    removed[[tt.stops.index(s) for s in [*stops, *blocked_stops]]] = True
    summary_after = csa.summarize(after, base.sources, tt.n_stops, removed=removed)

    open_dest = ~removed
    could = np.isfinite(base.minutes) & open_dest
    can = np.isfinite(after) & open_dest
    q = np.arange(len(base.sources))
    could[q, base.sources] = can[q, base.sources] = False

    per_query = pd.DataFrame({
        "source": base.sources,
        "lost": (could & ~can).sum(axis=1),
        "could": could.sum(axis=1),
        "added_sum": np.where(could & can, after - base.minutes, 0).sum(axis=1),
        "both": (could & can).sum(axis=1),
    })
    agg = per_query.groupby("source").sum()
    added_min = (agg["added_sum"] / agg["both"].where(agg["both"] > 0)).fillna(0).to_numpy()
    lost_share = (agg["lost"] / agg["could"].clip(lower=1)).to_numpy()
    n_times = len(base.sources) // tt.n_stops

    per_stop = pd.DataFrame({
        "lost_destinations": (agg["lost"] / n_times).round(1).to_numpy(),
        "added_min": added_min,
        "status": stop_status(removed, lost_share, added_min),
    }, index=pd.Index(tt.stops, name="node"))

    summary = {
        "efficiency_change": summary_after["efficiency"] / base.metrics["efficiency"] - 1,
        "pairs_lost": base.metrics["reachable_share"] - summary_after["reachable_share"],
        "mean_time_before_min": base.metrics["mean_time_min"],
        "mean_time_after_min": summary_after["mean_time_min"],
        "stops_cut_off": int((per_stop["status"] == "cut off").sum()),
        "stops_slower": int((per_stop["status"] == "slower").sum()),
    }
    return summary, per_stop


def route_impacts(
    base: TimetableBaseline,
    per_stop: pd.DataFrame,
    stops: list = (),
    canceled_routes: list[str] = (),
    reduced_routes: list[str] = (),
    delays_min: dict[str, float] | None = None,
    blocked_stops: list = (),
) -> pd.DataFrame:
    """How a scenario hits each route: its stops' status (from timetable_impact) and its service in the window.

    One row per route:
        stops, closed, cut_off, slower   counts of the route's stops by status
        added_min                        average extra minutes for trips starting at the route's open stops
        trips_before, trips_after        buses on the route that run during the window
        delay_min                        minutes late (if the scenario delays it)
        change                           "canceled", "halved", "late", "affected", or "normal"
    """
    tt = base.tt
    disrupted = disrupt(tt, stops, canceled_routes, reduced_routes, delays_min, blocked_stops)
    start, end = base.window_s

    def trips_in_window(events: pd.DataFrame) -> pd.Series:
        bus = events["bus"] if "bus" in events else events["trip"]  # a blocked trip's pieces are one bus
        running = events[(events["dep_s"] >= start) & (events["dep_s"] < end)]
        return running.assign(bus=bus).groupby("route")["bus"].nunique()

    status = per_stop["status"].to_numpy()
    added = per_stop["added_min"].to_numpy()
    rows = {}
    for route, stop_idx in tt.events.groupby("route")["stop"].unique().items():
        st = status[stop_idx]
        open_ = st != "closed"
        rows[route] = {
            "stops": len(stop_idx),
            "closed": int((st == "closed").sum()),
            "cut_off": int((st == "cut off").sum()),
            "slower": int((st == "slower").sum()),
            "added_min": float(added[stop_idx][open_].mean()) if open_.any() else 0.0,
        }
    out = pd.DataFrame.from_dict(rows, orient="index")
    out["trips_before"] = trips_in_window(tt.events).reindex(out.index, fill_value=0)
    out["trips_after"] = trips_in_window(disrupted.events).reindex(out.index, fill_value=0)
    out["delay_min"] = pd.Series(delays_min or {}, dtype=float).reindex(out.index, fill_value=0.0)
    out["change"] = np.select(
        [out.index.isin(canceled_routes), out.index.isin(reduced_routes), out["delay_min"] > 0,
         (out[["closed", "cut_off", "slower"]].sum(axis=1) > 0)],
        ["canceled", "halved", "late", "affected"], default="normal")
    return out.rename_axis("route")
