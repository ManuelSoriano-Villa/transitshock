"""Timetable view of a feed: every scheduled bus hop as a "connection", plus disruptions.

The static graph (build_graph.py) collapses each stop pair to one median travel
time, so it can't see waiting: a rider who just missed the bus waits for the next
one, and a transfer costs however long the connecting bus takes to show up.
A timetable keeps every individual departure, so routing over it (csa.py) gets
real door-to-door times, including waits and transfers.

A connection is one bus going from one stop to the next:
    (departure time, arrival time, from stop, to stop, trip)

Disruptions are applied to the underlying stop events and the connections are
rebuilt, so e.g. a closed stop is skipped (the bus drives past) rather than
breaking the trip in two.
"""
from dataclasses import dataclass

import networkx as nx
import numpy as np
import pandas as pd

from transitshock.build_graph import WALK_RADIUS_M, WALK_SPEED_M_S, node_ids, walking_pairs


@dataclass
class Timetable:
    stops: list               # node IDs; position = stop index used everywhere below
    events: pd.DataFrame      # one row per (trip, stop) visit: trip, stop, seq, arr_s, dep_s, route
    dep: np.ndarray           # connection arrays, sorted by departure time
    arr: np.ndarray
    from_stop: np.ndarray
    to_stop: np.ndarray
    trip: np.ndarray
    n_trips: int
    foot_ptr: np.ndarray      # walking transfers in CSR form: stop s walks to
    foot_to: np.ndarray       #   foot_to[foot_ptr[s]:foot_ptr[s+1]]
    foot_time: np.ndarray     #   taking foot_time[...] seconds

    @property
    def n_stops(self) -> int:
        return len(self.stops)

    def __len__(self) -> int:
        return len(self.dep)


def connections_from_events(events: pd.DataFrame) -> dict[str, np.ndarray]:
    """Consecutive visits on the same trip -> connections, sorted by departure time."""
    ev = events.sort_values(["trip", "seq"])
    nxt = ev.groupby("trip")[["stop", "arr_s"]].shift(-1)
    conn = pd.DataFrame({
        "dep": ev["dep_s"], "arr": nxt["arr_s"],
        "from_stop": ev["stop"], "to_stop": nxt["stop"], "trip": ev["trip"],
    }).dropna()
    conn = conn[conn["from_stop"] != conn["to_stop"]]  # platforms merged into one station
    conn = conn.sort_values(["dep", "arr"], kind="stable")
    return {
        "dep": conn["dep"].to_numpy(np.int32),
        "arr": conn["arr"].to_numpy(np.int32),
        "from_stop": conn["from_stop"].to_numpy(np.int32),
        "to_stop": conn["to_stop"].to_numpy(np.int32),
        "trip": conn["trip"].to_numpy(np.int32),
    }


def footpaths_csr(pairs: pd.DataFrame, n_stops: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(from, to, seconds) walking pairs -> CSR arrays so numba can loop over a stop's neighbors."""
    pairs = pairs.sort_values("u")
    counts = np.bincount(pairs["u"].to_numpy(), minlength=n_stops)
    ptr = np.concatenate([[0], np.cumsum(counts)]).astype(np.int32)
    return ptr, pairs["v"].to_numpy(np.int32), pairs["seconds"].to_numpy(np.int32)


def build_timetable(
    data: dict[str, pd.DataFrame],
    feed: str,
    G: nx.DiGraph,
    walk_radius_m: float = WALK_RADIUS_M,
    walk_speed_m_s: float = WALK_SPEED_M_S,
) -> Timetable:
    """Timetable for one feed (load_feed output), with stops indexed in the same order as G's nodes."""
    stops = list(G.nodes)
    index = {node: i for i, node in enumerate(stops)}

    st = data["stop_times"]
    node = st["stop_id"].map(node_ids(data["stops"], feed))
    trips = data["trips"].set_index("trip_id")["route_id"]
    trip_codes, _ = pd.factorize(st["trip_id"])
    events = pd.DataFrame({
        "trip": trip_codes.astype(np.int32),
        "stop": node.map(index),
        "seq": st["stop_sequence"].to_numpy(),
        "arr_s": st["arrival_s"].to_numpy(),
        "dep_s": st["departure_s"].to_numpy(),
        "route": st["trip_id"].map(trips).to_numpy(),
    }).dropna(subset=["stop"])
    events = events.astype({"stop": np.int32, "seq": np.int32, "arr_s": np.int32, "dep_s": np.int32})
    return timetable_from_events(events, G, walk_radius_m, walk_speed_m_s)


def timetable_from_events(
    events: pd.DataFrame,
    G: nx.DiGraph,
    walk_radius_m: float = WALK_RADIUS_M,
    walk_speed_m_s: float = WALK_SPEED_M_S,
) -> Timetable:
    """Timetable from stop events (stop = index into list(G.nodes)); walking transfers come from G's coordinates.

    Used both when building from GTFS and when loading the app's exported events.
    """
    stops = list(G.nodes)
    coords = pd.DataFrame({"lat": [G.nodes[n]["lat"] for n in stops],
                           "lon": [G.nodes[n]["lon"] for n in stops]}, index=range(len(stops)))
    pairs = walking_pairs(coords, walk_radius_m)
    pairs["seconds"] = np.ceil(pairs["distance_m"] / walk_speed_m_s)
    ptr, to, secs = footpaths_csr(pairs, len(stops))
    return Timetable(stops, events, **connections_from_events(events), n_trips=int(events["trip"].max()) + 1,
                     foot_ptr=ptr, foot_to=to, foot_time=secs)


def disrupt(
    tt: Timetable,
    closed_stops: list = (),
    canceled_routes: list[str] = (),
    reduced_routes: list[str] = (),
    delays_min: dict[str, float] | None = None,
    blocked_stops: list = (),
) -> Timetable:
    """A copy of the timetable with disruptions applied.

    closed_stops:    node IDs. Buses skip them (no boarding or alighting) and nobody can walk to/from them.
    blocked_stops:   node IDs. Nothing can pass (e.g. a road closure): each trip is split there, so
                     riders can't ride through. This matches removing the node from the static graph.
    canceled_routes: route IDs. Every trip removed.
    reduced_routes:  route IDs. Every other trip removed (in each direction's departure order),
                     e.g. a driver shortage halving frequency.
    delays_min:      {route ID: minutes}. Every trip on the route runs that many minutes late.
    """
    ev = tt.events
    closed = {tt.stops.index(s) for s in closed_stops}
    if closed:
        ev = ev[~ev["stop"].isin(closed)]
    if canceled_routes:
        ev = ev[~ev["route"].isin(canceled_routes)]
    if reduced_routes:
        on_route = ev[ev["route"].isin(reduced_routes)]
        first_stop = on_route.sort_values("seq").groupby("trip").agg(start=("dep_s", "first"), route=("route", "first"),
                                                                       origin=("stop", "first"))
        first_stop = first_stop.sort_values("start")
        # Within each route + starting stop (a proxy for direction), keep every other trip.
        drop = first_stop[first_stop.groupby(["route", "origin"]).cumcount() % 2 == 1].index
        ev = ev[~ev["trip"].isin(drop)]
    if delays_min:
        shift = ev["route"].map(delays_min).fillna(0).to_numpy() * 60
        ev = ev.assign(arr_s=ev["arr_s"] + shift, dep_s=ev["dep_s"] + shift)
    n_trips = tt.n_trips
    blocked = {tt.stops.index(s) for s in blocked_stops}
    if blocked:
        ev = ev.sort_values(["trip", "seq"])
        hit = ev["stop"].isin(blocked)
        # Number the pieces of each trip: the piece counter goes up after every blocked stop.
        piece = hit.astype(int).groupby(ev["trip"]).cumsum()
        ev = ev[~hit]
        if "bus" not in ev:
            ev = ev.assign(bus=ev["trip"])  # remember the original trip: the pieces are still one bus
        codes, uniques = pd.factorize(pd.MultiIndex.from_arrays([ev["trip"], piece[~hit]]))
        ev = ev.assign(trip=codes.astype(np.int32))
        n_trips = len(uniques)

    ptr, to, secs = tt.foot_ptr, tt.foot_to, tt.foot_time
    unusable = closed | blocked
    if unusable:
        src = np.repeat(np.arange(tt.n_stops), np.diff(ptr))
        keep = ~np.isin(src, list(unusable)) & ~np.isin(to, list(unusable))
        pairs = pd.DataFrame({"u": src[keep], "v": to[keep], "seconds": secs[keep]})
        ptr, to, secs = footpaths_csr(pairs, tt.n_stops)

    return Timetable(tt.stops, ev, **connections_from_events(ev), n_trips=n_trips,
                     foot_ptr=ptr, foot_to=to, foot_time=secs)
