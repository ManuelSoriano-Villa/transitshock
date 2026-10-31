"""Bus paths with timestamps, for animating the timetable on a map (deck.gl TripsLayer).

Each trip becomes a path with the time the bus reaches each point. When the feed
has street shapes (shapes.txt) and stop distances along them, buses follow the
actual streets: shape points between two stops get times interpolated by distance
traveled. Otherwise they move in straight lines between stops.

Each path also lists the stops it serves and when, so the map can answer "which
buses come to this stop next?" and "what's this bus's next stop?" on click.
"""
import numpy as np
import pandas as pd

from transitshock.timetable import Timetable

# Categorical palette (colorblind-friendly) cycled by route.
PALETTE = [
    [0, 114, 178], [230, 159, 0], [0, 158, 115], [204, 121, 167],
    [86, 180, 233], [213, 94, 0], [240, 228, 66], [170, 170, 170],
]
LATE_COLOR = [255, 80, 40]


def street_path(stop_dist, stop_times, stop_lat, stop_lon, shape) -> tuple[list, list] | None:
    """Follow the street shape from the first stop to the last.

    Returns ([[lon, lat], ...], [seconds, ...]) or None if the shape can't be used
    (missing or non-increasing distances), in which case the caller draws straight lines.
    """
    if shape is None or np.isnan(stop_dist).any():
        return None
    d_shape, lat_shape, lon_shape = shape
    # Times are interpolated by distance, which needs distances that only go forward.
    if np.any(np.diff(stop_dist) < 0) or len(d_shape) < 2:
        return None
    first, last = stop_dist[0], stop_dist[-1]
    inside = d_shape[(d_shape > first) & (d_shape < last)]
    dist = np.union1d(stop_dist, inside)
    # Unique stop distances for time interpolation (two stops at the same spot share a time).
    d_unique, idx = np.unique(stop_dist, return_index=True)
    times = np.interp(dist, d_unique, np.asarray(stop_times, dtype=float)[idx])
    lat = np.interp(dist, d_shape, lat_shape)
    lon = np.interp(dist, d_shape, lon_shape)
    # The ends snap to the stops themselves, so the bus starts and ends at the pole.
    lat[0], lon[0], lat[-1], lon[-1] = stop_lat[0], stop_lon[0], stop_lat[-1], stop_lon[-1]
    return np.column_stack([lon, lat]).round(6).tolist(), np.round(times, 1).tolist()


def trip_paths(
    tt: Timetable,
    lat: np.ndarray,
    lon: np.ndarray,
    start_s: int,
    end_s: int,
    highlight_routes: set[str] = frozenset(),
    route_names: dict[str, str] | None = None,
) -> list[dict]:
    """Every trip with at least one stop event in [start_s, end_s].

    Timestamps are seconds since start_s (the animation clock). Routes in
    highlight_routes (e.g. running late) are drawn in a warning color.
    """
    ev = tt.events
    active = ev.loc[ev["dep_s"].between(start_s, end_s) | ev["arr_s"].between(start_s, end_s), "trip"].unique()
    ev = ev[ev["trip"].isin(active)].sort_values(["trip", "seq"])
    routes = sorted(ev["route"].unique())
    color = {r: PALETTE[i % len(PALETTE)] for i, r in enumerate(routes)}
    route_names = route_names or {}

    paths = []
    for (trip, route), g in ev.groupby(["trip", "route"], sort=False):
        if len(g) < 2:
            continue
        stops = g["stop"].to_numpy()
        times = g["arr_s"].to_numpy() - start_s
        bus = int(g["bus"].iloc[0]) if "bus" in g else int(trip)  # blocked trips are split into pieces
        info = tt.trip_info.loc[bus] if tt.trip_info is not None and bus in tt.trip_info.index else None

        shape = None
        if info is not None and tt.shapes is not None and "dist" in g:
            shape = tt.shapes.get(info["shape_id"])
        street = street_path(g["dist"].to_numpy(dtype=float), times, lat[stops], lon[stops], shape) if shape else None
        path, stamps = street if street else (np.column_stack([lon[stops], lat[stops]]).round(6).tolist(),
                                              times.tolist())
        headsign = info["headsign"] if info is not None and isinstance(info["headsign"], str) else ""
        paths.append({
            "bus": bus,
            "route": str(route),
            "name": route_names.get(route, str(route)),
            "headsign": headsign.title(),
            "path": path,
            "timestamps": stamps,
            "stops": [[int(s), int(t)] for s, t in zip(stops, times)],
            "color": LATE_COLOR if route in highlight_routes else color[route],
            "late": route in highlight_routes,
        })
    return paths


def hhmm(seconds: int) -> str:
    seconds = int(seconds) % (24 * 3600)
    return f"{seconds // 3600:02d}:{seconds % 3600 // 60:02d}"


def buses_running(paths: list[dict], t: float) -> int:
    """How many buses are between their first and last stop at animation time t."""
    return len({p["bus"] for p in paths if p["timestamps"][0] <= t <= p["timestamps"][-1]})


def summarize_paths(paths: list[dict]) -> pd.Series:
    return pd.Series({
        "trips shown": len(paths),
        "routes": len({p["route"] for p in paths}),
        "points": sum(len(p["path"]) for p in paths),
        "following streets": sum(len(p["path"]) > len(p["stops"]) for p in paths),
    })
