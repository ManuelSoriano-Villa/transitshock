"""Bus paths with timestamps, for animating the timetable on a map (deck.gl TripsLayer).

Each trip becomes a path through its stops' coordinates, with the time it
reaches each stop. The map interpolates between stops, so buses glide along
straight segments between consecutive stops.
"""
import numpy as np
import pandas as pd

from transitshock.timetable import Timetable

# Categorical palette (colorblind-friendly) cycled by route.
PALETTE = [
    [0, 114, 178], [230, 159, 0], [0, 158, 115], [204, 121, 167],
    [86, 180, 233], [213, 94, 0], [240, 228, 66], [170, 170, 170],
]


def trip_paths(
    tt: Timetable,
    lat: np.ndarray,
    lon: np.ndarray,
    start_s: int,
    end_s: int,
    highlight_routes: set[str] = frozenset(),
) -> list[dict]:
    """Every trip with at least one stop event in [start_s, end_s].

    Timestamps are seconds since start_s (the animation clock). Routes in
    highlight_routes are drawn in a warning color.
    """
    ev = tt.events
    active = ev.loc[ev["dep_s"].between(start_s, end_s) | ev["arr_s"].between(start_s, end_s), "trip"].unique()
    ev = ev[ev["trip"].isin(active)].sort_values(["trip", "seq"])
    routes = sorted(ev["route"].unique())
    color = {r: PALETTE[i % len(PALETTE)] for i, r in enumerate(routes)}

    paths = []
    for (trip, route), g in ev.groupby(["trip", "route"], sort=False):
        if len(g) < 2:
            continue
        stops = g["stop"].to_numpy()
        paths.append({
            "bus": int(g["bus"].iloc[0]) if "bus" in g else int(trip),  # blocked trips are split into pieces
            "route": str(route),
            "path": np.column_stack([lon[stops], lat[stops]]).round(6).tolist(),
            "timestamps": (g["arr_s"].to_numpy() - start_s).tolist(),
            "color": [255, 80, 40] if route in highlight_routes else color[route],
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
    })
