"""Connection Scan Algorithm (CSA): earliest arrival times over a timetable.

Given a start stop and a departure time, CSA answers "what's the earliest I can
reach every other stop?" in ONE pass over all connections sorted by departure time:

    best[s] = earliest known arrival at stop s (start: departure time; others: never)
    for each connection c, in departure order:
        if I'm already on c's bus, or I can be at c's stop before it leaves:
            mark c's trip as boarded
            if c reaches its next stop earlier than best[next stop]: improve it,
            and also improve stops within walking distance of it

Waiting is built in: if the next bus leaves at 8:14 and you arrive at 8:02, best
travel time counts all 12 minutes. That's what the static graph (median travel
times) can't capture.

Reference: Dibbelt, Pajor, Strasser, Wagner, "Connection Scan Algorithm" (2013/2018).

The inner loop is compiled with numba when it's installed (about 100x faster);
otherwise the identical pure-Python code runs.
"""
import numpy as np
import pandas as pd

from transitshock.timetable import Timetable

try:
    from numba import njit, prange
except ImportError:  # same results, just slower
    def njit(*args, **kwargs):
        return args[0] if args and callable(args[0]) else (lambda f: f)
    prange = range

NEVER = np.int32(2**31 - 1)


@njit(cache=True)
def _earliest_arrival(dep, arr, from_stop, to_stop, trip, n_trips, foot_ptr, foot_to, foot_time,
                      source, t0, horizon):
    n_stops = len(foot_ptr) - 1
    best = np.full(n_stops, NEVER, dtype=np.int32)
    best[source] = t0
    for k in range(foot_ptr[source], foot_ptr[source + 1]):
        best[foot_to[k]] = min(best[foot_to[k]], t0 + foot_time[k])

    boarded = np.zeros(n_trips, dtype=np.bool_)
    i = np.searchsorted(dep, t0)  # connections are sorted by departure; skip earlier ones
    end_time = t0 + horizon
    while i < len(dep) and dep[i] <= end_time:
        tr = trip[i]
        if boarded[tr] or best[from_stop[i]] <= dep[i]:
            boarded[tr] = True
            s, a = to_stop[i], arr[i]
            if a < best[s]:
                best[s] = a
                for k in range(foot_ptr[s], foot_ptr[s + 1]):
                    w = a + foot_time[k]
                    if w < best[foot_to[k]]:
                        best[foot_to[k]] = w
        i += 1
    return best


@njit(parallel=True, cache=True)
def _many(dep, arr, from_stop, to_stop, trip, n_trips, foot_ptr, foot_to, foot_time,
          sources, t0s, horizon):
    out = np.empty((len(sources), len(foot_ptr) - 1), dtype=np.int32)
    for q in prange(len(sources)):
        out[q] = _earliest_arrival(dep, arr, from_stop, to_stop, trip, n_trips, foot_ptr, foot_to,
                                   foot_time, sources[q], t0s[q], horizon)
    return out


def earliest_arrival(tt: Timetable, source: int, t0: int, horizon_s: int = 3 * 3600) -> np.ndarray:
    """Earliest arrival (seconds after midnight) at every stop from `source` leaving at `t0`.

    Stops not reachable within `horizon_s` get NEVER.
    """
    return _earliest_arrival(tt.dep, tt.arr, tt.from_stop, tt.to_stop, tt.trip, tt.n_trips,
                             tt.foot_ptr, tt.foot_to, tt.foot_time, np.int32(source), np.int32(t0),
                             np.int32(horizon_s))


def travel_times(tt: Timetable, sources: np.ndarray, t0s: np.ndarray, horizon_s: int = 3 * 3600) -> np.ndarray:
    """Door-to-door minutes for every (source, departure time) query, shape (queries, stops).

    Includes waiting for the first bus and for every transfer. NaN = not reachable
    within the horizon. Runs queries in parallel across CPU cores when numba is installed.
    """
    sources = np.asarray(sources, dtype=np.int32)
    t0s = np.asarray(t0s, dtype=np.int32)
    best = _many(tt.dep, tt.arr, tt.from_stop, tt.to_stop, tt.trip, tt.n_trips,
                 tt.foot_ptr, tt.foot_to, tt.foot_time, sources, t0s, np.int32(horizon_s))
    # float32 halves memory; minute-level precision is plenty. Matters for big networks (LA: ~14K stops).
    minutes = ((best - t0s[:, None]) / 60.0).astype(np.float32)
    minutes[best == NEVER] = np.nan
    return minutes


def departure_grid(sources: np.ndarray, start: str = "07:00", end: str = "09:00", every_min: int = 15):
    """Every source paired with every departure time in [start, end) -> (sources, t0s) arrays."""
    to_s = lambda hhmm: int(hhmm[:2]) * 3600 + int(hhmm[3:5]) * 60
    times = np.arange(to_s(start), to_s(end), every_min * 60, dtype=np.int32)
    return np.repeat(np.asarray(sources, dtype=np.int32), len(times)), np.tile(times, len(sources))


def summarize(minutes: np.ndarray, sources: np.ndarray, n_stops: int, removed: np.ndarray | None = None) -> dict:
    """Network metrics from a travel_times result, comparable to metrics.evaluate.

    Pairs are (source, destination != source). As in metrics.evaluate, every original
    stop counts, so closed stops count as unreachable.
    """
    m = minutes.copy()
    m[np.arange(len(m)), sources] = np.nan  # ignore each source's trip to itself
    if removed is not None:
        m[:, removed] = np.nan
        m[removed[sources]] = np.nan
    pairs = len(m) * (n_stops - 1)
    reach = m[np.isfinite(m)]
    return {
        "efficiency": float((1 / np.maximum(reach, 1 / 60)).sum() / pairs),
        "reachable_share": len(reach) / pairs,
        "mean_time_min": float(reach.mean()) if len(reach) else float("nan"),
    }


def per_stop_minutes(minutes: np.ndarray, sources: np.ndarray) -> pd.DataFrame:
    """Average door-to-door minutes FROM each source stop (over departure times and reachable destinations)."""
    df = pd.DataFrame(minutes)
    df["source"] = sources
    return df.groupby("source").mean().mean(axis=1).rename("mean_min").to_frame()
