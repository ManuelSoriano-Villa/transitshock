"""Tests for the timetable and Connection Scan Algorithm on a hand-written schedule."""
import numpy as np
import pandas as pd
import pytest

from transitshock import csa
from transitshock.timetable import Timetable, connections_from_events, disrupt, footpaths_csr

STOPS = ["A", "B", "C", "D", "E"]
I = {s: i for i, s in enumerate(STOPS)}


def hm(t: str) -> int:
    h, m = t.split(":")
    return int(h) * 3600 + int(m) * 60


def make_tt(trips: dict, routes: dict, walks=(("C", "E", 60),)) -> Timetable:
    """trips: {trip_name: [(stop, "HH:MM"), ...]}; routes: {trip_name: route_id}."""
    rows, names = [], list(trips)
    for t, visits in trips.items():
        for seq, (stop, time) in enumerate(visits, start=1):
            rows.append({"trip": names.index(t), "stop": I[stop], "seq": seq,
                         "arr_s": hm(time), "dep_s": hm(time), "route": routes[t]})
    events = pd.DataFrame(rows)
    pairs = pd.DataFrame([(I[a], I[b], s) for a, b, s in walks] + [(I[b], I[a], s) for a, b, s in walks],
                         columns=["u", "v", "seconds"])
    ptr, to, secs = footpaths_csr(pairs, len(STOPS))
    return Timetable(STOPS, events, **connections_from_events(events), n_trips=len(names),
                     foot_ptr=ptr, foot_to=to, foot_time=secs)


@pytest.fixture
def tt():
    return make_tt(
        trips={
            "T1": [("A", "08:00"), ("B", "08:10"), ("C", "08:20")],
            "T2": [("A", "08:30"), ("B", "08:40"), ("C", "08:50")],
            "X1": [("C", "08:25"), ("D", "08:35")],   # connects with T1 at C (5 min transfer)
            "X2": [("C", "08:55"), ("D", "09:05")],
        },
        routes={"T1": "1", "T2": "1", "X1": "2", "X2": "2"},
    )


def arrival(tt, frm, t0, to):
    best = csa.earliest_arrival(tt, I[frm], hm(t0))
    return None if best[I[to]] == csa.NEVER else best[I[to]]


def test_direct_ride_includes_initial_wait(tt):
    assert arrival(tt, "A", "07:55", "C") == hm("08:20")
    minutes = csa.travel_times(tt, [I["A"]], [hm("07:55")])
    assert minutes[0, I["C"]] == pytest.approx(25)  # 5 min wait + 20 min ride


def test_missed_bus_waits_for_next(tt):
    assert arrival(tt, "A", "08:01", "C") == hm("08:50")


def test_transfer(tt):
    assert arrival(tt, "A", "07:55", "D") == hm("08:35")  # T1 then X1


def test_missed_transfer_takes_later_bus(tt):
    # Leaving after T1, you take T2 (arrives C 08:50) and must wait for X2 at 08:55.
    assert arrival(tt, "A", "08:01", "D") == hm("09:05")


def test_footpath_after_arrival(tt):
    assert arrival(tt, "A", "07:55", "E") == hm("08:21")  # arrive C 08:20, walk 60 s


def test_footpath_from_source(tt):
    assert arrival(tt, "E", "08:20", "D") == hm("08:35")  # walk to C by 08:21, catch X1 at 08:25


def test_unreachable_and_horizon(tt):
    assert arrival(tt, "D", "08:00", "A") is None  # no trips go backwards
    best = csa.earliest_arrival(tt, I["A"], hm("07:55"), horizon_s=10 * 60)
    assert best[I["C"]] == csa.NEVER  # T1 departs B at 08:10, past the 10-minute horizon


def test_numba_matches_pure_python(tt):
    compiled = csa.earliest_arrival(tt, I["A"], hm("07:55"))
    py = getattr(csa._earliest_arrival, "py_func", csa._earliest_arrival)
    pure = py(tt.dep, tt.arr, tt.from_stop, tt.to_stop, tt.trip, tt.n_trips, tt.foot_ptr, tt.foot_to,
              tt.foot_time, np.int32(I["A"]), np.int32(hm("07:55")), np.int32(3 * 3600))
    np.testing.assert_array_equal(compiled, pure)


def test_closed_stop_is_skipped_not_a_dead_end(tt):
    d = disrupt(tt, closed_stops=["B"])
    assert arrival(d, "A", "07:55", "C") == hm("08:20")  # bus drives past B
    assert arrival(d, "A", "07:55", "B") is None


def test_closed_stop_blocks_walking(tt):
    d = disrupt(tt, closed_stops=["E"])
    assert arrival(d, "A", "07:55", "E") is None
    assert arrival(d, "A", "07:55", "D") == hm("08:35")


def test_canceled_route(tt):
    d = disrupt(tt, canceled_routes=["2"])
    assert arrival(d, "A", "07:55", "D") is None


def test_reduced_route_keeps_every_other_trip(tt):
    d = disrupt(tt, reduced_routes=["1"])
    assert arrival(d, "A", "07:55", "C") == hm("08:20")  # T1 kept
    assert arrival(d, "A", "08:01", "C") is None          # T2 dropped


def test_delayed_route_can_break_a_transfer(tt):
    d = disrupt(tt, delays_min={"1": 7})  # T1 now reaches C at 08:27, after X1 leaves at 08:25
    assert arrival(d, "A", "07:55", "C") == hm("08:27")
    assert arrival(d, "A", "07:55", "D") == hm("09:05")


def test_departure_grid():
    src, t0 = csa.departure_grid([0, 1], "07:00", "08:00", every_min=30)
    assert src.tolist() == [0, 0, 1, 1]
    assert t0.tolist() == [hm("07:00"), hm("07:30")] * 2


def test_summarize_counts_closed_stops_as_unreachable(tt):
    src, t0 = csa.departure_grid([I["A"]], "07:55", "07:56")
    m = csa.travel_times(tt, src, t0)
    full = csa.summarize(m, src, len(STOPS))
    removed = np.array([s == "E" for s in STOPS])
    partial = csa.summarize(m, src, len(STOPS), removed=removed)
    assert full["reachable_share"] == pytest.approx(4 / 4)
    assert partial["reachable_share"] == pytest.approx(3 / 4)
    assert full["mean_time_min"] == pytest.approx((15 + 25 + 40 + 26) / 4)


def test_timetable_impact_close_and_delay(tt):
    from transitshock.scenario import timetable_baseline, timetable_impact

    base = timetable_baseline(tt, "07:55", "07:56")
    summary, per_stop = timetable_impact(base)
    assert summary["efficiency_change"] == pytest.approx(0)
    assert (per_stop["status"] == "unaffected").all()

    # Closing C strands D and E for riders from A and B (C was the only way there).
    summary, per_stop = timetable_impact(base, stops=["C"])
    assert per_stop.loc["C", "status"] == "closed"
    assert per_stop.loc["A", "status"] == "cut off"
    assert summary["pairs_lost"] > 0

    # Delaying route 1 by 7 minutes makes riders from A miss X1: D goes from 08:35 to 09:05.
    summary, per_stop = timetable_impact(base, delays_min={"1": 7})
    assert per_stop.loc["A", "added_min"] > 0
    assert summary["mean_time_after_min"] > summary["mean_time_before_min"]


def test_blocked_stop_splits_trips(tt):
    # Unlike a closed stop (bus skips it), a blocked stop can't be passed: A can no longer reach C.
    d = disrupt(tt, blocked_stops=["B"])
    assert arrival(d, "A", "07:55", "C") is None
    assert arrival(d, "A", "07:55", "B") is None
    # The trip pieces after B still run: nothing reaches B, but X1 still goes C -> D.
    assert arrival(d, "C", "08:20", "D") == hm("08:35")


def test_trip_paths_for_animation(tt):
    from transitshock.animation import buses_running, hhmm, trip_paths

    lat = np.arange(len(STOPS), dtype=float)
    lon = -np.arange(len(STOPS), dtype=float)
    paths = trip_paths(tt, lat, lon, hm("08:00"), hm("08:30"), highlight_routes={"2"})
    assert len(paths) == 3  # T1, X1, T2 (T2 departs A at 08:30); X2 starts at 08:55
    t1 = next(p for p in paths if p["timestamps"][0] == 0)
    assert t1["path"] == [[0.0, 0.0], [-1.0, 1.0], [-2.0, 2.0]]  # A, B, C as [lon, lat]
    assert t1["timestamps"] == [0, 600, 1200]
    assert all(p["color"] == [255, 80, 40] for p in paths if p["route"] == "2")
    assert buses_running(paths, 1500) == 1  # only X1 (08:25-08:35) is on the road at 08:25
    assert hhmm(hm("25:15")) == "01:15"


def test_blocked_trip_pieces_count_as_one_bus(tt):
    from transitshock.animation import buses_running, trip_paths

    lat = lon = np.arange(len(STOPS), dtype=float)
    d = disrupt(tt, blocked_stops=["B"])
    paths = trip_paths(d, lat, lon, hm("08:00"), hm("08:30"))
    # T1 (A-B-C) loses B and splits into A and C pieces with one stop each, so neither is drawn;
    # nothing is double-counted.
    assert buses_running(paths, 1500) == 1
