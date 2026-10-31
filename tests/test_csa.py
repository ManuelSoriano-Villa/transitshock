"""Tests for the timetable and Connection Scan Algorithm on a hand-written schedule."""
import networkx as nx
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


def test_timetable_removal_damage(tt):
    from transitshock.simulate import timetable_removal_damage

    dmg = timetable_removal_damage(tt, "blocked", "07:55", "08:05", every_min=5)
    assert list(dmg.index) == STOPS
    # Blocking C (the transfer point) strands D and E from A and B; blocking D (a dead end) only loses D.
    assert dmg.loc["C", "tt_efficiency_loss"] > dmg.loc["D", "tt_efficiency_loss"] > 0
    assert dmg.loc["C", "tt_reachable_loss"] > dmg.loc["D", "tt_reachable_loss"]
    closed = timetable_removal_damage(tt, "closed", "07:55", "08:05", every_min=5)
    assert closed.loc["B", "tt_efficiency_loss"] <= dmg.loc["B", "tt_efficiency_loss"]  # skipping is milder


def test_street_path_follows_shape_and_interpolates_time():
    from transitshock.animation import street_path

    # A shape that bends: east 1 mile, then north 1 mile. Stops at the start and the end.
    shape = (np.array([0.0, 0.5, 1.0, 1.5, 2.0]),
             np.array([33.0, 33.0, 33.0, 33.01, 33.02]),
             np.array([-118.02, -118.01, -118.0, -118.0, -118.0]))
    path, times = street_path(np.array([0.0, 2.0]), np.array([0, 400]),
                              np.array([33.0, 33.02]), np.array([-118.02, -118.0]), shape)
    assert len(path) == 5  # both stops plus the 3 shape points between them
    assert path[2] == [-118.0, 33.0]  # the corner is kept
    assert times == [0, 100, 200, 300, 400]  # time grows with distance traveled


def test_street_path_falls_back_when_distances_go_backwards():
    from transitshock.animation import street_path

    shape = (np.array([0.0, 1.0]), np.array([33.0, 33.1]), np.array([-118.0, -118.0]))
    assert street_path(np.array([1.0, 0.5]), np.array([0, 60]), np.zeros(2), np.zeros(2), shape) is None
    assert street_path(np.array([0.0, np.nan]), np.array([0, 60]), np.zeros(2), np.zeros(2), shape) is None


def test_simplify_mask_drops_collinear_points():
    from transitshock.timetable import simplify_mask

    lat = np.array([33.0, 33.001, 33.002, 33.003, 33.003])
    lon = np.array([-118.0, -118.0, -118.0, -118.0, -117.99])  # straight north, then a turn east
    keep = simplify_mask(lat, lon, tolerance_m=1.0)
    assert keep.tolist() == [True, False, False, True, True]


def test_greedy_blocking_picks_the_transfer_point_first(tt):
    from transitshock.simulate import greedy_blocking

    result = greedy_blocking(tt, 2, candidates=["A", "B", "C", "D", "E"], start="07:55", end="08:05", every_min=5)
    assert result["stop"].iloc[0] == "C"  # C is the only way to D and E
    assert result["total_loss"].is_monotonic_increasing
    assert result["extra_loss"].iloc[0] >= result["extra_loss"].iloc[1]


def test_lazy_and_plain_greedy_agree_on_small_case(tt):
    from transitshock.simulate import greedy_blocking

    kwargs = dict(candidates=["A", "B", "C", "D", "E"], start="07:55", end="08:05", every_min=5)
    lazy = greedy_blocking(tt, 3, lazy=True, **kwargs)
    plain = greedy_blocking(tt, 3, lazy=False, **kwargs)
    assert lazy["stop"].tolist() == plain["stop"].tolist()
    assert lazy["evaluations"].iloc[-1] <= plain["evaluations"].iloc[-1]


def test_route_impacts(tt):
    from transitshock.scenario import route_impacts, timetable_baseline, timetable_impact

    base = timetable_baseline(tt, "07:55", "08:56", every_min=30)
    assert base.window_s == (hm("07:55"), hm("08:56"))
    kw = dict(canceled_routes=["2"], blocked_stops=["B"])
    _, per_stop = timetable_impact(base, **kw)
    routes = route_impacts(base, per_stop, **kw)
    assert routes.loc["2", "change"] == "canceled"
    assert routes.loc["2", "trips_before"] == 2 and routes.loc["2", "trips_after"] == 0
    assert routes.loc["1", "closed"] == 1  # B is on route 1
    assert routes.loc["1", "change"] == "affected"
    # Route 1's trips are split at B but are still the same 2 buses.
    assert routes.loc["1", "trips_after"] == routes.loc["1", "trips_before"] == 2

    _, per_stop = timetable_impact(base, delays_min={"1": 7})
    late = route_impacts(base, per_stop, delays_min={"1": 7})
    assert late.loc["1", "change"] == "late" and late.loc["1", "delay_min"] == 7


def test_project_stops_on_straight_line():
    from transitshock.timetable import project_stops, shape_geometry

    shapes = pd.DataFrame({"shape_id": "s", "shape_pt_lat": np.linspace(33.0, 33.01, 11),
                           "shape_pt_lon": -118.0, "shape_pt_sequence": range(11),
                           "shape_dist_traveled": np.nan})
    geo = shape_geometry(shapes)["s"]
    assert geo[0][-1] == pytest.approx(1112, rel=0.01)  # ~1.1 km measured along the line
    d = project_stops([33.002, 33.0055, 33.009], [-118.0001] * 3, geo)
    assert d.tolist() == sorted(d.tolist())
    assert d[0] == pytest.approx(222, abs=10) and d[2] == pytest.approx(1000, abs=10)


def test_project_stops_keeps_order_on_a_loop():
    from transitshock.timetable import project_stops

    # Out and back on the same street: the return stop is near the start but must land at the end.
    lat = np.concatenate([np.linspace(33.0, 33.01, 11), np.linspace(33.01, 33.0, 11)[1:]])
    lon = np.concatenate([np.full(11, -118.0), np.full(10, -118.0001)])
    dist = np.arange(len(lat), dtype=float)
    d = project_stops([33.0005, 33.01, 33.0005], [-118.0, -118.0, -118.0001], (dist, lat, lon))
    assert d[0] < d[1] < d[2]
    assert d[2] >= 19  # matched on the way back, not the first point


def test_combine_timetables_prefixes_routes_and_shapes(tt):
    from dataclasses import replace

    from transitshock.timetable import combine_timetables

    G = nx.DiGraph()
    for i, n in enumerate(STOPS):
        G.add_node(n, lat=33.0 + i * 0.01, lon=-118.0)
    a = replace(tt, trip_info=pd.DataFrame({"shape_id": ["s1"] * tt.n_trips, "headsign": ""}),
                shapes={"s1": (np.array([0.0, 1.0]), np.array([33.0, 33.1]), np.array([-118.0, -118.0]))})
    both = combine_timetables({"lbt": a, "metro": a}, G)
    assert both.n_trips == 2 * tt.n_trips
    assert set(both.events["route"]) == {"lbt:1", "lbt:2", "metro:1", "metro:2"}
    assert set(both.shapes) == {"lbt:s1", "metro:s1"}
    assert both.trip_info.loc[tt.n_trips, "shape_id"] == "metro:s1"
    assert len(both) == 2 * len(tt)


def test_sampled_baseline_covers_every_stop(tt):
    from transitshock.scenario import timetable_baseline, timetable_impact

    base = timetable_baseline(tt, "07:55", "07:56", max_sources=2, seed=0)
    assert base.sampled and len(set(base.sources)) == 2
    summary, per_stop = timetable_impact(base, blocked_stops=["C"])
    assert len(per_stop) == len(STOPS)  # every stop gets a status, judged by trips to it
    assert per_stop.loc["C", "status"] == "closed"
    full = timetable_baseline(tt, "07:55", "07:56")
    assert not full.sampled


def test_clip_to_window_keeps_one_point_each_side():
    from transitshock.animation import clip_to_window

    path = [[i, 0] for i in range(6)]
    stamps = [-200, -100, 50, 120, 400, 900]
    p, t = clip_to_window(path, stamps, 300)
    assert t == [-100, 50, 120, 400]
    assert p == [[1, 0], [2, 0], [3, 0], [4, 0]]
    p, t = clip_to_window([[0, 0], [1, 0]], [-100, 500], 300)  # no point inside: keep the segment across it
    assert t == [-100, 500]
