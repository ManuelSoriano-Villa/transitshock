"""TransitShock demo: disrupt the Long Beach Transit network and see who's affected.

Run from the repo root:
    streamlit run app/streamlit_app.py

Reads the small committed files in app/data/ (written by `python -m transitshock.app_data`),
so it runs without the raw GTFS feeds or scikit-learn. That's what lets it deploy as-is.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # import the package without installing it (for hosting)

from transitshock.animation import buses_running, hhmm, summarize_paths, trip_paths  # noqa: E402
from transitshock.app_data import read_graph_csv, read_timetable  # noqa: E402
from transitshock.scenario import (baseline, impact, route_impacts, timetable_baseline,  # noqa: E402
                                   timetable_impact)
from transitshock.simulate import greedy_blocking  # noqa: E402
from transitshock.timetable import disrupt  # noqa: E402

APP_DIR = Path(__file__).resolve().parent
DATA = APP_DIR / "data"
PREFIX = "lbt"
TARGET = "efficiency_loss"
REPO_URL = "https://github.com/ManuelSoriano-Villa/transitshock"

WINDOWS = {
    "Morning peak (7–9 AM)": ("07:00", "09:00"),
    "Midday (11 AM–1 PM)": ("11:00", "13:00"),
    "Evening (8–10 PM)": ("20:00", "22:00"),
}
STATUS_COLORS = {
    "closed": [20, 20, 20],
    "cut off": [214, 32, 40],
    "slower": [245, 150, 30],
    "unaffected": [150, 150, 150],
}
STATUS_RADIUS = {"closed": 90, "cut off": 70, "slower": 60, "unaffected": 30}

st.set_page_config(page_title="TransitShock", page_icon="🚌", layout="wide")


# ---------- data (cached so it loads once per server) ----------

@st.cache_resource
def load_network():
    G = read_graph_csv(DATA, PREFIX)
    return G, baseline(G), read_timetable(DATA, PREFIX, G)


@st.cache_resource
def load_timetable_baseline(window: str):
    """Door-to-door times from every stop at every 15 minutes in the window (about 0.2 s)."""
    return timetable_baseline(TT, *WINDOWS[window])


@st.cache_data
def load_edges(_G) -> pd.DataFrame:
    rows = []
    for u, v, d in _G.edges(data=True):
        if d["kind"] == "transit":
            a, b = _G.nodes[u], _G.nodes[v]
            rows.append({"from": [a["lon"], a["lat"]], "to": [b["lon"], b["lat"]], "routes": d["routes"].split(",")})
    return pd.DataFrame(rows)


@st.cache_data
def load_predictions() -> tuple[pd.DataFrame, dict]:
    """Model predictions for every stop (each from a model that never saw that stop's neighborhood)."""
    return pd.read_csv(DATA / f"{PREFIX}_critical.csv", index_col="node"), \
        json.loads((DATA / f"{PREFIX}_meta.json").read_text())


if not (DATA / f"{PREFIX}_nodes.csv").exists():
    st.error("Missing app/data/. From the repo root, run:\n\n```\npython scripts/run_pipeline.py\n```")
    st.stop()

G, STATIC_BASE, TT = load_network()
EDGES = load_edges(G)
PRED, META = load_predictions()
PRED["name"] = [G.nodes[n]["name"] for n in PRED.index]
PRED["trips_per_day"] = [G.nodes[n]["trips_per_day"] for n in PRED.index]
ROUTE_NAMES = G.graph["route_names"]["lbt"]
ROUTES = sorted(ROUTE_NAMES, key=lambda r: (len(r), r))
STOP_LABELS = {n: f"{G.nodes[n]['name']} ({n.split(':')[1]})" for n in sorted(G, key=lambda n: G.nodes[n]["name"])}
LAT = np.array([G.nodes[n]["lat"] for n in TT.stops])
LON = np.array([G.nodes[n]["lon"] for n in TT.stops])


def route_label(route: str) -> str:
    """Short route label for stop popups: the route number if it starts the name, else the full name."""
    name = ROUTE_NAMES.get(route, route)
    return route if name.split(" ")[0] == route else name


@st.cache_data
def base_live_stops() -> list[dict]:
    order = lambda r: (len(r), r)
    return [{"i": i, "name": G.nodes[n]["name"], "position": [round(float(LON[i]), 6), round(float(LAT[i]), 6)],
             "routes": ", ".join(route_label(r) for r in sorted(str(G.nodes[n]["routes"]).split(","), key=order)),
             "status": ""} for i, n in enumerate(TT.stops)]


def live_stops(status: dict, per_stop: pd.DataFrame) -> list[dict]:
    """Every stop for the live map: blocked/closed per the scenario, plus how the scenario affects it."""
    impact_ = per_stop["status"].to_numpy()
    lost = per_stop["lost_destinations"].to_numpy()
    added_ = per_stop["added_min"].to_numpy()
    return [{**s, "status": status.get(TT.stops[s["i"]], ""), "impact": impact_[s["i"]],
             "lost": round(float(lost[s["i"]]), 1), "added": round(float(added_[s["i"]]), 1)}
            for s in base_live_stops()]


@st.cache_data
def normal_paths(start_s: int) -> list[dict]:
    """The undisrupted schedule's buses for the hour (to show canceled or late buses as ghosts)."""
    return trip_paths(TT, LAT, LON, start_s, start_s + 3600, route_names=ROUTE_NAMES)


@st.cache_resource
def greedy_cache() -> dict:
    """Greedy results by (window, k), shared across sessions: each search takes 10-20 seconds."""
    return {}


def run_greedy(window: str, k: int) -> None:
    """Find the k stops whose combined blockage hurts most, then block them (applied on the next rerun)."""
    key = (window, k)
    if key not in greedy_cache():
        # Candidates: the timetable model's top 40 plus the static model's top 15 (strand points).
        candidates = list(dict.fromkeys(PRED.nsmallest(40, "tt_predicted_rank").index.tolist()
                                        + PRED.nsmallest(15, "predicted_rank").index.tolist()))
        bar = st.progress(0.0, text="Starting greedy search...")
        greedy_cache()[key] = greedy_blocking(TT, k, candidates, *WINDOWS[window], every_min=30,
                                              progress=lambda f, msg: bar.progress(f, text=msg))
    result = greedy_cache()[key]
    st.session_state["greedy_result"] = {"window": window, "k": k, "table": result}
    st.session_state["pending_blocked"] = result["stop"].tolist()
    st.rerun()


# ---------- sidebar: build a scenario ----------

SCENARIO_KEYS = ["blocked", "closed", "canceled", "reduced", "delayed"]


def set_scenario(**values):
    for key in SCENARIO_KEYS:
        st.session_state[key] = list(values.get(key, []))


for key in SCENARIO_KEYS:
    st.session_state.setdefault(key, [])
if "pending_blocked" in st.session_state:
    # Set by the greedy search. Widgets can't be changed after they're drawn, so it's applied here, before them.
    set_scenario(blocked=st.session_state.pop("pending_blocked"))

with st.sidebar:
    st.header("Build a scenario")
    window = st.selectbox("Time of day", list(WINDOWS), key="window")

    st.subheader("Stops")
    st.multiselect("Block (nothing passes)", options=list(STOP_LABELS), format_func=STOP_LABELS.get, key="blocked",
                   help="Like a road closure: buses can't get past, so each trip is cut in two here.")
    st.multiselect("Close (buses skip them)", options=list(STOP_LABELS), format_func=STOP_LABELS.get, key="closed",
                   help="Buses drive past without stopping. Riders can't board or get off here.")

    st.subheader("Routes")
    st.multiselect("Cancel", options=ROUTES, format_func=ROUTE_NAMES.get, key="canceled")
    st.multiselect("Halve frequency", options=ROUTES, format_func=ROUTE_NAMES.get, key="reduced",
                   help="Every other trip is dropped, e.g. a driver shortage.")
    st.multiselect("Run late", options=ROUTES, format_func=ROUTE_NAMES.get, key="delayed")
    delay = st.slider("Minutes late", 1, 30, 10, key="delay_min", disabled=not st.session_state["delayed"])

    st.subheader("Presets")
    strand10 = PRED.nsmallest(10, "predicted_rank").index.tolist()     # static-trained model
    slow10 = PRED.nsmallest(10, "tt_predicted_rank").index.tolist()    # timetable-trained model
    busiest = PRED.nlargest(10, "trips_per_day").index.tolist()
    st.button("Block 10 stops that strand riders (static model)", key="preset_top10", on_click=set_scenario,
              kwargs={"blocked": strand10}, width="stretch",
              help="The static model's top 10: blocking them cuts small areas off completely.")
    st.button("Block 10 stops that slow everyone (timetable model)", key="preset_tt10", on_click=set_scenario,
              kwargs={"blocked": slow10}, width="stretch",
              help="The timetable-trained model's top 10: busy, frequent corridors.")
    st.button("Block the 10 busiest stops", key="preset_busy", on_click=set_scenario,
              kwargs={"blocked": busiest}, width="stretch")
    st.button("Close the 10 busiest stops (buses skip)", on_click=set_scenario,
              kwargs={"closed": busiest}, width="stretch")
    st.button("Driver shortage: halve the 5 busiest routes", on_click=set_scenario,
              kwargs={"reduced": TT.events.groupby("route")["trip"].nunique().nlargest(5).index.tolist()},
              width="stretch")
    st.button("Clear", key="preset_clear", on_click=set_scenario, width="stretch")

    st.subheader("Worst-case search")
    if st.toggle("Greedy search", key="greedy_on",
                 help="Instead of ranking stops one at a time, pick them one by one, each time taking the stop "
                      "that does the most extra damage given the ones already blocked. Finds a worse combination "
                      "than any single-stop ranking, but takes 10-20 seconds per search."):
        k = st.slider("Stops to block", 3, 15, 10, key="greedy_k")
        if st.button(f"Find and block the worst {k}", key="greedy_run", type="primary", width="stretch"):
            run_greedy(window, k)

    st.caption("Long Beach Transit, weekday schedule for Wed 2026-11-04. "
               f"{G.number_of_nodes():,} stops, {len(ROUTE_NAMES)} routes, {len(TT):,} scheduled bus hops.")

scenario = {key: st.session_state[key] for key in SCENARIO_KEYS}
delays = {r: delay for r in scenario["delayed"]}
nothing_selected = not any(scenario.values())
scenario_kwargs = dict(stops=scenario["closed"], canceled_routes=scenario["canceled"],
                       reduced_routes=scenario["reduced"], delays_min=delays, blocked_stops=scenario["blocked"])

# Computed once per rerun and shared by the what-if and live tabs.
base = load_timetable_baseline(window)
with st.spinner("Re-routing every stop at every departure time..."):
    summary, per_stop = timetable_impact(base, **scenario_kwargs)
    routes_impact = route_impacts(base, per_stop, **scenario_kwargs)
per_stop = per_stop.assign(name=[G.nodes[n]["name"] for n in per_stop.index],
                           lat=[G.nodes[n]["lat"] for n in per_stop.index],
                           lon=[G.nodes[n]["lon"] for n in per_stop.index])


def show_metrics(summary: dict) -> None:
    """The scenario's headline numbers (same on the what-if and live tabs)."""
    eff_change = 100 * summary["efficiency_change"]
    added = summary["mean_time_after_min"] - summary["mean_time_before_min"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Efficiency", f"{100 + eff_change:.1f}%", f"{eff_change:.2f}%" if abs(eff_change) >= 0.005 else None,
              help="Average of 1 / door-to-door time over all stop pairs, relative to normal service.")
    c2.metric("Average trip", f"{summary['mean_time_after_min']:.1f} min",
              f"{added:+.1f} min" if abs(added) >= 0.05 else None, delta_color="inverse",
              help="Door-to-door, including waiting for the bus and for transfers, over trips still possible.")
    c3.metric("Trips lost", f"{100 * summary['pairs_lost']:.2f}%",
              help="Share of stop-to-stop trips that can no longer be made within 3 hours.")
    c4, c5, c6 = st.columns(3)
    c4.metric("Stops cut off", summary["stops_cut_off"],
              help="Can no longer reach at least 10% of the stops they could before.")
    c5.metric("Slower stops", summary["stops_slower"],
              help="Trips from here take 2+ minutes longer on average.")
    hit = routes_impact[routes_impact["change"] != "normal"]
    c6.metric("Routes affected", f"{len(hit)} of {len(routes_impact)}",
              help="Routes that are canceled, halved, late, or have stops blocked, closed, cut off, or slower.")


def toggle_stop(kind: str, node: str) -> None:
    """Add the stop to (or remove it from) the blocked/closed list; a stop can't be both."""
    other = "closed" if kind == "blocked" else "blocked"
    current = st.session_state[kind]
    st.session_state[kind] = [n for n in current if n != node] if node in current else [*current, node]
    st.session_state[other] = [n for n in st.session_state[other] if n != node]


def stop_details(node: str, row: pd.Series, window: str) -> None:
    """Card for a clicked stop: what it is, how this scenario affects it, and buttons to disrupt it."""
    attrs = G.nodes[node]
    routes = sorted(str(attrs["routes"]).split(","), key=lambda r: (len(r), r))
    crit = PRED.loc[node] if node in PRED.index else None
    with st.container(border=True):
        left, right = st.columns([3, 2])
        with left:
            st.markdown(f"#### {attrs['name']}")
            st.caption(f"Stop {node.split(':')[1]} · {int(attrs['trips_per_day'])} trips/day · "
                       f"routes {', '.join(routes)}")
            state = ("blocked (nothing passes)" if node in st.session_state["blocked"] else
                     "closed (buses skip it)" if node in st.session_state["closed"] else row["status"])
            st.markdown(f"**In this scenario** ({window}): {state}")
            if row["status"] in ("cut off", "slower"):
                st.markdown(f"- Loses access to **{row['lost_destinations']:.0f}** stops on average\n"
                            f"- Trips from here take **{row['added_min']:+.1f} min** on average")
            if crit is not None:
                st.markdown(f"**Criticality** (blocking this stop alone, morning peak): costs "
                            f"**{100 * crit['tt_efficiency_loss']:.2f}%** of network efficiency with waiting counted "
                            f"(rank {int(crit['tt_true_rank'])} of {len(PRED)}; model rank {int(crit['tt_predicted_rank'])}), "
                            f"{100 * crit[TARGET]:.2f}% without (rank {int(crit['true_rank'])}).")
        with right:
            for kind, label in [("blocked", "Block"), ("closed", "Close")]:
                active = node in st.session_state[kind]
                st.button(f"{'Unblock' if kind == 'blocked' else 'Reopen'} this stop" if active else f"{label} this stop",
                          key=f"toggle_{kind}_{node}", on_click=toggle_stop, args=(kind, node), width="stretch")


# ---------- main ----------

st.title("🚌 TransitShock")
st.markdown("What happens to **Long Beach Transit** when stops are blocked, routes are cut, or buses run late? "
            "Build a scenario in the sidebar.")

tab_whatif, tab_live, tab_critical, tab_about = st.tabs(["What-if map", "Live buses", "Critical stops", "How it works"])

with tab_whatif:
    show_metrics(summary)

    greedy = st.session_state.get("greedy_result")
    if greedy and set(greedy["table"]["stop"]) == set(scenario["blocked"]) and greedy["window"] == window:
        with st.expander(f"Greedy search: the worst {greedy['k']} stops to block, in the order they were picked",
                         expanded=True):
            g = greedy["table"]
            st.dataframe(pd.DataFrame({
                "pick": g["step"],
                "stop": [G.nodes[s]["name"] for s in g["stop"]],
                "extra damage %": (100 * g["extra_loss"]).round(2),
                "total damage %": (100 * g["total_loss"]).round(2),
            }), width="stretch", hide_index=True)
            st.caption(f"{int(g['evaluations'].iloc[-1])} scenarios tested (departures every 30 minutes). Extra damage "
                       "can go *up* from one pick to the next: blockages compound, so a stop can matter more once "
                       "its neighbors are already blocked.")

    only_static_kinds = not (scenario["closed"] or scenario["reduced"] or scenario["delayed"])
    if not nothing_selected and only_static_kinds:
        static_summary, _ = impact(G, STATIC_BASE, scenario["blocked"], scenario["canceled"])
        st.caption(f"Without waiting times (the static model), this scenario would cost "
                   f"{-100 * static_summary['efficiency_change']:.2f}% efficiency.")

    cut_routes = set(scenario["canceled"])
    weaker_routes = set(scenario["reduced"]) | set(scenario["delayed"])
    route_state = EDGES["routes"].apply(
        lambda rs: "canceled" if set(rs) <= cut_routes and cut_routes else
                   "weaker" if set(rs) & weaker_routes else "normal")
    stops = per_stop.reset_index().assign(
        color=lambda d: d["status"].map(STATUS_COLORS),
        radius=lambda d: d["status"].map(STATUS_RADIUS),
        added=lambda d: d["added_min"].round(1),
    ).sort_values("status", key=lambda s: s.map({"unaffected": 0, "slower": 1, "cut off": 2, "closed": 3}))

    # Every layer needs an id for Streamlit to track clicks (selections) across reruns.
    line = lambda name, data, color, width: pdk.Layer("LineLayer", data, id=name, get_source_position="from",
                                                      get_target_position="to", get_color=color, get_width=width)
    layers = [
        line("routes-normal", EDGES[route_state == "normal"], [120, 120, 120, 90], 1),
        line("routes-weaker", EDGES[route_state == "weaker"], [30, 120, 220, 220], 3),
        line("routes-canceled", EDGES[route_state == "canceled"], [128, 0, 160, 220], 3),
        pdk.Layer("ScatterplotLayer", stops, id="stops", get_position=["lon", "lat"], get_fill_color="color",
                  get_radius="radius", radius_min_pixels=3, pickable=True),
    ]
    view = pdk.ViewState(latitude=float(stops["lat"].median()), longitude=float(stops["lon"].median()), zoom=10.5)
    tooltip = {"html": "<b>{name}</b><br/>{status}<br/>destinations lost: {lost_destinations}"
                       "<br/>extra minutes: {added}<br/><i>click for details</i>"}
    event = st.pydeck_chart(pdk.Deck(layers=layers, initial_view_state=view, tooltip=tooltip), height=540,
                            on_select="rerun", selection_mode="single-object", key="whatif_map")
    st.caption("⚫ blocked/closed  🔴 cut off  🟠 slower  ⚪ unaffected  "
               "🟪 canceled route segments  🟦 routes running less often or late. Click a stop for details.")

    picked = (event.selection.objects.get("stops") or [None])[0] if event and event.selection else None
    if picked:
        stop_details(picked["node"], per_stop.loc[picked["node"]], window)

    affected = per_stop[per_stop["status"].isin(["cut off", "slower"])]
    if len(affected):
        st.subheader("Most affected stops")
        st.dataframe(
            affected.sort_values(["lost_destinations", "added_min"], ascending=False)
            [["name", "status", "lost_destinations", "added_min"]]
            .rename(columns={"lost_destinations": "destinations lost", "added_min": "extra minutes"})
            .round(1).head(25),
            width="stretch", hide_index=True,
        )
    elif nothing_selected:
        st.info("Normal service. Try a preset in the sidebar.")

with tab_live:
    st.caption(f"Scenario impact, {window.lower()}:")
    show_metrics(summary)
    left, right = st.columns([3, 2])
    start = left.select_slider("Start time", options=[f"{h:02d}:00" for h in range(5, 24)],
                               value=WINDOWS[window][0], key="live_start")
    speed = right.radio("Speed", [30, 60, 120], index=1, horizontal=True, format_func=lambda s: f"{s}×", key="speed")
    start_s = int(start[:2]) * 3600
    live_tt = disrupt(TT, scenario["closed"], scenario["canceled"], scenario["reduced"], delays, scenario["blocked"])
    paths = trip_paths(live_tt, LAT, LON, start_s, start_s + 3600, highlight_routes=set(scenario["delayed"]),
                       route_names=ROUTE_NAMES)

    # Ghosts: buses from the normal schedule that this scenario removed (canceled/halved routes), and where
    # late buses *should* be. Drawn as hollow circles so the disruption itself is visible, not just missing buses.
    running = {p["bus"] for p in paths}
    ghosts = [{"name": p["name"], "headsign": p["headsign"], "route": p["route"], "path": p["path"],
               "timestamps": p["timestamps"],
               "kind": "canceled" if p["bus"] not in running else "scheduled"}
              for p in normal_paths(start_s)
              if p["bus"] not in running or p["route"] in delays]

    status = {**{n: "closed" for n in scenario["closed"]}, **{n: "blocked" for n in scenario["blocked"]}}
    route_json = {r: {k: (round(v, 2) if isinstance(v, float) else v) for k, v in row.items()}
                  for r, row in routes_impact.astype(object).to_dict("index").items()}
    # "</" is escaped so a stop or route name can never close the <script> tag early.
    as_js = lambda value: json.dumps(value, default=lambda o: o.item() if hasattr(o, "item") else str(o)).replace(
        "</", "<\\/")
    html = (APP_DIR / "live_buses.html").read_text()
    html = (html.replace("__TRIPS__", as_js(paths)).replace("__STOPS__", as_js(live_stops(status, per_stop)))
                .replace("__GHOSTS__", as_js(ghosts)).replace("__ROUTES__", as_js(route_json))
                .replace("__WINDOW__", as_js(window.lower())).replace("__DELAYS__", as_js(delays))
                .replace("__START__", str(start_s)).replace("__SPEED__", str(speed))
                .replace("__CENTER__", as_js([float(np.median(LON)), float(np.median(LAT))])))
    st.iframe(html, height=640)

    shown = summarize_paths(paths)
    canceled_now = sum(g["kind"] == "canceled" for g in ghosts)
    st.caption(
        f"One hour of the {'disrupted' if not nothing_selected else 'normal'} schedule from {start}, "
        f"played at {speed}×: {shown['trips shown']} trips on {shown['routes']} routes"
        + (f", **{canceled_now} trips removed** by the scenario" if canceled_now else "")
        + f". {buses_running(paths, 1800)} buses on the road at {hhmm(start_s + 1800)}. Buses follow the street "
        "shapes from the GTFS feed. **Click a bus** for its route's impact and next stops; **click a stop** for "
        "its impact and the next buses due. Stops are colored by impact (red: cut off, orange: slower); hollow "
        "gray rings are canceled buses, orange rings show where late buses should be."
    )

with tab_critical:
    t = META["timings"]
    measure = st.radio("Measure damage", ["With waiting (timetable)", "Without waiting (static graph)"],
                       horizontal=True, key="crit_measure")
    timetable = measure.startswith("With")
    px = "tt_" if timetable else ""
    cost = t["tt_labels_cpu_s"] if timetable else t["labels_cpu_s"]
    st.markdown(
        "Which single stop **blockages** hurt the network most? The exact answer means blocking each stop and "
        + ("re-routing every stop at every departure time with the timetable" if timetable else
           "recomputing every fastest path on the static graph")
        + f", about **{cost:.0f} CPU-seconds** for LBT. A random forest predicts it from cheap graph features in "
        f"about **{t['features_cpu_s']:.0f} seconds**. Every prediction below comes from a model that **never saw "
        "that stop's neighborhood** during training."
    )
    s = META["scores"]["tt" if timetable else "static"]
    m1, m2, m3 = st.columns(3)
    m1.metric("Spearman ρ", f"{s['spearman']:.2f}")
    m2.metric("Top-5% found", f"{100 * s['top5_overlap']:.0f}%")
    m3.metric("R²", f"{s['r2']:.2f}")

    n = st.slider("Show top N predicted", 10, 100, 25, step=5)
    top = PRED.nsmallest(n, f"{px}predicted_rank")
    st.dataframe(
        pd.DataFrame({
            "model rank": top[f"{px}predicted_rank"],
            "stop": top["name"],
            "predicted % lost": (100 * top[f"{px}predicted"]).round(2),
            "true % lost": (100 * top[f"{px}efficiency_loss"]).round(2),
            "true rank": top[f"{px}true_rank"],
            "why it matters": top[f"{px}type"],
        }),
        width="stretch", hide_index=True,
    )
    st.button(f"Block these {n} stops in the what-if map", on_click=set_scenario,
              kwargs={"blocked": top.index.tolist()})

with tab_about:
    st.markdown(f"""
**Data.** Long Beach Transit's public GTFS schedule for one weekday: {len(TT):,} scheduled stop-to-stop bus hops.

**Routing with waiting.** The what-if map uses the **Connection Scan Algorithm**. For a rider who shows up at a stop
at a given time, it scans every scheduled bus hop in departure order and finds the earliest arrival at every other stop,
including the wait for the first bus and for each transfer. Walking transfers up to 200 m (1.3 m/s) are allowed.
Each scenario re-routes from every stop at every 15 minutes in the chosen window: about 15,000 queries, compiled
with numba so it takes well under a second.

**Disruptions.**
- *Blocked* stops can't be passed, so trips are cut in two there.
- *Closed* stops are skipped by buses.
- *Canceled* routes are removed; *halved* routes lose every other trip; *late* routes shift every trip.

**Live map.** Buses follow the street shapes published in the GTFS feed. Each bus is placed along its shape by the
distance it has traveled, timed by the schedule. Click a bus for its route, destination, and upcoming stops; click a
stop for the next buses due there.

**Critical stops.** Blocking each stop, one at a time, gives the "true" damage, measured two ways:
- on the **timetable** (waiting counts): {META['timings']['tt_labels_cpu_s']:.0f} CPU-seconds for all stops
- on the **static graph** (one median travel time per stop pair, no waiting)

A random forest learns to predict that damage from 14 cheap graph features. Two of them do most of the work:
- *betweenness*: how many fastest paths pass through the stop
- a *dominator-tree* measure: which stops can only be reached through this one

On held-out neighborhoods, the model ranks stops with ρ ≈ {META['scores']['tt']['spearman']:.2f} on timetable damage
and ρ ≈ {META['scores']['static']['spearman']:.2f} on static damage. Waiting makes damage harder to predict, because it
depends on how often buses come, not just where the stops sit in the network.

**Limitations.**
- The schedule is assumed to run perfectly, apart from your scenario.
- Riders are spread evenly across stops and departure times, because ridership data isn't public.
- Fares aren't modeled.
- About 86% of bus times are interpolated by the agency.

Code and notebooks: [{REPO_URL}]({REPO_URL})
""")
