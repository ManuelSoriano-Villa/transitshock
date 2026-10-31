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
import streamlit.components.v1 as components

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # import the package without installing it (for hosting)

from transitshock.animation import buses_running, hhmm, summarize_paths, trip_paths  # noqa: E402
from transitshock.app_data import read_graph_csv, read_timetable  # noqa: E402
from transitshock.scenario import baseline, impact, timetable_baseline, timetable_impact  # noqa: E402
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


# ---------- sidebar: build a scenario ----------

SCENARIO_KEYS = ["blocked", "closed", "canceled", "reduced", "delayed"]


def set_scenario(**values):
    for key in SCENARIO_KEYS:
        st.session_state[key] = list(values.get(key, []))


for key in SCENARIO_KEYS:
    st.session_state.setdefault(key, [])

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
    top10 = PRED.nsmallest(10, "predicted_rank").index.tolist()
    busiest = PRED.nlargest(10, "trips_per_day").index.tolist()
    st.button("Block the model's top 10 critical stops", key="preset_top10", on_click=set_scenario,
              kwargs={"blocked": top10}, width="stretch")
    st.button("Block the 10 busiest stops", key="preset_busy", on_click=set_scenario,
              kwargs={"blocked": busiest}, width="stretch")
    st.button("Close the 10 busiest stops (buses skip)", on_click=set_scenario,
              kwargs={"closed": busiest}, width="stretch")
    st.button("Driver shortage: halve the 5 busiest routes", on_click=set_scenario,
              kwargs={"reduced": TT.events.groupby("route")["trip"].nunique().nlargest(5).index.tolist()},
              width="stretch")
    st.button("Clear", key="preset_clear", on_click=set_scenario, width="stretch")

    st.caption("Long Beach Transit, weekday schedule for Wed 2026-11-04. "
               f"{G.number_of_nodes():,} stops, {len(ROUTE_NAMES)} routes, {len(TT):,} scheduled bus hops.")

scenario = {key: st.session_state[key] for key in SCENARIO_KEYS}
delays = {r: delay for r in scenario["delayed"]}
nothing_selected = not any(scenario.values())


# ---------- main ----------

st.title("🚌 TransitShock")
st.markdown("What happens to **Long Beach Transit** when stops are blocked, routes are cut, or buses run late? "
            "Build a scenario in the sidebar.")

tab_whatif, tab_live, tab_critical, tab_about = st.tabs(["What-if map", "Live buses", "Critical stops", "How it works"])

with tab_whatif:
    base = load_timetable_baseline(window)
    with st.spinner("Re-routing every stop at every departure time..."):
        summary, per_stop = timetable_impact(
            base, stops=scenario["closed"], canceled_routes=scenario["canceled"],
            reduced_routes=scenario["reduced"], delays_min=delays, blocked_stops=scenario["blocked"],
        )
    per_stop = per_stop.assign(name=[G.nodes[n]["name"] for n in per_stop.index],
                               lat=[G.nodes[n]["lat"] for n in per_stop.index],
                               lon=[G.nodes[n]["lon"] for n in per_stop.index])

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
    c4, c5, _ = st.columns(3)
    c4.metric("Stops cut off", summary["stops_cut_off"],
              help="Can no longer reach at least 10% of the stops they could before.")
    c5.metric("Slower stops", summary["stops_slower"],
              help="Trips from here take 2+ minutes longer on average.")

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

    line = lambda data, color, width: pdk.Layer("LineLayer", data, get_source_position="from",
                                                get_target_position="to", get_color=color, get_width=width)
    layers = [
        line(EDGES[route_state == "normal"], [120, 120, 120, 90], 1),
        line(EDGES[route_state == "weaker"], [30, 120, 220, 220], 3),
        line(EDGES[route_state == "canceled"], [128, 0, 160, 220], 3),
        pdk.Layer("ScatterplotLayer", stops, get_position=["lon", "lat"], get_fill_color="color",
                  get_radius="radius", radius_min_pixels=2, pickable=True),
    ]
    view = pdk.ViewState(latitude=float(stops["lat"].median()), longitude=float(stops["lon"].median()), zoom=10.5)
    tooltip = {"html": "<b>{name}</b><br/>{status}<br/>destinations lost: {lost_destinations}"
                       "<br/>extra minutes: {added}"}
    st.pydeck_chart(pdk.Deck(layers=layers, initial_view_state=view, tooltip=tooltip), height=540)
    st.caption("⚫ blocked/closed  🔴 cut off  🟠 slower  ⚪ unaffected  "
               "🟪 canceled route segments  🟦 routes running less often or late")

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
    start = st.select_slider("Start time", options=[f"{h:02d}:00" for h in range(5, 24)],
                             value=WINDOWS[window][0], key="live_start")
    speed = st.radio("Speed", [30, 60, 120], index=1, horizontal=True, format_func=lambda s: f"{s}×", key="speed")
    start_s = int(start[:2]) * 3600
    live_tt = disrupt(TT, scenario["closed"], scenario["canceled"], scenario["reduced"], delays, scenario["blocked"])
    paths = trip_paths(live_tt, LAT, LON, start_s, start_s + 3600, highlight_routes=set(scenario["delayed"]))
    marked = [{"position": [float(LON[TT.stops.index(n)]), float(LAT[TT.stops.index(n)])],
               "kind": kind} for kind in ("blocked", "closed") for n in scenario[kind]]
    html = (APP_DIR / "live_buses.html").read_text()
    html = (html.replace("__TRIPS__", json.dumps(paths)).replace("__MARKED__", json.dumps(marked))
                .replace("__START__", str(start_s)).replace("__SPEED__", str(speed))
                .replace("__CENTER__", json.dumps([float(np.median(LON)), float(np.median(LAT))])))
    components.html(html, height=560)
    shown = summarize_paths(paths)
    st.caption(f"One hour of the {'disrupted' if not nothing_selected else 'normal'} schedule from {start}, "
               f"played at {speed}×: {shown['trips shown']} trips on {shown['routes']} routes "
               f"({buses_running(paths, 1800)} buses on the road at {hhmm(start_s + 1800)}). "
               "Buses move in straight lines between stops. Red trails are routes running late; "
               "black dots are blocked stops, gray dots are closed ones.")

with tab_critical:
    t = META["timings"]
    st.markdown(
        "Which single stop **blockages** hurt the network most? The exact answer means blocking each stop and "
        f"recomputing every fastest path, about **{t['labels_cpu_s']:.0f} CPU-seconds** for LBT. The model predicts "
        f"it from cheap graph features in about **{t['features_cpu_s']:.0f} seconds**. Every prediction below comes "
        "from a model that **never saw that stop's neighborhood** during training. (The model is trained on the "
        "static network, without waiting times.)"
    )
    s = META["scores"]
    m1, m2, m3 = st.columns(3)
    m1.metric("Spearman ρ", f"{s['spearman']:.2f}")
    m2.metric("Top-5% found", f"{100 * s['top5_overlap']:.0f}%")
    m3.metric("R²", f"{s['r2']:.2f}")

    n = st.slider("Show top N predicted", 10, 100, 25, step=5)
    top = PRED.nsmallest(n, "predicted_rank")
    st.dataframe(
        pd.DataFrame({
            "model rank": top["predicted_rank"],
            "stop": top["name"],
            "predicted % lost": (100 * top["predicted"]).round(2),
            "true % lost": (100 * top[TARGET]).round(2),
            "true rank": top["true_rank"],
            "why it matters": top["type"],
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

**Static model and critical stops.** The original model gives each stop pair one median travel time and doesn't wait
for buses. It's fast and good at finding stops whose *blockage* strands others: it identifies the same 44 cut-off stops
as the timetable. But it underestimates travel times by about half (28.6 vs. 56.6 minutes on average in the morning
peak). A random forest trained on it ranks critical stops with ρ ≈ 0.88 on held-out neighborhoods. Its two most useful
features are *betweenness* and a *dominator-tree* measure of which stops can only be reached through a given stop.

**Limitations.**
- The schedule is assumed to run perfectly, apart from your scenario.
- Riders are spread evenly across stops and departure times, because ridership data isn't public.
- Fares aren't modeled.
- About 86% of bus times are interpolated by the agency.

Code and notebooks: [{REPO_URL}]({REPO_URL})
""")
