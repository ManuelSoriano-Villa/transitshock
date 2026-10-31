"""TransitShock demo: close stops or routes on the Long Beach Transit network and see who's affected.

Run from the repo root:
    streamlit run app/streamlit_app.py

Needs the graph and model dataset in data/processed/ (see README for the build commands).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

from transitshock.build_graph import load_graph
from transitshock.model import MODELS, TARGET, out_of_fold_predictions, score, spatial_groups
from transitshock.scenario import baseline, impact

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
GRAPH_PATH = DATA / "lbt_20261104.gpickle"
DATASET_PATH = DATA / "lbt_20261104_dataset.csv"
REPO_URL = "https://github.com/ManuelSoriano-Villa/transitshock"

STATUS_COLORS = {
    "closed": [20, 20, 20],
    "cut off": [214, 32, 40],
    "slower": [245, 150, 30],
    "unaffected": [150, 150, 150],
}
STATUS_RADIUS = {"closed": 90, "cut off": 70, "slower": 60, "unaffected": 30}

st.set_page_config(page_title="TransitShock", page_icon="🚌", layout="wide")


# ---------- data (cached so it loads once per session) ----------

@st.cache_resource
def load_network():
    G = load_graph(GRAPH_PATH)
    return G, baseline(G)


@st.cache_data
def load_edges(_G) -> pd.DataFrame:
    rows = []
    for u, v, d in _G.edges(data=True):
        if d["kind"] == "transit":
            a, b = _G.nodes[u], _G.nodes[v]
            rows.append({"from": [a["lon"], a["lat"]], "to": [b["lon"], b["lat"]],
                         "routes": d["routes"].split(","), "trips": d["trips"]})
    return pd.DataFrame(rows)


@st.cache_data
def load_predictions() -> pd.DataFrame:
    """Model predictions for every stop, each made by a model that never saw that stop's neighborhood."""
    df = pd.read_csv(DATASET_PATH, index_col="node")
    df["predicted"] = out_of_fold_predictions(df, MODELS["random_forest"], groups=spatial_groups(df))
    df["true_rank"] = df[TARGET].rank(ascending=False).astype(int)
    df["predicted_rank"] = df["predicted"].rank(ascending=False, method="first").astype(int)
    df["type"] = np.where(df["reachable_loss"] > 2 / len(df) + 1e-9, "cuts off stops", "slows trips")
    return df


if not GRAPH_PATH.exists() or not DATASET_PATH.exists():
    st.error(
        "Missing data. From the repo root, run:\n\n"
        "```\npython -m transitshock.build_graph lbt --date 20261104\n"
        "python -m transitshock.features data/processed/lbt_20261104.gpickle\n```"
    )
    st.stop()

G, BASE = load_network()
EDGES = load_edges(G)
PRED = load_predictions()
ROUTE_NAMES = G.graph["route_names"]["lbt"]
STOP_LABELS = {n: f"{G.nodes[n]['name']} ({n.split(':')[1]})" for n in sorted(G, key=lambda n: G.nodes[n]["name"])}


# ---------- sidebar: build a scenario ----------

def set_scenario(stops=(), routes=()):
    st.session_state["stops"] = list(stops)
    st.session_state["routes"] = list(routes)


st.session_state.setdefault("stops", [])
st.session_state.setdefault("routes", [])

with st.sidebar:
    st.header("Build a scenario")
    st.multiselect("Close stops", options=list(STOP_LABELS), format_func=STOP_LABELS.get, key="stops",
                   help="Closed stops can't be used to board, alight, or pass through.")
    st.multiselect("Stop running routes", options=sorted(ROUTE_NAMES, key=lambda r: (len(r), r)),
                   format_func=ROUTE_NAMES.get, key="routes",
                   help="Segments only this route serves disappear. Its stops stay open for other routes and walking.")

    st.subheader("Presets")
    top10 = PRED.nsmallest(10, "predicted_rank").index.tolist()
    st.button("Close the model's top 10 critical stops", key="preset_top10", on_click=set_scenario, kwargs={"stops": top10},
              width="stretch")
    busiest = PRED.nlargest(10, "trips_per_day").index.tolist()
    st.button("Close the 10 busiest stops", on_click=set_scenario, kwargs={"stops": busiest},
              width="stretch")
    random5 = [str(n) for n in np.random.default_rng(7).choice(list(G), size=int(0.05 * len(G)), replace=False)]
    st.button("Random failure: 5% of stops", on_click=set_scenario, kwargs={"stops": random5},
              width="stretch")
    st.button("Clear", on_click=set_scenario, width="stretch")

    st.caption("Long Beach Transit, weekday service on Wed 2026-11-04. "
               f"{G.number_of_nodes():,} stops, {len(ROUTE_NAMES)} routes.")


# ---------- main ----------

st.title("🚌 TransitShock")
st.markdown("What happens to **Long Beach Transit** when stops or routes go down? "
            "Pick a scenario in the sidebar and see who gets cut off.")

tab_whatif, tab_critical, tab_about = st.tabs(["What-if map", "Critical stops", "How it works"])

with tab_whatif:
    with st.spinner("Recomputing every fastest path..."):
        summary, per_stop = impact(G, BASE, st.session_state["stops"], st.session_state["routes"])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Efficiency", f"{100 * (1 + summary['efficiency_change']):.1f}%",
              f"{100 * summary['efficiency_change']:.2f}%",
              help="Average of 1 / travel time over all stop pairs, relative to the intact network.")
    c2.metric("Trips lost", f"{100 * summary['pairs_lost']:.2f}%",
              help="Share of all stop-to-stop trips that can no longer be made.")
    c3.metric("Stops cut off", summary["stops_cut_off"],
              help="Can no longer reach at least 10% of the stops they could before.")
    c4.metric("Slower stops", summary["stops_slower"],
              help="Trips to still-reachable stops take 2+ minutes longer on average.")

    closed_routes = set(st.session_state["routes"])
    edges = EDGES.assign(closed=EDGES["routes"].apply(lambda rs: set(rs) <= closed_routes if closed_routes else False))
    stops = per_stop.reset_index().assign(
        color=lambda d: d["status"].map(STATUS_COLORS),
        radius=lambda d: d["status"].map(STATUS_RADIUS),
        added=lambda d: d["added_min"].round(1),
    ).sort_values("status", key=lambda s: s.map({"unaffected": 0, "slower": 1, "cut off": 2, "closed": 3}))

    layers = [
        pdk.Layer("LineLayer", edges[~edges["closed"]], get_source_position="from", get_target_position="to",
                  get_color=[120, 120, 120, 90], get_width=1),
        pdk.Layer("LineLayer", edges[edges["closed"]], get_source_position="from", get_target_position="to",
                  get_color=[128, 0, 160, 220], get_width=3),
        pdk.Layer("ScatterplotLayer", stops, get_position=["lon", "lat"], get_fill_color="color",
                  get_radius="radius", radius_min_pixels=2, pickable=True),
    ]
    view = pdk.ViewState(latitude=float(stops["lat"].median()), longitude=float(stops["lon"].median()), zoom=10.5)
    tooltip = {"html": "<b>{name}</b><br/>{status}<br/>lost destinations: {lost_destinations}"
                       "<br/>added minutes: {added}"}
    st.pydeck_chart(pdk.Deck(layers=layers, initial_view_state=view, tooltip=tooltip), height=560)
    st.caption("⚫ closed  🔴 cut off  🟠 slower  ⚪ unaffected  🟪 route segments no longer running")

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
    elif not st.session_state["stops"] and not st.session_state["routes"]:
        st.info("Nothing closed yet. Try a preset in the sidebar.")

with tab_critical:
    st.markdown(
        "Which single stop closures hurt the network most? The exact answer means closing each stop and "
        "recomputing every fastest path, about **570 CPU-seconds** for LBT. The model predicts it from cheap "
        "graph features in about **9 seconds**. Every prediction below comes from a model that **never saw that "
        "stop's neighborhood** during training."
    )
    s = score(PRED[TARGET], PRED["predicted"])
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
    st.button(f"Close these {n} stops in the what-if map", on_click=set_scenario,
              kwargs={"stops": top.index.tolist()})

with tab_about:
    st.markdown(f"""
**Data.** Long Beach Transit's public GTFS schedule, filtered to one weekday.

**Graph.** Every served stop is a node. Consecutive stops on a trip are linked by a directed edge weighted by the
median scheduled travel time. Stops within 200 m are linked by walking edges (1.3 m/s).

**What-if.** Closing stops or routes removes them from the graph. Every fastest path is then recomputed
(scipy's Dijkstra, about 0.5 s) and compared with the intact network.

**Model.** A random forest predicts how much efficiency a single stop's closure costs, using 14 graph features.
Two features do most of the work:
- *betweenness*: how many fastest paths go through the stop
- *stranded share*: computed with dominator trees, it measures which stops can only be reached through this one

On held-out neighborhoods (spatial cross-validation) it ranks stops with ρ ≈ 0.88 and finds about 65% of the true
top-5% critical stops. Betweenness alone gets ρ ≈ 0.80 and 47%.

**Limitations.**
- Transfers have no waiting-time penalty.
- Fares aren't modeled.
- One weekday of scheduled (not real-time) service.
- About 86% of bus times are interpolated by the agency.

Code and notebooks: [{REPO_URL}]({REPO_URL})
""")
