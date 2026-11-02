"""Export TransitShock results as flat CSVs for a Power BI dashboard (into powerbi/).

Usage (from the repo root, after scripts/run_pipeline.py --with-la):
    python scripts/export_powerbi.py

Inputs:
    app/data/lbt_*  and  app/data/la_*         (committed app data)
    data/processed/lbt_20261104_dataset.csv    (features + labels; local, from the pipeline)

Outputs (powerbi/):
    stops.csv          every stop in the combined LBT + LA Metro network
    stop_metrics.csv   LBT stops only (the critical-stop model is trained on LBT)
    route_impacts.csv  every route: damage if it were canceled (timetable routing, morning peak)
    summary.csv        one row per headline metric

Every value comes from the pipeline's own functions or files; nothing is filled in by hand.
Keys: stop_id is the node ID ("lbt:0002", "metro_bus:1"), route_id is "feed:route" ("metro_rail:801"),
so both are unique across agencies.
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from transitshock.app_data import read_graph_csv, read_timetable
from transitshock.metrics import evaluate, to_matrix
from transitshock.scenario import baseline, timetable_baseline, timetable_impact
from transitshock.simulate import greedy_blocking

ROOT = Path(__file__).resolve().parents[1]
APP_DATA = ROOT / "app" / "data"
DATASET = ROOT / "data" / "processed" / "lbt_20261104_dataset.csv"
OUT = ROOT / "powerbi"

AGENCIES = {"lbt": "Long Beach Transit", "metro_bus": "LA Metro Bus", "metro_rail": "LA Metro Rail"}
WINDOW = ("07:00", "09:00")   # morning peak, same as the app's default
LA_SAMPLED_ORIGINS = 300      # same sampling as the app's LA toggle


def export_stops(la_G) -> pd.DataFrame:
    """Every stop (rail platforms merged into stations) in the combined network."""
    return pd.DataFrame([{
        "stop_id": node,
        "stop_name": attrs["name"],
        "agency": AGENCIES[attrs["feed"]],
        "lat": attrs["lat"],
        "lon": attrs["lon"],
    } for node, attrs in la_G.nodes(data=True)])


def export_stop_metrics() -> pd.DataFrame:
    """LBT stops: timetable-based single-stop blockage damage, simulated vs. predicted (spatial CV).

    Impacts are % of network efficiency lost when the stop alone is blocked, morning peak,
    waiting counted. critical_rank ranks the simulated impact (1 = most damaging).
    """
    crit = pd.read_csv(APP_DATA / "lbt_critical.csv", index_col="node")
    betweenness = pd.read_csv(DATASET, index_col="node", usecols=["node", "betweenness"])["betweenness"]
    top5 = int(round(0.05 * len(crit)))
    return pd.DataFrame({
        "stop_id": crit.index,
        "predicted_impact": (100 * crit["tt_predicted"]).round(4).to_numpy(),
        "simulated_impact": (100 * crit["tt_efficiency_loss"]).round(4).to_numpy(),
        "betweenness": betweenness.reindex(crit.index).round(6).to_numpy(),
        "critical_rank": crit["tt_true_rank"].to_numpy(),
        "is_top5pct": (crit["tt_true_rank"] <= top5).to_numpy(),
    })


def route_modes(la_G) -> dict[str, str]:
    """"feed:route" -> "bus" / "rail", from the mode each route's segments carry (GTFS route_type)."""
    modes: dict[str, set] = {}
    for u, _, d in la_G.edges(data=True):
        if d["kind"] != "transit":
            continue
        feed = la_G.nodes[u]["feed"]
        for route in str(d["routes"]).split(","):
            modes.setdefault(f"{feed}:{route}", set()).update(str(d["modes"]).split(","))
    return {r: "rail" if "rail" in m else "bus" for r, m in modes.items()}


def export_route_impacts(la_G, la_base) -> pd.DataFrame:
    """Cancel each route on its own (whole LBT + LA Metro network, morning peak, timetable routing)."""
    names = la_G.graph["route_names"]
    modes = route_modes(la_G)
    routes = sorted(la_base.tt.events["route"].unique())  # routes with service that day
    rows = []
    for i, route in enumerate(routes, 1):
        summary, _ = timetable_impact(la_base, canceled_routes=[route])
        feed, rid = route.split(":", 1)
        rows.append({
            "route_id": route,
            "route_name": names[feed].get(rid, rid),
            "agency": AGENCIES[feed],
            "mode": modes.get(route, "bus"),
            "efficiency_loss_pct": round(-100 * summary["efficiency_change"], 4),
            "stops_cut_off": summary["stops_cut_off"],
        })
        if i % 25 == 0:
            print(f"  {i}/{len(routes)} routes")
    return pd.DataFrame(rows)


def export_summary(lbt_G, lbt_tt, la_G, la_tt, la_base) -> pd.DataFrame:
    """Headline metrics. Time windows and sampling are in the metric names."""
    meta = json.loads((APP_DATA / "lbt_meta.json").read_text())
    lbt_static = baseline(lbt_G).metrics
    lbt_am = timetable_baseline(lbt_tt, *WINDOW, every_min=15)
    _, la_matrix = to_matrix(la_G)
    la_sources = np.unique(la_base.sources)  # the same sampled origins as the timetable baseline
    la_static = evaluate(la_matrix, sources=la_sources)

    crit = pd.read_csv(APP_DATA / "lbt_critical.csv", index_col="node")
    candidates = list(dict.fromkeys(crit.nsmallest(40, "tt_predicted_rank").index.tolist()
                                    + crit.nsmallest(15, "predicted_rank").index.tolist()))
    greedy = greedy_blocking(lbt_tt, 10, candidates, *WINDOW, every_min=30)
    greedy_effect, _ = timetable_impact(lbt_am, blocked_stops=greedy["stop"].tolist())

    trips = lambda tt: tt.events["trip"].nunique()
    routes = lambda tt: tt.events["route"].nunique()
    metrics = {
        "lbt_stops": lbt_G.number_of_nodes(),
        "lbt_routes_in_service": routes(lbt_tt),
        "lbt_trips_per_weekday": trips(lbt_tt),
        "lbt_scheduled_stop_to_stop_hops": len(lbt_tt),
        "la_network_stops": la_G.number_of_nodes(),
        "la_network_routes_in_service": routes(la_tt),
        "la_network_trips_per_weekday": trips(la_tt),
        "la_network_scheduled_stop_to_stop_hops": len(la_tt),
        "lbt_avg_door_to_door_min_with_waiting_am_peak": lbt_am.metrics["mean_time_min"],
        "lbt_avg_trip_min_without_waiting_static": lbt_static["mean_time_min"],
        "lbt_reachable_share_within_3h_am_peak": lbt_am.metrics["reachable_share"],
        "la_avg_door_to_door_min_with_waiting_am_peak_sampled": la_base.metrics["mean_time_min"],
        "la_avg_trip_min_without_waiting_static_sampled": la_static["mean_time_min"],
        "la_reachable_share_within_3h_am_peak_sampled": la_base.metrics["reachable_share"],
        "la_sampled_origin_stops": len(la_sources),
        "model_timetable_spearman": meta["scores"]["tt"]["spearman"],
        "model_timetable_top5pct_found": meta["scores"]["tt"]["top5_overlap"],
        "model_timetable_r2": meta["scores"]["tt"]["r2"],
        "model_static_spearman": meta["scores"]["static"]["spearman"],
        "model_static_top5pct_found": meta["scores"]["static"]["top5_overlap"],
        "model_static_r2": meta["scores"]["static"]["r2"],
        "exhaustive_static_labels_cpu_s": meta["timings"]["labels_cpu_s"],
        "exhaustive_timetable_labels_cpu_s": meta["timings"]["tt_labels_cpu_s"],
        "model_features_cpu_s": meta["timings"]["features_cpu_s"],
        "greedy_worst10_efficiency_loss_pct_am_peak": -100 * greedy_effect["efficiency_change"],
        "greedy_worst10_stops_cut_off_am_peak": greedy_effect["stops_cut_off"],
        "greedy_worst10_scenarios_tested": int(greedy["evaluations"].iloc[-1]),
    }
    # Counts stay whole numbers (1853, not 1853.0); everything else is rounded to 4 decimals.
    value = lambda v: int(v) if float(v).is_integer() else round(float(v), 4)
    # dtype=object stops pandas from turning the whole column into floats.
    return pd.DataFrame({"metric": list(metrics),
                         "value": pd.Series([value(v) for v in metrics.values()], dtype=object)})


def main() -> None:
    OUT.mkdir(exist_ok=True)
    t0 = time.perf_counter()
    lbt_G = read_graph_csv(APP_DATA, "lbt")
    lbt_tt = read_timetable(APP_DATA, "lbt", lbt_G)
    la_G = read_graph_csv(APP_DATA, "la")
    la_tt = read_timetable(APP_DATA, "la", la_G)
    la_base = timetable_baseline(la_tt, *WINDOW, every_min=30, max_sources=LA_SAMPLED_ORIGINS)

    outputs = {"stops.csv": export_stops(la_G), "stop_metrics.csv": export_stop_metrics()}
    print("Canceling each route (about 1 s per route)...")
    outputs["route_impacts.csv"] = export_route_impacts(la_G, la_base)
    print("Summary metrics (includes a ~15 s greedy search)...")
    outputs["summary.csv"] = export_summary(lbt_G, lbt_tt, la_G, la_tt, la_base)

    for name, df in outputs.items():
        df.to_csv(OUT / name, index=False)
        print(f"powerbi/{name}: {len(df):,} rows, {len(df.columns)} columns")
    print(f"Done in {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
