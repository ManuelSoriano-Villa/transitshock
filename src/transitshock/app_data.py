"""Small, committed data files for the hosted demo.

The full pipeline needs the raw GTFS feeds (not committed) and takes minutes
to run. The demo only needs the finished graph and the model's predictions, so
this module writes those as plain CSV/JSON into app/data/ (well under 1 MB)
and reads them back without scikit-learn or the raw data.

Export (run after build_graph and features):
    python -m transitshock.app_data data/processed/lbt_20261104.gpickle --date 20261104
"""
import argparse
import json
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from transitshock.build_graph import load_graph
from transitshock.timetable import Timetable, timetable_from_events

ROOT = Path(__file__).resolve().parents[2]
APP_DATA = ROOT / "app" / "data"


def write_graph_csv(G: nx.DiGraph, out_dir: Path, prefix: str) -> None:
    """Graph -> <prefix>_nodes.csv, <prefix>_edges.csv, <prefix>_graph.json."""
    out_dir.mkdir(parents=True, exist_ok=True)
    nodes = pd.DataFrame.from_dict(dict(G.nodes(data=True)), orient="index")
    nodes.index.name = "node"
    nodes.to_csv(out_dir / f"{prefix}_nodes.csv")
    edges = pd.DataFrame([{"u": u, "v": v, **d} for u, v, d in G.edges(data=True)])
    edges.to_csv(out_dir / f"{prefix}_edges.csv", index=False)
    (out_dir / f"{prefix}_graph.json").write_text(json.dumps(G.graph, indent=1))


def read_graph_csv(out_dir: Path, prefix: str) -> nx.DiGraph:
    """Inverse of write_graph_csv. Empty cells (e.g. distance_m on transit edges) are skipped."""
    G = nx.DiGraph(**json.loads((out_dir / f"{prefix}_graph.json").read_text()))
    nodes = pd.read_csv(out_dir / f"{prefix}_nodes.csv", index_col="node", dtype={"routes": str},
                        float_precision="round_trip")
    for node, row in nodes.iterrows():
        G.add_node(node, **row.dropna().to_dict())
    edges = pd.read_csv(out_dir / f"{prefix}_edges.csv", dtype={"routes": str}, float_precision="round_trip")
    for row in edges.to_dict("records"):
        u, v = row.pop("u"), row.pop("v")
        G.add_edge(u, v, **{k: val for k, val in row.items() if not pd.isna(val)})
    return G


def write_timetable(tt: Timetable, out_dir: Path, prefix: str) -> None:
    """Timetable -> <prefix>_events.csv.gz (stops by node ID, so the file doesn't depend on node order),
    plus <prefix>_trips.csv and <prefix>_shapes.csv.gz when the timetable has street shapes."""
    events = tt.events.assign(stop=[tt.stops[i] for i in tt.events["stop"]])
    events.to_csv(out_dir / f"{prefix}_events.csv.gz", index=False)
    if tt.trip_info is not None:
        tt.trip_info.to_csv(out_dir / f"{prefix}_trips.csv")
    if tt.shapes:
        rows = [pd.DataFrame({"shape_id": sid, "dist": d, "lat": la, "lon": lo})
                for sid, (d, la, lo) in tt.shapes.items()]
        pd.concat(rows).to_csv(out_dir / f"{prefix}_shapes.csv.gz", index=False, float_format="%.6f")


def read_timetable(out_dir: Path, prefix: str, G: nx.DiGraph) -> Timetable:
    """Inverse of write_timetable; walking transfers are rebuilt from G's coordinates."""
    events = pd.read_csv(out_dir / f"{prefix}_events.csv.gz", dtype={"route": str})
    index = {node: i for i, node in enumerate(G.nodes)}
    events["stop"] = events["stop"].map(index).astype(np.int32)
    trip_info = shapes = None
    if (out_dir / f"{prefix}_trips.csv").exists():
        trip_info = pd.read_csv(out_dir / f"{prefix}_trips.csv", index_col="trip",
                                dtype={"shape_id": str, "headsign": str})
    if (out_dir / f"{prefix}_shapes.csv.gz").exists():
        sh = pd.read_csv(out_dir / f"{prefix}_shapes.csv.gz", dtype={"shape_id": str})
        shapes = {sid: (g["dist"].to_numpy(), g["lat"].to_numpy(), g["lon"].to_numpy())
                  for sid, g in sh.groupby("shape_id", sort=False)}
    return timetable_from_events(events.astype({"trip": np.int32, "seq": np.int32,
                                                "arr_s": np.int32, "dep_s": np.int32}), G,
                                 trip_info=trip_info, shapes=shapes)


def critical_table(dataset: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Out-of-fold (spatial CV) predictions for every stop, plus the overall scores.

    Always includes the static model (efficiency_loss). If the dataset has timetable
    labels (tt_efficiency_loss: blockage damage with waiting counted), a second model
    is trained on those, and its columns are prefixed tt_.
    """
    from transitshock.model import MODELS, out_of_fold_predictions, score, spatial_groups

    groups = spatial_groups(dataset)
    own_pairs = 2 / len(dataset)
    table, scores = pd.DataFrame(index=dataset.index), {}
    labels = [("", "efficiency_loss", "reachable_loss"), ("tt_", "tt_efficiency_loss", "tt_reachable_loss")]
    for prefix, target, reachable in labels:
        if target not in dataset:
            continue
        pred = out_of_fold_predictions(dataset, MODELS["random_forest"], target=target, groups=groups)
        table[f"{prefix}efficiency_loss"] = dataset[target]
        table[f"{prefix}reachable_loss"] = dataset[reachable]
        table[f"{prefix}predicted"] = pred
        table[f"{prefix}true_rank"] = dataset[target].rank(ascending=False, method="first").astype(int)
        table[f"{prefix}predicted_rank"] = pd.Series(pred, index=dataset.index).rank(
            ascending=False, method="first").astype(int)
        table[f"{prefix}type"] = np.where(dataset[reachable] > own_pairs + 1e-9, "cuts off stops", "slows trips")
        scores[prefix.rstrip("_") or "static"] = {k: float(v) for k, v in score(dataset[target], pred).items()}
    return table, scores


def export_app_data(graph_path: Path, out_dir: Path = APP_DATA, prefix: str = "lbt", tt: Timetable | None = None) -> None:
    """Write everything the app needs. Pass `tt` (build_timetable output) to include the timetable."""
    graph_path = Path(graph_path)
    G = load_graph(graph_path)
    dataset = pd.read_csv(graph_path.with_name(graph_path.stem + "_dataset.csv"), index_col="node")
    timings = pd.read_csv(graph_path.with_name(graph_path.stem + "_timings.csv"), index_col=0)["seconds"]

    write_graph_csv(G, out_dir, prefix)
    if tt is not None:
        write_timetable(tt, out_dir, prefix)
    table, scores = critical_table(dataset)
    table.to_csv(out_dir / f"{prefix}_critical.csv")
    meta = {"scores": scores, "timings": timings.to_dict(), "source_graph": graph_path.name}
    (out_dir / f"{prefix}_meta.json").write_text(json.dumps(meta, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the demo's data files to app/data/.")
    parser.add_argument("graph", help="path to a .gpickle (its _dataset.csv and _timings.csv must exist)")
    parser.add_argument("--prefix", default="lbt", help="feed name (also the data/raw/ folder for --date)")
    parser.add_argument("--date", help="YYYYMMDD: also export the timetable, built from data/raw/<prefix>")
    args = parser.parse_args()
    tt = None
    if args.date:
        from transitshock.load_gtfs import load_feed
        from transitshock.timetable import build_timetable
        tt = build_timetable(load_feed(ROOT / "data" / "raw" / args.prefix, args.date), args.prefix,
                             load_graph(args.graph))
    export_app_data(Path(args.graph), prefix=args.prefix, tt=tt)
    for f in sorted(APP_DATA.glob(f"{args.prefix}_*")):
        print(f"{f.relative_to(ROOT)}  {f.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
