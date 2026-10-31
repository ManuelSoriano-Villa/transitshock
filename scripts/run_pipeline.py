"""Run the whole pipeline: raw GTFS -> graph -> features + labels -> app data.

Usage (from the repo root, after scripts/download_data.sh):
    python scripts/run_pipeline.py                     # LBT, Wed 2026-11-04
    python scripts/run_pipeline.py --feed lbt --date 20261104

Each step writes to data/processed/ and the last one to app/data/.
"""
import argparse
import time

import pandas as pd

from transitshock.app_data import export_app_data, export_network
from transitshock.build_graph import ROOT, build_graph, graph_summary, save_graph
from transitshock.features import build_dataset
from transitshock.load_gtfs import load_feed
from transitshock.timetable import build_timetable


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--feed", default="lbt", help="feed folder in data/raw/")
    parser.add_argument("--date", default="20261104", help="service date, YYYYMMDD")
    parser.add_argument("--workers", type=int, default=None, help="processes for the simulation (default: all cores)")
    parser.add_argument("--with-la", action="store_true",
                        help="also export the combined LBT + LA Metro bus & rail network for the app's toggle "
                             "(needs data/raw/metro_bus and data/raw/metro_rail)")
    args = parser.parse_args()

    processed = ROOT / "data" / "processed"
    graph_path = processed / f"{args.feed}_{args.date}.gpickle"

    def step(name):
        print(f"\n== {name}")
        return time.perf_counter()

    t = step(f"1/3 Build graph ({args.feed}, {args.date})")
    data = load_feed(ROOT / "data" / "raw" / args.feed, args.date)
    G = build_graph({args.feed: data})
    tt = build_timetable(data, args.feed, G)
    save_graph(G, graph_path)
    print(graph_summary(G).to_string())
    print(f"({time.perf_counter() - t:.1f}s) -> {graph_path.relative_to(ROOT)}")

    t = step("2/3 Features + exhaustive single-stop damage, static and timetable (the slow part)")
    dataset, timings = build_dataset(G, workers=args.workers, tt=tt)
    dataset.to_csv(graph_path.with_name(graph_path.stem + "_dataset.csv"))
    pd.Series(timings).to_csv(graph_path.with_name(graph_path.stem + "_timings.csv"), header=["seconds"])
    print(f"({time.perf_counter() - t:.1f}s) {len(dataset)} stops")

    t = step("3/3 Export app data (timetable + model predictions with spatial CV)")
    export_app_data(graph_path, prefix=args.feed, tt=tt)
    print(f"({time.perf_counter() - t:.1f}s) -> app/data/{args.feed}_*")

    if args.with_la:
        t = step("Extra: LBT + LA Metro bus & rail network for the app")
        export_network([args.feed, "metro_bus", "metro_rail"], args.date, "la")
        print(f"({time.perf_counter() - t:.1f}s) -> app/data/la_*")


if __name__ == "__main__":
    main()
