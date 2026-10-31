# TransitShock

**What happens to a city's bus network when stops or routes go down, and can a model predict which stops matter most without simulating every failure?**

TransitShock turns Long Beach Transit's public schedule (GTFS) into a weighted graph of 1,853 stops. It simulates closures and trains a model that ranks how damaging each stop's closure would be. A Streamlit app lets you close stops or routes and see who gets cut off.

**Live demo:** _add your Streamlit Cloud link here_

![TransitShock demo: closing the model's top 10 critical stops cuts off 44 stops](figures/app_screenshot.jpg)

## Key results

| Finding | Number |
|---|---|
| Closing the **10 busiest** stops | −1.4% network efficiency, 0 stops cut off |
| Closing the model's **10 most critical** stops | −4.7% network efficiency, **44 stops cut off** |
| Model vs. betweenness alone at ranking stops (held-out neighborhoods) | Spearman ρ **0.88** vs. 0.80 |
| True top-5% critical stops the model finds | **65%** (betweenness alone: 47%) |
| Compute vs. exhaustive simulation | **≈58× less** (≈10 s vs. 573 CPU-s) |
| Random failures vs. targeting busy stops | Random failures halve efficiency after removing 10% of stops; targeting busy stops takes 20% |

**Takeaways:**
- **Busy stops aren't the critical ones.** Trips per day has essentially zero correlation with how much a closure hurts the network.
- **Bus routes are chains.** 85% of consecutive stops are farther apart than a 200 m walk, so losing one stop often cuts its route. That's why random failures do more damage than targeting busy stops.
- **The model's biggest gain came from looking at its mistakes.** It was missing stops that strand other stops. A feature built on **dominator trees** (a compiler algorithm) measures stranding exactly, and adding it raised the critical stops found from 51% to 65%.

## How it works

```
GTFS feeds ──► load_gtfs ──► build_graph ──► metrics / simulate ──► features ──► model ──► app
(schedule)     one weekday    stops + edges    what breaks?          per-stop     predict    what-if
                              (travel time,                          clues        damage     demo
                               walking ≤200 m)
```

| Step | What it does |
|---|---|
| **Graph** | Stops are nodes. Consecutive stops on a trip are linked by directed edges weighted by median scheduled travel time. Stops within 200 m are linked by walking edges. Rail platforms are merged into their stations. |
| **Health metric** | Network *efficiency*: the average of 1 / travel time over all stop pairs. All-pairs shortest paths use scipy's C-based Dijkstra, about 0.3 s per network. |
| **Simulation** | Random vs. targeted attacks (degree, trips/day, betweenness; static and adaptive), single-route outages, and removing each of the 1,853 stops one at a time, run in parallel across CPU cores. |
| **Model** | A random forest predicts the log of single-stop damage from 14 graph features. It's evaluated with **spatial cross-validation**, holding out whole neighborhoods, so neighboring stops can't leak between training and testing. |

## Repo layout

```
src/transitshock/   load_gtfs, build_graph, metrics, simulate, features, model, scenario, app_data
app/                streamlit_app.py + data/ (small CSV/JSON export the demo runs on)
notebooks/          01 explore GTFS · 02 graph stats · 03 robustness · 04 model
notes/              gtfs_notes.md (data quirks) · code_walkthrough.md (module-by-module guide)
scripts/            download_data.sh · run_pipeline.py
tests/              53 pytest tests on small hand-built networks + app smoke tests
```

## Run it

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .

streamlit run app/streamlit_app.py      # demo (uses the committed app/data/)
pytest                                  # tests
```

To rebuild everything from the raw schedules (about 1.5 minutes on 10 cores):

```bash
./scripts/download_data.sh
python scripts/run_pipeline.py --feed lbt --date 20261104
```

## Limitations

- **No transfer waiting time.** Switching routes adds no wait, so travel times are underestimated.
- **Fares aren't modeled.** Travel time is the only cost.
- **One weekday of scheduled service.** The agency interpolates about 86% of bus times, and there's no real-time data.
- **The model is trained and evaluated on one agency (LBT).** The next test is whether it transfers to LA Metro, which is 6× larger.

## Built with

Python · pandas · NetworkX · SciPy (sparse graphs, Dijkstra) · scikit-learn · Streamlit · pydeck · pytest · GitHub Actions
