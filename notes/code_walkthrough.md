# Code Walkthrough

How the `transitshock` package works, module by module. Read it alongside the code in `src/transitshock/`.

## The pipeline at a glance

```
GTFS files (data/raw/lbt/*.txt)
   │  load_gtfs.load_feed(feed_dir, date)
   ▼
Clean tables for one weekday: stops, routes, trips, stop_times
   │  build_graph.build_graph({feed: tables})
   ▼
Directed weighted graph (networkx DiGraph), saved to data/processed/*.gpickle
   │
   ├── metrics.evaluate(...)         → "how healthy is this network?"
   ├── simulate.attack_curve(...)    → remove many stops, watch it degrade        (Week 3)
   ├── simulate.route_outages(...)   → shut down one route at a time            (Week 3)
   └── simulate.single_removal_damage(...) → true damage of each stop (slow)    (Week 4 labels)
            │
            │  features.node_features(G) → cheap per-stop numbers (Week 4 inputs)
            ▼
       model.compare(...) / model.fit(...) → predict damage without simulating  (Week 4)
```

Every module does one job and hands a plain data structure (a DataFrame, a graph, or a dict) to the next. That's why each one can be tested on its own with tiny fake inputs.

---

## 1. `load_gtfs.py`: raw files → clean tables

**Job:** read the GTFS text files and keep only the trips that run on one chosen date.

| Function | What it does |
|---|---|
| `read_table(feed_dir, name)` | Reads one file as **all strings**, strips whitespace, keeps only the columns listed in `COLUMNS`, and adds missing optional columns as empty. |
| `time_to_seconds(series)` | `"25:15:00"` → `90900`. A manual split on `:` because GTFS times can exceed 24:00. |
| `active_service_ids(calendar, calendar_dates, date)` | Applies the GTFS calendar rule: weekly patterns from `calendar.txt`, then add (`1`) or remove (`2`) exceptions from `calendar_dates.txt`. |
| `clean_stops`, `clean_stop_times` | Convert the text columns that should be numbers (lat/lon, `stop_sequence`, times) and sort stop_times by trip and sequence. |
| `load_feed(feed_dir, date)` | Runs the steps in order: find active services → keep those trips → keep those trips' stop_times → clean. Raises `ValueError` if nothing runs that day. |

**Why read everything as strings?** It fixes three of the feed quirks from `gtfs_notes.md` in one place:
- LBT's leading-zero IDs (`0002`) survive.
- Metro's padded values (`"  33.97"`, `"1 "`) get stripped.
- Comparisons like `exception_type == "1"` behave predictably.

Numbers are converted only where needed.

**Interview angle:** "Real-world data is messy. I profiled the feeds first, wrote down the quirks, and wrote tests for each one, so the loader is defensive."

---

## 2. `build_graph.py`: tables → graph

**Job:** turn one weekday of service into a directed, weighted graph.

**Key decisions:**
- **Node = a served stop**, named `"<feed>:<stop_id>"` (e.g. `lbt:0002`). The prefix lets feeds be combined without ID clashes.
- **Rail platforms merge into their station** (`node_ids`). A station with 3 platforms is one place to a rider.
- **Directed edges**, because buses go A→B, and B→A may not exist (one-way streets, loops).
- **Transit edge weight = median scheduled travel time**, with a 15-second minimum. The median ignores the odd slow trip, and the minimum avoids 0-second edges.
- **Walk edges** connect stops within 200 m, weighted by walking time at 1.3 m/s. They're added only where no transit edge already exists.

| Function | What it does |
|---|---|
| `node_ids(stops, feed)` | Maps each stop_id to its node ID (platform → parent station). |
| `trip_segments(data, feed)` | **The core trick:** sort stop_times by trip and sequence, then `groupby("trip_id").shift(-1)` puts each stop's *next* stop on the same row. Each row is then one hop: `u`, `v`, `travel_s = next arrival − this departure`. |
| `transit_edges(segments)` | Groups hops by `(u, v)`: median time, trip count, and which routes and modes use it. |
| `node_table(...)` | Node attributes: name, coordinates, trips/day, routes, modes. |
| `walking_pairs(nodes, radius)` | Finds all stop pairs within 200 m **without comparing every pair** (see below). |
| `build_graph(feeds)` | Puts it all together into a `networkx.DiGraph`. |
| `graph_summary(G)` | Node/edge counts and connectivity numbers. |
| `save_graph` / `load_graph` | Pickle to and from disk. |

**How `walking_pairs` avoids comparing every pair:** comparing all pairs is n² (190 million comparisons for 13.8K LA stops). Instead it:
1. Converts lat/lon to meters.
2. Drops every stop into a 200 m × 200 m grid cell.
3. Notes that any two stops within 200 m must be in the same or an adjacent cell.
4. So it only compares stops across the 9 neighboring cells. A test checks the result against the slow comparison of every pair.

**Interview angle:** explain the `shift(-1)` trick and the spatial grid. Both are standard techniques (vectorized lag, spatial hashing), and both are easy to explain.

---

## 3. `metrics.py`: how healthy is a network?

**Job:** given a graph and a set of removed stops, compute health numbers fast.

| Function | What it does |
|---|---|
| `to_matrix(G)` | Converts the graph to a **sparse adjacency matrix** (row = from, column = to, value = seconds). It keeps a fixed node order so "remove node #57" means the same stop every time. |
| `evaluate(matrix, removed, sources)` | Deletes the removed rows and columns, runs **scipy's Dijkstra** from every source, and computes the metrics. |

**Metrics** (all measured against the *original* stop set, so a removed stop counts as a loss):
- **Efficiency** = the average of `1 / travel_time` over all ordered stop pairs, where an unreachable pair counts as 0. It's the one number that captures both "trips got slower" and "trips became impossible".
- **Reachable share** = the fraction of pairs that can still connect.
- **Mean time** = the average travel time among pairs that are still reachable.
- **LSCC share** = the size of the largest group where everyone can reach everyone (largest strongly connected component).

**Why scipy and not networkx?** networkx's shortest paths are written in Python, while scipy's run in C. For 1,853 stops, scipy computes *all-pairs* shortest paths in about 0.3 s. That speed makes Week 3 (hundreds of damaged networks) and Week 4 (1,853 removals) practical.

**`sources` sampling:** for big graphs, run Dijkstra from a random sample of stops instead of all of them. That gives the same metrics as an estimate, much faster.

---

## 4. `simulate.py`: break things on purpose

| Function | What it does |
|---|---|
| `rank_nodes(G, strategy)` | Orders stops most-important-first by `random`, `degree`, `trips_per_day`, or `betweenness`. |
| `attack_curve(G, strategy, fractions, adaptive)` | For each fraction (0%, 2%, 4%, …), removes that many top-ranked stops and calls `evaluate`. **Static** ranks once; **adaptive** re-ranks the damaged graph each round. |
| `random_curves(...)` | Several random-failure curves (different seeds) so you can average them. |
| `route_edges(G)` | For each route, the transit edges **only that route** serves. |
| `route_outages(G)` | Removes each route's exclusive edges, one route at a time. Stops stay open. |
| `single_removal_damage(G, workers)` | Removes **each stop on its own** and measures the loss. These are the Week 4 labels. |

**Parallelism in `single_removal_damage`:** each removal is independent, so the work is spread across CPU cores with `ProcessPoolExecutor`. The `_init_worker` initializer sends the matrix to each worker **once**, instead of once per task. On 10 cores, 9.5 CPU-minutes of work finishes in about 1 minute. A test checks that the parallel and one-core results are identical.

**Interview angle:** "Static attacks underestimated vulnerability: the top betweenness stops were clustered on one corridor, so removing more of them was redundant. Adaptive attacks, which re-rank after each round, showed the true worst case."

---

## 5. `features.py`: cheap clues about each stop

**Job:** describe each stop with numbers that are fast to compute and might predict its damage.

| Group | Features | Intuition |
|---|---|---|
| Service | `trips_per_day`, `n_routes` | Busy stops *might* matter more (Week 3 said: not much). |
| Local structure | `transit_in`, `transit_out`, `walk_degree`, `is_terminus` | How many ways in and out. |
| **Chain structure** | `nearest_stop_m`, `bypass_gap_m` | **Can riders get around this stop?** `bypass_gap_m` is the shortest distance from any stop *before* it to any stop *after* it on a route. A big gap means closing this stop cuts the route. This was designed from Week 3's finding. |
| Global structure | `betweenness`, `harmonic_in`, `harmonic_out`, `pagerank` | Position in the whole network. Harmonic closeness = the average of 1/travel time to (out) or from (in) everyone else. |
| Geography | `dist_center_km` | Edge-of-network stops behave differently. |

- **`build_dataset(G)`** = features + exhaustive labels + timings (wall time and CPU time).
- **`main()`** is the command-line entry point. It writes `<graph>_dataset.csv` and `<graph>_timings.csv`.

**A bug the tests caught:** the first `bypass_gap` paired a stop's predecessor with the *same* stop as successor on two-way segments (A→B→A), which gave a gap of 0. The fix compares only *different* stops. Worth mentioning in an interview as an example of why you write tests.

---

## 6. `model.py`: learn the damage

| Function | What it does |
|---|---|
| `MODELS` | Three regressors: **Ridge** (linear baseline), **Random Forest**, and **Histogram Gradient Boosting** (sklearn's equivalent of LightGBM). |
| `score(y_true, y_pred)` | Three metrics: **R²** (are the values right), **Spearman ρ** (is the ranking right), **top-5% overlap** (did it find the critical stops). |
| `spatial_groups(df)` | K-means on lat/lon → 10 neighborhoods. |
| `out_of_fold_predictions(df, model, groups)` | K-fold cross-validation, so **every stop is predicted by a model that never saw it**. With `groups`, whole neighborhoods are held out. |
| `compare(df, groups)` | Scores single-feature heuristics vs. trained models on the same stops. |
| `fit(df)` | Final model trained on all stops. |

**Three ideas worth understanding:**
1. **Log target:** damage is skewed (a few stops matter a lot). Training on `log(damage)` stops those few from dominating the fit, and `np.exp` converts predictions back.
2. **Spatial cross-validation:** neighboring stops are near-duplicates. A random split leaks information from training into testing, so the spatial split is the honest estimate of how the model performs on an unseen part of the city.
3. **Ranking metrics matter more than R² here.** The use case is "which stops should we protect first?", so Spearman and top-5% overlap answer the real question.

---

## 7. Tests (`tests/`)

Each test file builds a **tiny graph or feed by hand** where you can work out the right answer on paper, for example "a line A–B–C where removing B disconnects A from C".

- **Why tiny fake data instead of the real feeds?** `data/raw` isn't committed, so anyone who clones the repo can still run `pytest`. Tiny cases also make failures easy to understand.
- **Run them:** `pytest` (about 4 s).

## 8. Notebooks (`notebooks/`)

The notebooks *use* the package and don't define logic of their own. If you find yourself writing a long function in a notebook, it probably belongs in `src/transitshock/` with a test.

| Notebook | Question |
|---|---|
| `01_explore_gtfs` | What's in the data? |
| `02_graph_stats` | What does the network look like as a graph? |
| `03_robustness` | How does it fail? |
| `04_model` | Can a model predict which stops matter? |
