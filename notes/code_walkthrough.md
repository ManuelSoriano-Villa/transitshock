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

Clean tables ──► timetable.build_timetable(...) → every scheduled bus hop      (Week 7)
                    │  timetable.disrupt(...)   → blocked/closed stops, canceled/halved/late routes
                    ▼
                 csa.travel_times(...)          → door-to-door minutes, including waiting
                    │
                    ▼
                 scenario.timetable_impact(...) → what the app's what-if map shows
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
| `timetable_removal_damage(tt, mode)` | The same, measured on the timetable with waiting counted. These are the notebook 06 labels. |
| `greedy_blocking(tt, k, candidates)` | Finds the **worst combination** of k stops to block (see below). |

**Parallelism in `single_removal_damage`:** each removal is independent, so the work is spread across CPU cores with `ProcessPoolExecutor`. The `_init_worker` initializer sends the matrix to each worker **once**, instead of once per task. On 10 cores, 9.5 CPU-minutes of work finishes in about 1 minute. A test checks that the parallel and one-core results are identical.

**`greedy_blocking`: the worst set of stops.**
- **Why not just take the top 10?** Damage isn't additive. Two stops on the same corridor protect the same riders, while a stop can matter *more* once its neighbors are blocked.
- **How greedy works:** pick one stop at a time, each time taking the candidate that adds the most damage given the ones already chosen.
- **The lazy (CELF) trick:**
  - Keep each candidate's damage from the last round as an estimate.
  - Re-measure only the current leader. If it's still on top, take it; otherwise re-sort and repeat.
  - That cuts about 500 evaluations down to about 100.
  - It's exact when damage has diminishing returns, and a heuristic when it doesn't. `lazy=False` runs plain greedy, and a test checks that both agree on a small case.
- **In the app:** a `progress` callback drives the progress bar, results are cached per (window, k), and the chosen stops are applied on the next rerun through `pending_blocked`. Streamlit doesn't allow changing a widget's value after it's been drawn.

**Interview angle:** "Static attacks underestimated vulnerability: the top betweenness stops were clustered on one corridor, so removing more of them was redundant. Adaptive attacks, which re-rank after each round, showed the true worst case."

---

## 5. `features.py`: cheap clues about each stop

**Job:** describe each stop with numbers that are fast to compute and might predict its damage.

| Group | Features | Intuition |
|---|---|---|
| Service | `trips_per_day`, `n_routes` | Busy stops *might* matter more (Week 3 said: not much). |
| Local structure | `transit_in`, `transit_out`, `walk_degree`, `is_terminus` | How many ways in and out. |
| **Chain structure** | `nearest_stop_m`, `bypass_gap_m`, `stranded_share` | **Can riders get around this stop?** `bypass_gap_m` is the shortest distance from any stop *before* it to any stop *after* it on a route; a big gap means closing this stop cuts the route (designed from Week 3's finding). `stranded_share` is the share of stop pairs that closing this stop disconnects (see below). |
| Global structure | `betweenness`, `harmonic_in`, `harmonic_out`, `pagerank` | Position in the whole network. Harmonic closeness = the average of 1/travel time to (out) or from (in) everyone else. |
| Geography | `dist_center_km` | Edge-of-network stops behave differently. |

- **`build_dataset(G)`** = features + exhaustive labels + timings (wall time and CPU time).
- **`main()`** is the command-line entry point. It writes `<graph>_dataset.csv` and `<graph>_timings.csv`.

**`stranded_share` and dominator trees.** This feature was added after the first model missed most of the critical stops that *cut other stops off*. The idea comes from compilers:
- From a starting stop `r`, stop `v` **dominates** stop `x` if *every* path from `r` to `x` goes through `v`.
- So if `v` closes, a rider starting at `r` loses exactly the stops `v` dominates.
- `nx.immediate_dominators(G, r)` returns each stop's closest dominator. Those links form a tree, and the size of `v`'s subtree is the number of stops it strands for riders starting at `r`.
- The code builds the tree's children lists, orders stops breadth-first, then walks that order backwards to add up subtree sizes from the leaves.
- Summing over every starting stop gives the exact number of disconnected pairs, which is what the simulation measures, but without removing anything. A unit test checks that the two match exactly.
- Using all 1,853 starting stops takes about 21 s. Sampling 200 takes about 2 s and is close enough, which is the default.

**Interview angle:** "I looked at *which* critical stops the model missed, saw they were the ones that strand others, and found a graph algorithm that computes stranding exactly. Critical stops found went from 51% to 65%."

**A bug the tests caught:** the first `bypass_gap` paired a stop's predecessor with the *same* stop as successor on two-way segments (A→B→A), which gave a gap of 0. The fix compares only *different* stops. Worth mentioning in an interview as an example of why you write tests.

---

## 6. `model.py`: learn the damage

| Function | What it does |
|---|---|
| `MODELS` | Three regressors: **Ridge** (linear baseline), **Random Forest**, and **Histogram Gradient Boosting** (sklearn's equivalent of LightGBM). |
| `score(y_true, y_pred)` | Three metrics: **R²** (are the values right), **Spearman ρ** (is the ranking right), **top-5% overlap** (did it find the critical stops). |
| `spatial_groups(df)` | K-means on lat/lon → 10 neighborhoods. |
| `out_of_fold_predictions(df, model, groups, features)` | K-fold cross-validation, so **every stop is predicted by a model that never saw it**. With `groups`, whole neighborhoods are held out. Pass `features` (a subset) to measure what one feature adds. |
| `compare(df, groups)` | Scores single-feature heuristics vs. trained models on the same stops. |
| `fit(df)` | Final model (random forest by default, the best on held-out neighborhoods) trained on all stops. |

**Two sets of labels** (both from `simulate.py`):
- `efficiency_loss` is the static graph's single-stop damage (`single_removal_damage`).
- `tt_efficiency_loss` is the same, measured on the timetable with waiting counted (`timetable_removal_damage`, about 4.5 minutes for all stops).

Every function takes a `target=` argument, so the same code trains both. Notebook 06 compares them. Timetable-aware features (`features.timetable_features`) were tested there too and *didn't* help, so the final model keeps the 14 graph features.

**Three ideas worth understanding:**
1. **Log target:** damage is skewed (a few stops matter a lot). Training on `log(damage)` stops those few from dominating the fit, and `np.exp` converts predictions back.
2. **Spatial cross-validation:** neighboring stops are near-duplicates. A random split leaks information from training into testing, so the spatial split is the honest estimate of how the model performs on an unseen part of the city.
3. **Ranking metrics matter more than R² here.** The use case is "which stops should we protect first?", so Spearman and top-5% overlap answer the real question.

---

## 7. `scenario.py` + `app/streamlit_app.py`: the demo

**`scenario.py`** is the engine for "what if these stops or routes close?"

| Function | What it does |
|---|---|
| `baseline(G)` | Computes the intact network **once**: node order, matrix, all-pairs travel times, and metrics. The app caches it. |
| `damaged_matrix(G, base, routes)` | Removes the segments only those routes serve. It reuses `simulate.route_edges`. |
| `impact(G, base, stops, routes)` | Removes the stops (they can't be passed, i.e. *blocked*), recomputes all fastest paths (about 0.5 s), and compares each stop's trips with the baseline. |
| `timetable_baseline` / `timetable_impact` | The same idea on the timetable, with waiting counted (section 8). **This is what the app's what-if map now uses.** |

`impact` gives each stop a **status**:
- **closed:** you closed it.
- **cut off:** it can no longer reach 10%+ of the stops it could reach before.
- **slower:** its trips take 2+ minutes longer on average.
- **unaffected:** everything else.

The per-stop math works on whole matrices: `could` = reachable before, `can` = reachable now. `lost = could & ~can` counts destinations lost, and the average of `dist_after − dist_before` over pairs reachable in both gives the extra minutes. No Python loops over stops.

**`streamlit_app.py`** is deliberately thin. All the logic lives in the package, and the app only handles input and display:
- **Caching:** `@st.cache_resource` (the graph, timetable, and one timetable baseline per time-of-day window) and `@st.cache_data` (edges and model predictions) mean the slow work happens once, not on every click.
- **Sidebar:** time of day, stops to block or close, routes to cancel, halve, or delay, plus preset buttons. The presets use `on_click` callbacks that write to `st.session_state`, which is how a button can change another widget's value.
- **What-if tab:** metric cards plus a pydeck map. Stops are colored by status; canceled route segments are purple and halved or late routes are blue. When the scenario is something the static model can also represent, a caption shows its estimate for comparison.
- **Live buses tab:** the animated map (section 9).
- **Critical stops tab:** the model's ranking. Predictions are **out-of-fold** (spatial CV), so no stop's prediction comes from a model trained on that stop or its neighborhood.
- **Smoke tests:** `tests/test_app.py` uses Streamlit's `AppTest` to run the whole app headless: it clicks a preset, sets a route outage, a frequency cut, a delay, and the evening window. It's skipped if `app/data/` is missing.

**Demo talking point:**
- Compare "block the 10 busiest stops" (−2.5% efficiency with waiting, 0 stops cut off) with "block the model's top 10" (−5.8%, 44 stops cut off).
- Then switch to the evening window: the same blockage cuts off 99 stops.

---

## 8. `timetable.py` + `csa.py`: routing with waiting (Week 7)

**Why:** the static graph gives each stop pair one *median* travel time, so it never waits for a bus. Real trips include waiting for the first bus and for every transfer. With waiting counted, the average trip goes from 28.6 to 56.6 minutes.

**`timetable.py`**: every scheduled bus hop as a **connection**: `(departure time, arrival time, from stop, to stop, trip)`.

| Function | What it does |
|---|---|
| `build_timetable(data, feed, G)` | Turns stop_times into **events** (one per trip × stop), then connections, with stops numbered in the same order as `G`'s nodes. |
| `connections_from_events(events)` | The same `groupby(trip).shift(-1)` trick as `trip_segments`: each visit plus the next visit on the same trip makes one connection. Sorted by departure time, which CSA relies on. |
| `footpaths_csr(...)` | Walking transfers in **CSR form** (`ptr`, `to`, `time` arrays). Stop `s`'s neighbors are `to[ptr[s]:ptr[s+1]]`. Plain arrays are what numba can loop over fast. |
| `disrupt(tt, ...)` | Edits the **events**, then rebuilds connections, so the bus's path stays consistent. |

What each disruption does to the events:
- **closed:** drop the stop's events. The next connection jumps over it, so the bus skips it.
- **blocked:** drop the stop's events *and* give each trip a new ID after that point, so nobody can ride through. A `bus` column remembers the original trip for the animation.
- **canceled:** drop the route's trips.
- **halved:** drop every other trip in each direction.
- **late:** add N minutes to the route's times.

**`csa.py`**: the **Connection Scan Algorithm** (Dibbelt et al.):

```
best[source] = t0; best[everything else] = never
for each connection c, in departure order (starting at t0):
    if I'm already on c's bus, or best[c.from] <= c.departure:   # can I catch it?
        mark c's trip as boarded
        if c.arrival < best[c.to]: improve it, and improve stops within walking distance
```

- **Why it's fast:** one pass over the array, no priority queue (unlike Dijkstra), and it stops at a 3-hour horizon. With `@njit` from numba it compiles to machine code, at 0.05 ms per query. `_many` runs queries in parallel with `prange`.
- **Fallback:** if numba isn't installed, the same Python code runs, just slower. A test checks that the compiled and pure-Python versions give identical results.
- `travel_times(tt, sources, t0s)` → minutes matrix (queries × stops), with NaN where a stop isn't reachable.
- `departure_grid(...)` builds "every stop × every 15 minutes".
- `summarize(...)` gives the same metrics as `metrics.evaluate`, so the static and timetable results can be compared directly.

**`scenario.timetable_impact`** mirrors `impact`, but it compares minutes matrices before and after the disruption. It averages each stop's numbers over all departure times in the window.

**Interview angle:**
- "The static model doubled as a sanity check. It identified exactly the same 44 cut-off stops as the timetable under blockage, which validated both."
- "Then the timetable showed what the static model couldn't: frequency cuts, night service, and the difference between a stop being *closed* and a road being *blocked*."

## 9. `animation.py` + `app/live_buses.html`: the live map

**Following the streets.** GTFS `shapes.txt` gives each route's street geometry as points with a `shape_dist_traveled` (how far along the line each point is). `stop_times.txt` gives the same distance for every stop, so both share one ruler.
- **`shape_arrays`** (in `timetable.py`) loads each shape and simplifies it with **Douglas-Peucker**: it keeps only points that move the line by more than 4 m. That takes LBT's shapes from 89K points to 5.8K.
- **`street_path(stop_dist, stop_times, ..., shape)`** merges the stop distances with the shape points between the first and last stop.
  - Each point's **position** is interpolated along the shape by distance (`np.interp(dist, shape_dist, shape_lat)`).
  - Each point's **time** is interpolated between the scheduled stop times by distance traveled.
  - So the bus drives along the real street at the speed the schedule implies.
  - It falls back to straight lines if distances are missing or go backwards; a test covers this.

**`trip_paths(...)`** turns every trip active in the hour into:
`{bus, route, name, headsign, path, timestamps, stops: [[stop_index, t], ...], color, late}`.
- `stops` lists the trip's stops with their times. That's what the click panels use.
- `bus` is the original trip ID, so a blocked trip's two pieces still count as one bus.

**`live_buses.html`** (deck.gl, all in the browser):
- **`TripsLayer`** draws fading trails behind each bus.
- **A bus-marker layer:** each frame, `position(p, t)` binary-searches the trip's timestamps and interpolates between the two surrounding path points. Those positions feed a `ScatterplotLayer` of bus markers.
- **A stops layer:** all 1,853 stops. Blocked stops are drawn black and closed ones gray.
- **Clicking:** `onClick` sets `selected`.
  - **A bus** shows its route, destination, next stop and upcoming stops, and dims every other route.
  - **A stop** shows the next buses due there, found by scanning every trip's `stops` list.
- **Making it usable:** `pickingRadius: 6` makes tiny stops clickable, and the info panel only refreshes about 3 times a second so it stays readable.
- **Safety:** the app embeds the data with `json.dumps(...).replace("</", "<\\/")`, so a name can never close the `<script>` tag early. Names are HTML-escaped before display. The page is rendered with `st.iframe`.

**Impact on the live map:**
- **`scenario.route_impacts(...)`** gives each route its stops' status counts, the average extra minutes, buses in the window before and after, and the delay. The bus panel shows these, so clicking a bus shows *its route's* damage.
- **The stop panel** shows that stop's status, extra minutes and destinations lost.
- **Stops are colored by impact:** red for cut off, orange for slower.
- **Ghost buses:** the app animates the *normal* schedule too, and any bus that's missing from the disrupted one becomes a gray hollow ring (canceled). For late routes, an orange ring shows where the bus *should* be.
- **The same metric cards** as the what-if tab appear above the map, because both tabs share one scenario computation (`summary`, `per_stop`, `routes_impact`) per rerun.

**Click a stop in the what-if map:** `st.pydeck_chart(..., on_select="rerun")` returns the clicked stop. **Every layer needs an `id`** for Streamlit to track selections; that was the bug when clicks first did nothing. The card it shows has "Block / Close this stop" buttons that edit the scenario through `on_click` callbacks.

---

## 10. Tests (`tests/`)

Each test file builds a **tiny graph or feed by hand** where you can work out the right answer on paper, for example "a line A–B–C where removing B disconnects A from C".

- **Why tiny fake data instead of the real feeds?** `data/raw` isn't committed, so anyone who clones the repo can still run `pytest`. Tiny cases also make failures easy to understand.
- **Run them:** `pytest` (about 4 s).

## 11. Notebooks (`notebooks/`)

The notebooks *use* the package and don't define logic of their own. If you find yourself writing a long function in a notebook, it probably belongs in `src/transitshock/` with a test.

| Notebook | Question |
|---|---|
| `01_explore_gtfs` | What's in the data? |
| `02_graph_stats` | What does the network look like as a graph? |
| `03_robustness` | How does it fail? |
| `04_model` | Can a model predict which stops matter? |
| `05_timetable` | What changes when waiting for buses counts? |
| `06_timetable_model` | Which stops are critical when waiting counts, and can the model learn it? |
