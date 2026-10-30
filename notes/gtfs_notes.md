# GTFS Notes

Reference: https://gtfs.org/documentation/schedule/reference/

## Files

### stops.txt
**One row =** one physical location in the system: a bus stop, a rail platform, a station building, an entrance, or a point inside a station. Primary key is `stop_id`.

**Role in this project:** graph nodes.

| Column | Meaning | Notes for this project |
|---|---|---|
| `stop_id` | Unique ID for the location | Join key to `stop_times.stop_id`. Read as a string (LBT IDs have leading zeros like `0002`). |
| `stop_name` | Rider-facing name | Used for labels on maps and in the demo. Not unique: a station and its platforms share a name. |
| `stop_lat`, `stop_lon` | Coordinates of the pole/platform where riders board | Used for maps and for computing walking-transfer distances. Metro bus values have leading whitespace. |
| `location_type` | What kind of location the row is | `0`/empty = stop or platform, `1` = station, `2` = entrance/exit, `3` = generic node, `4` = boarding area. Only `0` rows appear in `stop_times`. |
| `parent_station` | `stop_id` of the containing station | Set on platforms (`0`) to point at their station (`1`); always empty for stations. |

**Hierarchy:** station (`1`) → platforms (`0`) → boarding areas (`4`). Entrances (`2`) and generic nodes (`3`) also hang off a station.

**Implications for the graph:**
- Vehicles only stop at `location_type = 0`, so those are the rows `stop_times` references.
- A rail station with several platforms would show up as several nodes. In Week 2, collapse platforms into their `parent_station` so each station is one node (for example, Metro rail `80101` → `80101S`, Downtown Long Beach Station).
- Bus stops usually have no parent, so they stay as their own nodes.
- Fields like `stop_code`, `zone_id`, `wheelchair_boarding`, `platform_code`, `stop_timezone` are not needed for the graph.

### routes.txt
**One row =** one rider-facing line or service (e.g., LBT route 1, Metro A Line). A route is a group of trips; it does not itself list stops or times. Primary key is `route_id`.

**Role in this project:** groups stops/edges into lines, which drives "remove a whole line" attacks and the route-count node feature.

| Column | Meaning | Notes for this project |
|---|---|---|
| `route_id` | Unique ID for the route | Join key to `trips.route_id`. Read as a string. |
| `route_short_name` | Short rider label ("1", "101", "Green") | Metro bus: the line number. Metro rail: empty. |
| `route_long_name` | Longer descriptive name | Metro rail: "Metro A Line" etc. Metro bus: generic "Metro Local Line", so not useful there. |
| `route_type` | Mode of transport | `0` = light rail/tram, `1` = subway, `2` = intercity rail, `3` = bus (others: ferry, cable car, etc.). |
| `route_color` | Hex color from agency branding | Use for map line colors. Metro rail has official line colors; LBT uses one color (`D62028`) for every route; Metro bus is mostly empty. |

**Display name rule:** use `route_short_name`, falling back to `route_long_name` when it's empty (needed for Metro rail).

**What's in our feeds:**

| Feed | Routes | `route_type` breakdown |
|---|---|---|
| LBT | 35 | all bus (`3`) |
| Metro bus | 114 | all bus (`3`) |
| Metro rail | 6 | 4 light rail (`0`), 2 subway (`1`) |

**Implications for the graph:**
- `route_type` can become a node/edge attribute (bus vs. rail), useful when combining feeds and as a model feature.
- Number of distinct routes serving a stop = a strong candidate feature for "how critical is this stop."
- Fields like `agency_id`, `route_url`, `route_text_color`, `route_sort_order`, `continuous_pickup`/`continuous_drop_off`, `network_id`, `cemv_support` are not needed.

### trips.txt
**One row =** one run of a vehicle along a route at a specific time (e.g., the 7:15 AM northbound route 61 bus). A trip belongs to one route and runs on the days given by its `service_id`. Primary key is `trip_id`.

**Role in this project:** the link between routes, service days, and stop sequences. Filtering trips by `service_id` is how we keep only one weekday.

| Column | Meaning | Notes for this project |
|---|---|---|
| `trip_id` | Unique ID for the trip | Join key to `stop_times.trip_id`. Read as a string. |
| `route_id` | Route this trip belongs to | Join to `routes.txt` to get route name/type. |
| `service_id` | Set of dates this trip runs | Join to `calendar.txt` / `calendar_dates.txt`. **Filter on this first**, or weekday + Saturday + Sunday trips all get counted together. |
| `direction_id` | `0` or `1` for the two directions | Spec says not to use for routing. Useful only for sanity checks (roughly half the trips each way). |
| `trip_headsign` | Destination shown on the vehicle | Nice for labels; empty in both Metro feeds. |
| `shape_id` | Geographic path the vehicle follows | Only needed if we draw true route lines on the map instead of straight stop-to-stop segments. |

**Chain:** `routes` → `trips` (by `route_id`) → `stop_times` (by `trip_id`) → `stops` (by `stop_id`). Days come from `trips.service_id` → calendar files.

**What's in our feeds (all days combined, before filtering):**

| Feed | Trips | Distinct `service_id`s | Direction split (0 / 1) |
|---|---|---|---|
| LBT | 17,423 | 10 | 8,698 / 8,725 |
| Metro bus | 33,614 | 172 | 16,817 / 16,797 |
| Metro rail | 7,473 | 24 | 3,733 / 3,740 |

- Metro bus `service_id`s encode division + day type (e.g., `JUNE26-D05CAR-3_Sunday`), which is why there are so many.
- Metro rail `service_id`s cover short date windows (e.g., `RJUN26-801-1_Weekday-58`), matching its 2-day calendar ranges.
- Skip: `trip_short_name`, `block_id`, `wheelchair_accessible`, `bikes_allowed`, `cars_allowed`, `safe_duration_*`, Metro's extra `trip_id_event`.

### stop_times.txt
**One row =** one trip's visit to one stop: which stop, in what order, and at what time. Primary key is (`trip_id`, `stop_sequence`). This is the biggest file by far.

**Role in this project:** builds the graph's edges. For each trip, sort by `stop_sequence`; each consecutive pair of stops (A → B) is an edge, and `arrival_time(B) − departure_time(A)` is that edge's travel time. Counting trips per stop or per edge gives service frequency.

| Column | Meaning | Notes for this project |
|---|---|---|
| `trip_id` | Trip this visit belongs to | Join to `trips.txt` (and filter to the chosen weekday's trips). |
| `stop_id` | Stop being served | Always a `location_type = 0` stop/platform, never a station. Join to `stops.txt`. |
| `stop_sequence` | Order of the stop within the trip | Increases along the trip but **is not consecutive** (1, 23, 40 is valid). Sort by it; never assume `+1`. |
| `arrival_time` / `departure_time` | Scheduled times, `HH:MM:SS` | Can exceed `24:00:00` for after-midnight service on the same service day. Parse to seconds manually, not with `pd.to_datetime`. |
| `timepoint` | `1` = exact time, `0` = approximate/interpolated | Most of our rows are `0`, so edge travel times are estimates. Fine for this project; mention it under limitations. |
| `pickup_type` / `drop_off_type` | `0` = normal, `1` = not available, `2`/`3` = arrange by phone/driver | `1` usually marks the first/last stop of a trip. The vehicle still travels the segment, so keep these rows when building edges. |
| `shape_dist_traveled` | Distance along the shape so far | Could give edge distances; optional. |

**What's in our feeds (all days combined):**

| Feed | Rows | Blank arrival times | Times ≥ 24:00:00 | `timepoint` = 0 |
|---|---|---|---|---|
| LBT | 821,208 | 0 | 4,193 | 701,708 (85%) |
| Metro bus | 2,106,179 | 0 | 81,553 | 1,876,467 (89%) |
| Metro rail | 150,896 | 0 | 3,301 | 0 (all exact) |

- **No blank times in any feed**, so the interpolation step mentioned in the Week 1 plan isn't needed (still worth a guard in code).
- After-midnight times appear in every feed, so the custom seconds parser is required.
- Metro bus at ~2.1M rows: load with `dtype=str` and only the needed columns (`usecols`) to keep memory down.
- Skip: `location_group_id`, `location_id`, `start/end_pickup_drop_off_window` (on-demand service; not in our feeds), `stop_headsign`, `continuous_*`, booking rule IDs, and Metro's extra `route_code`, `destination_code`, `bay_num`, `trip_id_event`.

### calendar_dates.txt
**One row =** one exception for one service on one date: "service X runs on date D" or "service X does not run on date D." Primary key is (`service_id`, `date`).

**Role in this project:** together with `calendar.txt`, decides which `service_id`s (and so which trips) are active on the analysis date.

**Two ways agencies use it:**
1. **With `calendar.txt` (recommended by spec):** `calendar.txt` gives the weekly pattern; `calendar_dates.txt` adds/removes specific dates (holidays, special events). Used by Metro bus and Metro rail.
2. **Instead of `calendar.txt`:** every service date is listed explicitly. Used by **LBT** (no `calendar.txt` at all).

| Column | Meaning | Notes for this project |
|---|---|---|
| `service_id` | Service being changed | Matches `trips.service_id` (and `calendar.service_id` if that file exists). |
| `date` | Date of the exception, `YYYYMMDD` | Compare as strings; same format as `calendar.start_date`/`end_date`. |
| `exception_type` | `1` = service added on that date, `2` = service removed | **Strip whitespace before comparing.** Metro bus has values like `"1 "` and `"1  "`. |

**Rule for "is service S active on date D?"**
1. Start with every `service_id` in `calendar.txt` whose weekday column is `1` for D's weekday and where `start_date ≤ D ≤ end_date`.
2. Add every `service_id` with a row (`date = D`, `exception_type = 1`).
3. Remove every `service_id` with a row (`date = D`, `exception_type = 2`).

`calendar_dates` always wins over `calendar`.

**What's in our feeds:**

| Feed | Rows | Added (`1`) | Removed (`2`) | Date range |
|---|---|---|---|---|
| LBT | 168 | 168 | 0 | 2026-08-23 → 2027-02-06 (168 dates, one row per day) |
| Metro bus | 7,086 | 305 | 6,781 | 2026-01-12 → 2027-04-01 |
| Metro rail | 7 | 0 | 7 | 2026-10-28 → 2026-11-05 |

- Metro bus removes service a lot: `calendar.txt` gives wide patterns, and `calendar_dates.txt` carves out dates per division.
- LBT has exactly one active `service_id` per day.

## Traced trip (LBT route 104, trip 10795278)
- Headsign: "104 WILLOW TO SANTA FE AVE VIA AIRPORT", direction `0`
- Service: `AUG26-Weekday-73-AUG26-1111100-` (active on 2026-11-04)
- First 20 stops shown (the trip continues past stop 20)

| seq | stop_id | stop name | arrival | Δ from previous | shape_dist_traveled | timepoint |
|---|---|---|---|---|---|---|
| 1 | 1802 | Carson & Norwalk SW | 05:29:00 | — | 0.0 | 1 |
| 2 | 1807 | Norwalk & Civic Center NW | 05:29:44 | 44s | 0.182 | 0 |
| 3 | 1808 | Norwalk & 221st SW | 05:30:27 | 43s | 0.361 | 0 |
| 4 | 1810 | Norwalk & 226th SW | 05:31:34 | 67s | 0.64 | 0 |
| 5 | 1811 | Norwalk & Ring SW | 05:32:05 | 31s | 0.77 | 0 |
| 6 | 1812 | Norwalk & Wardlow NW | 05:33:00 | 55s | 1.0 | 1 |
| 7 | 8516 | Los Alamitos & Cerritos NW | 05:34:55 | 115s | 1.512 | 0 |
| 8 | 8514 | Spring & Claremore NW | 05:36:24 | 89s | 1.909 | 0 |
| 9 | 8518 | Spring & El Dorado Park NW | 05:38:00 | 96s | 2.337 | 1 |
| 10 | 1842 | Spring & Studebaker NW | 05:40:25 | 145s | 3.114 | 0 |
| 11 | 1843 | Spring & Nipomo NW | 05:40:56 | 31s | 3.282 | 0 |
| 12 | 1844 | Spring & Palo Verde NW | 05:42:00 | 64s | 3.623 | 1 |
| 13 | 1846 | Spring & Woodruff NW | 05:43:20 | 80s | 4.045 | 0 |
| 14 | 1847 | Spring & Los Coyotes NW | 05:43:36 | 16s | 4.133 | 0 |
| 15 | 1848 | Spring & San Anseline NW | 05:44:26 | 50s | 4.397 | 0 |
| 16 | 1849 | Spring & Bellflower NW | 05:45:08 | 42s | 4.618 | 0 |
| 17 | 1850 | Spring & Charlemagne NW | 05:46:06 | 58s | 4.928 | 0 |
| 18 | 1851 | Spring & Clark NW | 05:46:45 | 39s | 5.134 | 0 |
| 19 | 1852 | Spring & Airport Plaza Dr NE | 05:47:11 | 26s | 5.269 | 0 |
| 20 | 1358 | Lakewood & Spring NE | 05:47:50 | 39s | 5.477 | 0 |

**Takeaways:**
- Each row pair becomes one directed edge, e.g. `1802 → 1807` with weight 44s. These 20 stops give 19 edges.
- Stops 1–20 cover 5.477 shape units in 18m50s. If the units are miles, that's about 17 mph, which is a believable bus speed.
- Timepoints (`timepoint = 1`) appear every few stops (seq 1, 6, 9, 12). The times in between are interpolated, which is why they have odd seconds like 05:29:44.
- `arrival_time` equals `departure_time` at every stop, so there's no dwell time to subtract.
- The longest gap, 145s from seq 9 to 10, matches the biggest jump in distance (0.78 units). Travel time tracks distance, which is a good sanity check.

## Analysis date
- Chosen date: **2026-11-04 (Wednesday)**

Active service on candidate weekdays, computed with the rule above:

| Date | LBT trips | Metro bus trips | Metro rail trips |
|---|---|---|---|
| Tue 2026-11-03 | 1,914 | 13,339 | 1,318 |
| **Wed 2026-11-04** | **1,914** | **13,335** | **1,319** |
| Thu 2026-11-05 | 1,914 | 13,335 | 1,318 |
| Tue 2026-11-10 | 1,914 | 13,339 | 1,254 |
| Tue 2026-11-17 | 1,914 | 13,339 | **0** (rail feed doesn't cover it) |

- Nov 3–5 work in all three feeds; Nov 4 avoids Election Day (Nov 3) and Veterans Day (Nov 11).
- The current rail feed runs out around mid-November. Re-downloading rail later will need a newer date, but LBT and bus are good through Dec 2026 at least.

## Feed sizes
| Feed | stops | stop_times rows | Weekday trips (2026-11-04) |
|---|---|---|---|
| LBT | 1,873 | 821,208 | 1,914 |
| Metro bus | 11,891 | 2,106,179 | 13,335 |

- Counts exclude the header row. Metro bus `stop_times.txt` has no trailing newline, so `wc -l` already reports the true row count for that file.
- Metro bus has about 6× the stops and 2.6× the `stop_times` rows of LBT, so building and testing on LBT first keeps iteration fast.

## Data quirks found
- LBT has no calendar.txt; only calendar_dates.txt
- LBT stop_ids have leading zeros
- Metro bus lat/lon have leading whitespace
- Metro rail uses parent stations with "S" suffix; short service windows
- Metro bus `calendar_dates.exception_type` has trailing whitespace (`"1 "`, `"1  "`); strip before comparing
- No blank `arrival_time`/`departure_time` in any feed; after-midnight times (≥ 24:00:00) in all three
- Most bus `stop_times` rows are `timepoint = 0` (approximate times)
