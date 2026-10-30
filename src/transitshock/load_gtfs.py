"""Load and clean a GTFS feed for a single service day."""
from pathlib import Path

import pandas as pd

# Only the columns the pipeline uses; anything else in the feed is dropped on read.
COLUMNS = {
    "stops": ["stop_id", "stop_name", "stop_lat", "stop_lon", "location_type", "parent_station"],
    "routes": ["route_id", "route_short_name", "route_long_name", "route_type", "route_color"],
    "trips": ["route_id", "service_id", "trip_id", "direction_id", "shape_id"],
    "stop_times": [
        "trip_id", "arrival_time", "departure_time", "stop_id",
        "stop_sequence", "timepoint", "shape_dist_traveled",
    ],
    "calendar": [
        "service_id", "monday", "tuesday", "wednesday", "thursday",
        "friday", "saturday", "sunday", "start_date", "end_date",
    ],
    "calendar_dates": ["service_id", "date", "exception_type"],
}


def read_table(feed_dir: Path, name: str) -> pd.DataFrame:
    """Read one GTFS file as strings with whitespace stripped.

    Everything is read as text so IDs keep leading zeros (LBT stop "0002") and
    padded values (Metro's "  33.97", exception_type "1 ") are cleaned in one place.
    Missing optional columns are added as empty so callers can rely on them.
    """
    path = feed_dir / f"{name}.txt"
    wanted = COLUMNS[name]
    if not path.exists():
        return pd.DataFrame(columns=wanted)

    df = pd.read_csv(
        path, dtype=str, keep_default_na=False,
        usecols=lambda c: c.strip() in wanted,
        encoding="utf-8-sig",  # some feeds start with a byte-order mark
    )
    df.columns = df.columns.str.strip()
    for col in df.columns:
        df[col] = df[col].str.strip()
    for col in wanted:
        if col not in df.columns:
            df[col] = ""
    return df[wanted]


def time_to_seconds(series: pd.Series) -> pd.Series:
    """Convert 'HH:MM:SS' (HH may exceed 23) to seconds after midnight; blanks become NaN."""
    parts = series.str.split(":", expand=True)
    hours = pd.to_numeric(parts[0], errors="coerce")
    minutes = pd.to_numeric(parts[1], errors="coerce")
    seconds = pd.to_numeric(parts[2], errors="coerce")
    return hours * 3600 + minutes * 60 + seconds


def active_service_ids(calendar: pd.DataFrame, calendar_dates: pd.DataFrame, date: str) -> set[str]:
    """Service IDs running on `date` (YYYYMMDD).

    Weekly patterns come from calendar.txt (if the feed has one), then
    calendar_dates.txt adds (exception_type 1) or removes (2) specific dates.
    """
    weekday = pd.Timestamp(date).day_name().lower()

    on_day = calendar[
        (calendar[weekday] == "1")
        & (calendar["start_date"] <= date)
        & (calendar["end_date"] >= date)
    ]
    services = set(on_day["service_id"])

    exceptions = calendar_dates[calendar_dates["date"] == date]
    services |= set(exceptions.loc[exceptions["exception_type"] == "1", "service_id"])
    services -= set(exceptions.loc[exceptions["exception_type"] == "2", "service_id"])
    return services


def clean_stops(stops: pd.DataFrame) -> pd.DataFrame:
    stops = stops.copy()
    stops["stop_lat"] = pd.to_numeric(stops["stop_lat"], errors="coerce")
    stops["stop_lon"] = pd.to_numeric(stops["stop_lon"], errors="coerce")
    stops["location_type"] = pd.to_numeric(stops["location_type"], errors="coerce").fillna(0).astype(int)
    return stops


def clean_stop_times(stop_times: pd.DataFrame) -> pd.DataFrame:
    stop_times = stop_times.copy()
    stop_times["stop_sequence"] = pd.to_numeric(stop_times["stop_sequence"]).astype(int)
    stop_times["arrival_s"] = time_to_seconds(stop_times["arrival_time"])
    stop_times["departure_s"] = time_to_seconds(stop_times["departure_time"])
    stop_times["shape_dist_traveled"] = pd.to_numeric(stop_times["shape_dist_traveled"], errors="coerce")
    # Spec: no timepoint values means all times are exact.
    stop_times["timepoint"] = pd.to_numeric(stop_times["timepoint"], errors="coerce").fillna(1).astype(int)
    return stop_times.sort_values(["trip_id", "stop_sequence"]).reset_index(drop=True)


def load_feed(feed_dir: str | Path, date: str) -> dict[str, pd.DataFrame]:
    """Return stops, routes, trips, and stop_times filtered to one service day.

    `date` is YYYYMMDD. Raises ValueError if no service runs that day, which
    usually means the date is outside the feed's calendar range.
    """
    feed_dir = Path(feed_dir)
    calendar = read_table(feed_dir, "calendar")
    calendar_dates = read_table(feed_dir, "calendar_dates")

    services = active_service_ids(calendar, calendar_dates, date)
    if not services:
        raise ValueError(f"No active service on {date} in {feed_dir}; pick a date inside the feed's range.")

    trips = read_table(feed_dir, "trips")
    trips = trips[trips["service_id"].isin(services)].reset_index(drop=True)

    stop_times = read_table(feed_dir, "stop_times")
    stop_times = clean_stop_times(stop_times[stop_times["trip_id"].isin(trips["trip_id"])])

    return {
        "stops": clean_stops(read_table(feed_dir, "stops")),
        "routes": read_table(feed_dir, "routes"),
        "trips": trips,
        "stop_times": stop_times,
    }
