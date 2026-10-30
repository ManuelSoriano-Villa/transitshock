"""Tests for load_gtfs using a tiny synthetic feed (raw data isn't committed)."""
import pandas as pd
import pytest

from transitshock.load_gtfs import active_service_ids, load_feed, read_table, time_to_seconds

# 2026-11-04 is a Wednesday.
DATE = "20261104"


def write_feed(path, files):
    path.mkdir()
    for name, text in files.items():
        # Strip only newlines so intentional trailing spaces in values survive.
        (path / f"{name}.txt").write_text(text.strip("\n") + "\n")
    return path


BASE_FILES = {
    "stops": """
stop_id,stop_name,stop_lat,stop_lon,location_type,parent_station
0002,A St,  33.70,-118.10,,
0004,B St,33.71,-118.11,0,
0006,C St,33.72,-118.12,0,
""",
    "routes": """
route_id,route_short_name,route_long_name,route_type
1,1,Main Line,3
""",
    "trips": """
route_id,service_id,trip_id,direction_id
1,WKDY,t1,0
1,SAT,t2,0
""",
    "stop_times": """
trip_id,arrival_time,departure_time,stop_id,stop_sequence
t1,25:00:00,25:00:00,0002,1
t1,25:01:30,25:01:30,0004,5
t1,25:03:00,25:03:00,0006,12
t2,08:00:00,08:00:00,0002,1
""",
}


@pytest.fixture
def feed(tmp_path):
    files = dict(BASE_FILES)
    files["calendar"] = """
service_id,monday,tuesday,wednesday,thursday,friday,saturday,sunday,start_date,end_date
WKDY,1,1,1,1,1,0,0,20260101,20261231
SAT,0,0,0,0,0,1,0,20260101,20261231
"""
    return write_feed(tmp_path / "feed", files)


def test_time_to_seconds_handles_after_midnight():
    result = time_to_seconds(pd.Series(["25:15:00", "05:29:44", ""]))
    assert result[0] == 90900
    assert result[1] == 5 * 3600 + 29 * 60 + 44
    assert pd.isna(result[2])


def test_read_table_keeps_leading_zeros_and_strips_whitespace(feed):
    stops = read_table(feed, "stops")
    assert stops["stop_id"].tolist() == ["0002", "0004", "0006"]
    assert stops.loc[0, "stop_lat"] == "33.70"


def test_read_table_missing_file_returns_empty_with_columns(tmp_path):
    df = read_table(tmp_path, "calendar")
    assert df.empty
    assert "monday" in df.columns


def test_calendar_dates_only_feed(tmp_path):
    """LBT style: no calendar.txt, every service date listed explicitly."""
    calendar_dates = pd.DataFrame(
        {"service_id": ["WKDY", "WKDY"], "date": [DATE, "20261105"], "exception_type": ["1", "1"]}
    )
    empty_calendar = read_table(tmp_path, "calendar")
    assert active_service_ids(empty_calendar, calendar_dates, DATE) == {"WKDY"}


def test_calendar_dates_removes_service(feed):
    calendar = read_table(feed, "calendar")
    calendar_dates = pd.DataFrame({"service_id": ["WKDY"], "date": [DATE], "exception_type": ["2"]})
    assert active_service_ids(calendar, calendar_dates, DATE) == set()


def test_exception_type_with_trailing_whitespace(tmp_path):
    """Metro bus style: exception_type values like '1 '."""
    files = dict(BASE_FILES)
    files["calendar_dates"] = "service_id,date,exception_type\nWKDY,20261104,1  "
    feed_dir = write_feed(tmp_path / "feed", files)
    assert set(load_feed(feed_dir, DATE)["trips"]["trip_id"]) == {"t1"}


def test_load_feed_filters_to_weekday(feed):
    data = load_feed(feed, DATE)
    assert data["trips"]["trip_id"].tolist() == ["t1"]
    assert set(data["stop_times"]["trip_id"]) == {"t1"}


def test_load_feed_parses_stop_times(feed):
    st = load_feed(feed, DATE)["stop_times"]
    assert st["stop_sequence"].tolist() == [1, 5, 12]  # non-consecutive is valid
    assert st["arrival_s"].diff().iloc[1:].tolist() == [90, 90]


def test_load_feed_cleans_stops(feed):
    stops = load_feed(feed, DATE)["stops"]
    assert stops["stop_lat"].iloc[0] == pytest.approx(33.70)
    assert stops["location_type"].tolist() == [0, 0, 0]  # empty means 0


def test_load_feed_no_service_raises(feed):
    with pytest.raises(ValueError, match="No active service"):
        load_feed(feed, "20270615")
