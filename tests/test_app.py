"""Smoke test: the Streamlit app runs end to end (skipped if the data hasn't been built)."""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app" / "streamlit_app.py"
DATA = ROOT / "data" / "processed"

pytestmark = pytest.mark.skipif(
    not (DATA / "lbt_20261104.gpickle").exists() or not (DATA / "lbt_20261104_dataset.csv").exists(),
    reason="build the LBT graph and dataset first",
)


def test_app_runs_and_applies_preset():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP), default_timeout=120).run()
    assert not at.exception
    assert at.metric[0].value == "100.0%"  # nothing closed yet

    at.button(key="preset_top10").click().run()  # "Close the model's top 10 critical stops"
    assert not at.exception
    assert len(at.multiselect[0].value) == 10
    assert at.metric[0].value != "100.0%"


def test_route_outage_scenario():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP), default_timeout=120).run()
    at.multiselect(key="routes").set_value(["71"]).run()
    assert not at.exception
    assert at.metric[0].value != "100.0%"
