"""Smoke tests: the Streamlit app runs end to end on the committed app/data files."""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app" / "streamlit_app.py"
DATA = ROOT / "app" / "data"

pytestmark = pytest.mark.skipif(
    not (DATA / "lbt_nodes.csv").exists(), reason="export app data first: python scripts/run_pipeline.py"
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
