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

    at.button(key="preset_top10").click().run()  # "Block the model's top 10 critical stops"
    assert not at.exception
    assert len(at.multiselect(key="blocked").value) == 10
    assert at.metric[0].value != "100.0%"


def test_route_outage_scenario():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP), default_timeout=120).run()
    at.multiselect(key="canceled").set_value(["71"]).run()
    assert not at.exception
    assert at.metric[0].value != "100.0%"


def test_frequency_cut_and_delay_scenarios():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP), default_timeout=120).run()
    at.multiselect(key="reduced").set_value(["61"]).run()
    assert not at.exception
    assert at.metric[0].value != "100.0%"
    at.multiselect(key="delayed").set_value(["71"]).run()
    at.selectbox(key="window").set_value("Evening (8–10 PM)").run()
    assert not at.exception


def test_critical_tab_switches_between_models():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP), default_timeout=120).run()
    assert not at.exception
    at.radio(key="crit_measure").set_value("Without waiting (static graph)").run()
    assert not at.exception


def test_greedy_search_toggle_blocks_k_stops():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP), default_timeout=300).run()
    assert "greedy_run" not in [b.key for b in at.button]  # hidden until the toggle is on
    at.toggle(key="greedy_on").set_value(True).run()
    at.slider(key="greedy_k").set_value(3).run()
    at.button(key="greedy_run").click().run()
    assert not at.exception
    assert len(at.multiselect(key="blocked").value) == 3
    assert at.metric[0].value != "100.0%"


@pytest.mark.skipif(not (DATA / "la_nodes.csv").exists(), reason="export the LA network first")
def test_la_network_toggle():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP), default_timeout=300).run()
    at.toggle(key="include_la").set_value(True).run()
    assert not at.exception
    assert at.metric[0].value == "100.0%"
    assert any("A Line" in label for label in at.multiselect(key="canceled").options)  # rail lines are listed
    at.multiselect(key="canceled").set_value(["metro_rail:801"]).run()  # cancel the Metro A Line
    assert not at.exception
    assert at.metric[0].value != "100.0%"
