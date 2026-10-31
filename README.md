# TransitShock

Transit network resilience analyzer. It models Long Beach Transit (and LA Metro) as a weighted graph built from GTFS data, simulates stop and route outages, and trains a model that predicts how disruptive a stop closure will be without running the full simulation.

## Status

- Weeks 1–5 are done: data loading, graph, resilience simulation, surrogate model, and Streamlit demo.
- Week 6 (deploy and polish) is next.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Build the data

The raw data is not committed, so run these once:

```bash
./scripts/download_data.sh
python -m transitshock.build_graph lbt --date 20261104
python -m transitshock.features data/processed/lbt_20261104.gpickle
```

## Run the demo

```bash
streamlit run app/streamlit_app.py
```

## Tests

```bash
pytest
```

## Data

Public GTFS Schedule feeds from Long Beach Transit and LA Metro. See `scripts/download_data.sh` and `notes/gtfs_notes.md`.

## Learn more

- `notes/code_walkthrough.md` explains the code, module by module.
- `notebooks/01`–`04` cover exploration, graph stats, robustness, and the model.
