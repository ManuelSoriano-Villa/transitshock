# TransitShock

Transit network resilience analyzer. Models a transit system (Long Beach Transit, then LA Metro) as a weighted graph built from GTFS data, simulates stop and line outages, and trains a model that predicts how disruptive a failure will be without running the full simulation.

## Status

Week 1: data loading and exploration.

## Data

Public GTFS Schedule feeds from Long Beach Transit and LA Metro. Raw data is not committed; see `scripts/download_data.sh`.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
