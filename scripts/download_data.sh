#!/usr/bin/env bash
# Download GTFS feeds into data/raw/. Run from the repo root.
set -euo pipefail

RAW=data/raw
mkdir -p "$RAW"

LBT_URL="https://files.mobilitydatabase.org/mdb-1198/mdb-1198-202608250048/mdb-1198-202608250048.zip"
METRO_BUS_URL="https://gitlab.com/LACMTA/gtfs_bus/raw/master/gtfs_bus.zip"
METRO_RAIL_URL="https://gitlab.com/LACMTA/gtfs_rail/raw/master/gtfs_rail.zip"

fetch() {
  local name=$1 url=$2
  echo "Downloading $name..."
  curl -fL "$url" -o "$RAW/$name.zip"
  unzip -oq "$RAW/$name.zip" -d "$RAW/$name"
}

fetch lbt "$LBT_URL"
fetch metro_bus "$METRO_BUS_URL"
fetch metro_rail "$METRO_RAIL_URL"

echo "Done. Feeds in $RAW/"