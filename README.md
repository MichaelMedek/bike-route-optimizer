# bike-route-optimizer

Plan a bike route that avoids climbing, bad surfaces, and busy roads — and optionally hops on a train where it saves a big hill. Optimised for the Germany + Austria + Switzerland (DACH) region.

## Setup

```bash
python3 -m venv .venv                 # once
source .venv/bin/activate             # every new shell (macOS / Linux)
pip install -r requirements.txt
```

## Run the web app (3D map)

```bash
source .venv/bin/activate
streamlit run app_webmap.py
```

Open the Local URL it prints (default http://localhost:8501). Type a start and end place, press "Set start & end" to mark them on the 3D map, tune the five preference sliders, then press "Compute route". The route is drawn as a coloured ribbon floating above the terrain (bike vs train legs in different colours); you can download the GPX track, download the elevation PNG, or copy a Google Maps link.

## Run the CLI

```bash
source .venv/bin/activate
python bike_route.py "Freudenstadt, Germany" "Pforzheim, Germany"
```

Add `--extra_km_per_uphill_100m 100` (or any other preference flag) to bias the route; run with `-v` for progress logging.

## The five preferences

Each preference is expressed in the same intuitive unit: **how many extra kilometres would you happily pedal to avoid one unit of the bad thing.** `0` = ignore it, `1` = mild nudge, `10` = go far out of your way to avoid it.

| Flag | `0` | `1` | `10` (high) |
|---|---|---|---|
| `--extra_km_per_uphill_100m` | climbs freely, shortest path | mild flattening | long detours to dodge hills |
| `--extra_km_per_unpaved_km` | happily rides gravel | mild paved preference | detours far to stay paved |
| `--extra_km_per_main_road_km` | uses busy roads freely | mild quiet-road preference | detours far to avoid main roads |
| `--extra_km_per_rail_km` | train distance is free (flat-rate ticket) | mild bias off trains | avoids long train legs |
| `--extra_km_per_boarding` | boards trains freely | mild bias against boarding | avoids catching trains at all |

## The physics & mathematics

Routing happens on a graph: junctions are nodes, the roads between them are edges. Every edge carries its true length in metres, the height difference between its ends, its surface, its road class, and — for train links — a rail marker. The planner turns each edge into a single "felt cost" in metres, then finds the path from start to finish with the least total felt cost. Because every part of the cost is a real distance plus non-negative penalties, the cheapest possible edge is just its raw length, which keeps the search provably optimal.

For a **bike** edge the felt cost adds up like this, where every term is already in metres so they combine cleanly:

$$
\text{felt\_cost} = \text{length} + \frac{\text{climb\_m}}{100}\cdot p_{\text{uphill}}\cdot 1000 + \text{length\_km}\cdot p_{\text{unpaved}}\cdot 1000 \cdot \text{tier} + \text{length\_km}_{\text{main}}\cdot p_{\text{mainroad}}\cdot 1000
$$

Climb counts only going up (downhill adds nothing, so the same street is cheaper downhill than up); `tier` is 0 for tarmac and 1 for gravel, and the worst surfaces are excluded outright.

A **train** edge (station to station) is priced completely differently — a train doesn't care about hills, surface, or traffic, so it only pays for distance and the act of boarding:

$$
\text{felt\_cost} = \text{length} + p_{\text{boarding}}\cdot 1000 + \text{length\_km}\cdot p_{\text{rail}}\cdot 1000
$$

The boarding term is charged once, the moment you get on.

Once the path is chosen, its **ride time** comes from a speed that falls with the slope: roughly 25 km/h on flat good tarmac, and slower on gravel. Speed is dropping linearly toward walking pace on a steep 12 % climb. Downhill and flat hold the top speed. A train leg instead moves at a fixed 80 km/h, and each boarding adds a flat 30-minute wait. Distance and climb come straight from the real road geometry, so the reported kilometres, minutes, and metres-of-ascent all agree with the drawn track.

## Preprocessing, publishing, and auto-download

The routing graph is built once offline, not per request. Building it reads raw OpenStreetMap extracts (one downloadable `.osm.pbf` file per region), drops unrideable surfaces, merges near-identical junctions within about 25 m, bakes in elevation sampled from a terrain model so no elevation data is ever needed at query time, adds the train links, and writes the result as compact map-data files split into small geographic tiles.

```bash
# quick check on a small area (Schwarzwald, covering Freudenstadt → Pforzheim)
python scripts/build_country_graph.py data/pbf/karlsruhe-regbez.osm.pbf --out data/dach_graph --bbox 8.30 48.40 8.80 48.95

# full DACH, overnight and resumable (downloads the region extracts on demand, checkpoints after each)
python scripts/build_dach_graph.py -v

# publish the finished graph to Hugging Face (log in once when prompted)
python scripts/upload_graph_to_huggingface.py
```

The app looks for the graph locally first; if it is missing it downloads it once from Hugging Face and caches it, so every later run is instant and offline.
