---
license: odbl
task_categories:
  - other
tags:
  - geospatial
  - routing
  - graph
  - cycling
  - railway
  - openstreetmap
  - geoparquet
  - gis
pretty_name: DACH Bike + Rail Routing Graph
size_categories:
  - 1M<n<10M
---

# DACH Bike + Rail Routing Graph

A prebuilt, ready-to-route cycling + railway graph covering **Germany, Austria, and Switzerland (DACH)**, stored as lat/lon-tiled GeoParquet. Built from OpenStreetMap by the [Bike Route Optimizer](https://github.com/MichaelMedek/bike-route-optimizer) for flat-preferring, surface-aware bike routing that can also hop on a train uphill.

Everything routing needs is **baked in** — node elevations and full 3D edge geometry — so an application downloads this once and routes offline, with **no Overpass and no elevation service at query time.**

![DACH bike + rail graph overview](dach_graph_overview.png)

*The whole network — bike roads (thin blue) and rail lines (thick purple). Shipped as `dach_graph_overview.png`.*

## Coverage

| Parameter | Value |
|-----------|-------|
| **West**  | ~5.9° E |
| **East**  | ~17.2° E |
| **South** | ~45.8° N |
| **North** | ~55.1° N |
| **Region** | Germany + Austria + Switzerland |
| **Base data** | OpenStreetMap cycling network + railway (via Geofabrik extracts) |
| **Elevation** | EuroDEM, sampled and baked into every node and edge vertex at build time |

The exact bounding box built is recorded in `meta.json` (`bbox` = `[west, south, east, north]`).

## What's in the graph

- **Bike network** — every bike-legal way, with genuinely unrideable surfaces (mud, sand, rock, …) removed, degree-2 pass-through nodes contracted, and intersections within **25 m** consolidated to shrink the graph while keeping the true road shape.
- **Railway** — heavy rail plus regional `light_rail` and `narrow_gauge` (trams, funiculars, and park/miniature railways are excluded). Each station is snapped onto the nearest rail line by a *rail* edge, and joined to nearby bike nodes (within **200 m**) by bike↔station *station* access edges.
- **Baked elevation** — every node has an `elevation_m`, and every bike and rail edge's geometry is a 3D `LINESTRING Z (lon lat elev …)`, so the elevation profile follows the real road/track.

## File layout

The dataset is tiled on a 0.5° lat/lon grid so a consumer reads only the tiles a route corridor spans, not the whole country.

```
meta.json                     # bbox, tile_deg, consolidation tolerance, node/edge/station counts, source regions
nodes/tile_<row>_<col>.parquet
edges/tile_<row>_<col>.parquet
edge_unreliable_elevation/tile_<row>_<col>.parquet   # optional; only where offenders exist
dach_graph_overview.png       # whole-network preview image
```

### Schema

**nodes** — `osmid, lat, lon, elevation_m, node_type, station_name` (`node_type` is `bike` or `rail`; `station_name` is null for bike nodes and, on a rail node, may still be null for an unnamed halt).

**edges** — `from_node, to_node, key, length_m, height_diff_m, surface, highway, mode, geometry_wkt`

- `mode` is one of `bike`, `rail`, `station` (a `station` edge is the bike↔station access link).
- `geometry_wkt` is a 3D `LINESTRING Z` for bike and rail edges; null for straight station hops.
- `key` is the parallel-edge index (a `(from_node, to_node, key)` triple is unique).

**edge_unreliable_elevation** *(optional)* — `from_node, to_node, key, elevation_deviation_m`: the max metres a bike edge's baked terrain deviates from its straight node-to-node line, for edges past a 50 m threshold only. A consumer may join it on `(from_node, to_node, key)` to deprioritize edges whose elevation profile is unreliable; routing works fine without it.

Coordinates are WGS84 (EPSG:4326), rounded to 6 decimals (~0.1 m).

## Usage

```python
from huggingface_hub import hf_hub_download, list_repo_files

repo = "MichaelMedek/dach_bike_graph"
for filename in list_repo_files(repo_id=repo, repo_type="dataset"):
    hf_hub_download(repo_id=repo, repo_type="dataset", filename=filename, local_dir="dach_graph")
```

Then read the tiles covering your area with pandas / pyarrow and rebuild a graph — or just use the [Bike Route Optimizer](https://github.com/MichaelMedek/bike-route-optimizer), which downloads and windows this dataset automatically.

## How it was built

Reproducible from OpenStreetMap: each Geofabrik region extract is read with pyrosm, the cycling + railway networks are extracted, surfaces filtered, nodes contracted and consolidated, elevation baked from the EuroDEM, and railway links added; the 31 orginal DACH sub-regions (Autria and Switzerland are furter split during pre-processing) are then merged, seam-deduplicated, and tiled.

## License

Derived from OpenStreetMap, so distributed under the **Open Database License (ODbL)**. © OpenStreetMap contributors. Elevation baked from the EuroDEM (EuroGeographics).
