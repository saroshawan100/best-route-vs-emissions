from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

GBA_FLOW_NPZ = config.LARGEST_DIR / "gba_2019.npz"
COUNTS_XML = config.COUNTS_XML
CANDIDATES_ROU = config.CANDIDATES_ROU
DEMAND_ROU = config.DEMAND_ROU
DEFAULT_ARTERIAL_AADT = config.ARTERIAL_AADT
ARTERIAL_PEAK_FRACTION = 0.08


def load_profile(start_hour: int, end_hour: int,
                 interval_s: int) -> tuple[pd.DataFrame, np.ndarray]:
    archive = np.load(GBA_FLOW_NPZ)
    flow, sensors, times = archive["flow"], archive["sensors"], archive["times"]
    timestamps = pd.to_datetime(times)

    weekday_peak_mask = ((timestamps.dayofweek < 5) & (timestamps.hour >= start_hour)
                         & (timestamps.hour < end_hour))

    peak_flow = flow[weekday_peak_mask]
    peak_times = timestamps[weekday_peak_mask]
    seconds_since_start = (peak_times.hour * 3600 + peak_times.minute * 60) - start_hour * 3600
    interval_index = (seconds_since_start // interval_s).astype(int)

    interval_rows = []
    for index in sorted(set(interval_index)):
        samples = peak_flow[interval_index == index]
        median_per_5min = np.nanmedian(samples, axis=0)
        interval_rows.append(median_per_5min * (interval_s / 300.0))
    return pd.DataFrame(interval_rows, columns=sensors.astype(str)), sensors


def sensor_to_edge() -> pd.DataFrame:
    import osmnx as ox
    import sumolib
    from sim.translate import Translator, build_way_to_edges

    coverage_csv = config.RESULTS / "sensor_coverage.csv"
    if not coverage_csv.is_file():
        raise SystemExit("Run src/data/check_sensor_coverage.py first")
    sensor_meta = pd.read_csv(coverage_csv)

    road_graph = ox.load_graphml(config.GRAPHML)
    sumo_net = sumolib.net.readNet(str(config.NET_XML), withInternal=True)
    translator = Translator(sumo_net, road_graph,
                            build_way_to_edges(sumo_net, config.NET_XML))
    freeway_classes = {"motorway", "motorway_link", "trunk", "trunk_link"}

    def _is_freeway(edge_data) -> bool:
        highway_tag = edge_data.get("highway")
        return any(tag in freeway_classes for tag in
                   (highway_tag if isinstance(highway_tag, list) else [highway_tag]))

    freeway_subgraph = road_graph.edge_subgraph(
        [(from_node, to_node, edge_key) for from_node, to_node, edge_key, edge_data
         in road_graph.edges(keys=True, data=True) if _is_freeway(edge_data)])
    nearest_graph_edges = ox.distance.nearest_edges(
        freeway_subgraph, X=sensor_meta["Lng"].to_numpy(),
        Y=sensor_meta["Lat"].to_numpy())
    sensor_edge_rows = []
    for (from_node, to_node, edge_key), sensor_id in zip(nearest_graph_edges, sensor_meta["ID"]):
        edge_data = road_graph.edges[from_node, to_node, edge_key]
        matched_edges = translator.match_edges(from_node, to_node, edge_data)
        if matched_edges:
            sensor_edge_rows.append({"ID": sensor_id, "edge": matched_edges[0]})
    mapping = pd.DataFrame(sensor_edge_rows)
    return mapping


def find_arterial_count_edges(street_name: str, per_direction: int = 4) -> list[str]:
    import osmnx as ox
    import sumolib
    from sim.translate import Translator, build_way_to_edges

    road_graph = ox.load_graphml(config.GRAPHML)
    sumo_net = sumolib.net.readNet(str(config.NET_XML), withInternal=True)
    translator = Translator(sumo_net, road_graph,
                            build_way_to_edges(sumo_net, config.NET_XML))

    matches: list[tuple[float, str]] = []
    for from_node, to_node, edge_key, edge_data in road_graph.edges(keys=True, data=True):
        name_tag = edge_data.get("name")
        name_tags = name_tag if isinstance(name_tag, list) else [name_tag]
        if not any(isinstance(value, str) and street_name.lower() in value.lower()
                   for value in name_tags):
            continue
        matched_edges = translator.match_edges(from_node, to_node, edge_data)
        if matched_edges:
            matches.append((road_graph.nodes[from_node]["y"], matched_edges[0]))

    if not matches:
        print(f"[arterial] WARNING: no edges named '{street_name}' found in the graph")
        return []

    matches.sort()
    seen_ids, sampled_edges = set(), []
    stride = max(1, len(matches) // (per_direction * 2))
    for index in range(0, len(matches), stride):
        edge_id = matches[index][1]
        if edge_id not in seen_ids:
            seen_ids.add(edge_id)
            sampled_edges.append(edge_id)
    return sampled_edges


def write_counts(flow_profile: pd.DataFrame, sensor_edge_map: pd.DataFrame,
                 start_hour: int, interval_s: int, arterial_aadt: int,
                 arterial_edge_ids: list[str]) -> None:
    arterial_per_interval = int(round(
        arterial_aadt * ARTERIAL_PEAK_FRACTION / 2.0 * (interval_s / 3600.0)))
    lines = ['<data>']
    for index in range(len(flow_profile)):
        begin_s = index * interval_s
        lines.append(f'  <interval id="i{index}" begin="{begin_s}" end="{begin_s + interval_s}">')
        counts_by_edge: dict[str, list[float]] = {}
        for _, row in sensor_edge_map.iterrows():
            sensor_column = str(row["ID"])
            if sensor_column not in flow_profile.columns:
                continue
            count = flow_profile.iloc[index][sensor_column]
            if not np.isfinite(count) or count <= 0:
                continue
            counts_by_edge.setdefault(row["edge"], []).append(float(count))
        for edge_id, counts in counts_by_edge.items():
            lines.append(f'    <edge id="{edge_id}" '
                         f'entered="{int(round(np.mean(counts)))}"/>')
        for edge_id in arterial_edge_ids:
            lines.append(f'    <edge id="{edge_id}" entered="{arterial_per_interval}"/>')
        lines.append('  </interval>')
    lines.append('</data>')
    COUNTS_XML.write_text("\n".join(lines), encoding="utf-8")


def run_sampler(start_hour: int, end_hour: int, seed: int, period: float,
                fringe_factor: float | None, min_distance: float | None) -> int:
    duration_s = (end_hour - start_hour) * 3600
    command = [sys.executable, str(config.sumo_tool("randomTrips.py")),
               "-n", str(config.NET_XML),
               "-o", str(config.SUMO_DIR / f"{config.SUMO_PREFIX}candidates.trips.xml"),
               "-r", str(CANDIDATES_ROU), "-b", "0", "-e", str(duration_s),
               "--period", str(period), "--validate", "--seed", str(seed),
               "--vehicle-class", "passenger"]
    if fringe_factor is not None:
        command += ["--fringe-factor", str(fringe_factor)]
    if min_distance is not None:
        command += ["--min-distance", str(min_distance)]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr[-2500:])
        return 1

    result = subprocess.run(
        [sys.executable, str(config.sumo_tool("routeSampler.py")),
         "-r", str(CANDIDATES_ROU), "--edgedata-files", str(COUNTS_XML),
         "-o", str(DEMAND_ROU), "--seed", str(seed),
         "--edgedata-attribute", "entered", "--optimize", "full"],
        capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr[-2500:])
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hours", nargs=2, type=int, default=[7, 9],
                        metavar=("START", "END"))
    parser.add_argument("--interval", type=int, default=900, help="seconds")
    parser.add_argument("--arterial-aadt", type=int, default=DEFAULT_ARTERIAL_AADT)
    parser.add_argument("--arterial-name", default=config.ARTERIAL_NAME,
                        help="street name to carry AADT-derived counts")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--period", type=float, default=0.2,
                        help="randomTrips spacing; smaller = more candidate routes")
    parser.add_argument("--fringe-factor", type=float, default=None,
                        help="bias trip endpoints to boundary edges (through traffic)")
    parser.add_argument("--min-distance", type=float, default=None,
                        help="minimum candidate trip length in metres")
    args = parser.parse_args()

    if not GBA_FLOW_NPZ.is_file():
        raise SystemExit("Run src/data/build_gba_subset.py first")

    start_hour, end_hour = args.hours
    flow_profile, _ = load_profile(start_hour, end_hour, args.interval)
    sensor_edge_map = sensor_to_edge()
    arterial_edge_ids = find_arterial_count_edges(args.arterial_name)
    write_counts(flow_profile, sensor_edge_map, start_hour, args.interval,
                 args.arterial_aadt, arterial_edge_ids)
    return run_sampler(start_hour, end_hour, args.seed, args.period,
                       args.fringe_factor, args.min_distance)


if __name__ == "__main__":
    sys.exit(main())
