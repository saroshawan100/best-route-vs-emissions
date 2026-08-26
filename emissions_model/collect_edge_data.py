from __future__ import annotations
from pathlib import Path

import argparse
import subprocess
import sys
import xml.etree.ElementTree as ET
import pandas as pd
import sumolib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

EDGEDATA_ADDITIONAL_XML = config.SUMO_DIR / "edgedata.add.xml"
EDGE_TRAFFIC_XML = config.SUMO_DIR / "edge_traffic.xml"
EDGE_EMISSIONS_XML = config.SUMO_DIR / "edge_emissions.xml"
TRAINING_DATA_CSV = config.RESULTS / "emissions_training_data.csv"


def write_additional(period: int) -> None:
    EDGEDATA_ADDITIONAL_XML.write_text(
        '<additional>\n'
        f'  <edgeData id="traffic" file="{EDGE_TRAFFIC_XML.name}" period="{period}" '
        'excludeEmpty="true"/>\n'
        f'  <edgeData id="emis" type="emissions" file="{EDGE_EMISSIONS_XML.name}" '
        f'period="{period}" excludeEmpty="true"/>\n'
        '</additional>\n', encoding="utf-8")


def run_sim(hours: tuple[int, int], period: int, seed: int) -> int:
    demand_file = config.DEMAND_ROU
    if not demand_file.is_file():
        raise SystemExit(f"No {demand_file.name} -- run src/sim/gen_demand.py first")
    duration_s = (hours[1] - hours[0]) * 3600
    write_additional(period)

    command = [config.sumo_bin("sumo"), "-n", str(config.NET_XML), "-r", str(demand_file),
               "-a", str(EDGEDATA_ADDITIONAL_XML), "--begin", "0", "--end", str(duration_s + 1800),
               "--seed", str(seed), "--no-step-log", "true",
               "--time-to-teleport", "300",
               "--device.emissions.probability", "1"]
    result = subprocess.run(command, capture_output=True, text=True, cwd=str(config.SUMO_DIR))
    if result.returncode != 0:
        print(result.stderr[-3000:])
        return 1
    return 0


def parse_edgedata(path: Path, attribute_map: dict[str, str]) -> pd.DataFrame:
    rows = []
    for _, element in ET.iterparse(str(path), events=("end",)):
        if element.tag != "interval":
            continue
        begin_s = float(element.get("begin", 0))
        for edge_element in element.findall("edge"):
            record = {"edge": edge_element.get("id"), "interval": begin_s}
            for xml_attribute, column_name in attribute_map.items():
                raw_value = edge_element.get(xml_attribute)
                record[column_name] = float(raw_value) if raw_value is not None else None
            rows.append(record)
        element.clear()
    return pd.DataFrame(rows)


def static_features() -> pd.DataFrame:
    sumo_net = sumolib.net.readNet(str(config.NET_XML), withInternal=False)
    signalised_node_ids = set()
    for traffic_light in sumo_net.getTrafficLights():
        for connection in traffic_light.getConnections():
            try:
                signalised_node_ids.add(connection[0].getEdge().getToNode().getID())
            except Exception:
                pass

    rows = []
    for sumo_edge in sumo_net.getEdges():
        to_node = sumo_edge.getToNode()
        rows.append({
            "edge": sumo_edge.getID(),
            "length": sumo_edge.getLength(),
            "lanes": sumo_edge.getLaneNumber(),
            "speed_limit": sumo_edge.getSpeed(),
            "priority": sumo_edge.getPriority(),
            "road_class": sumo_edge.getType() or "unknown",
            "signalised": int(to_node.getType() == "traffic_light"
                              or to_node.getID() in signalised_node_ids),
            "n_outgoing": len(sumo_edge.getOutgoing()),
        })
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hours", nargs=2, type=int, default=[7, 9])
    parser.add_argument("--period", type=int, default=900)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if run_sim(tuple(args.hours), args.period, args.seed) != 0:
        return 1

    traffic = parse_edgedata(EDGE_TRAFFIC_XML, {
        "speed": "speed", "density": "density", "occupancy": "occupancy",
        "waitingTime": "waiting_time", "traveltime": "travel_time",
        "sampledSeconds": "sampled_seconds", "left": "left", "entered": "entered",
    })
    emissions = parse_edgedata(EDGE_EMISSIONS_XML, {
        "CO2_abs": "co2_abs", "NOx_abs": "nox_abs", "PMx_abs": "pmx_abs",
        "fuel_abs": "fuel_abs", "CO2_normed": "co2_normed",
        "sampledSeconds": "emis_sampled_seconds",
    })

    samples = traffic.merge(emissions, on=["edge", "interval"], how="inner")
    samples = samples.merge(static_features(), on="edge", how="left")

    vehicles_entered = samples["entered"].replace(0, pd.NA)
    samples["co2_per_veh"] = samples["co2_abs"] / 1000.0 / vehicles_entered
    samples["nox_per_veh"] = samples["nox_abs"] / 1000.0 / vehicles_entered
    samples["co2_per_veh_km"] = samples["co2_per_veh"] / (samples["length"] / 1000.0)
    samples["speed_ratio"] = samples["speed"] / samples["speed_limit"].replace(0, pd.NA)

    samples = samples.dropna(subset=["co2_per_veh", "speed", "length"])
    samples = samples[(samples["entered"] >= 3) & (samples["length"] >= 5)]

    config.RESULTS.mkdir(parents=True, exist_ok=True)
    samples.to_csv(TRAINING_DATA_CSV, index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
