from __future__ import annotations

import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import networkx as nx
import osmnx as ox
import sumolib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from sim.translate import Translator, build_way_to_edges
EMISSION_CLASS = "HBEFA3/PC_G_EU4"


def main(with_demand: bool = False) -> int:
    sumo_net = sumolib.net.readNet(str(config.NET_XML), withInternal=True)
    road_graph = ox.load_graphml(config.GRAPHML)
    translator = Translator(sumo_net, road_graph,
                            build_way_to_edges(sumo_net, config.NET_XML))

    graph_nodes = list(road_graph.nodes)
    origin = min(graph_nodes,
                 key=lambda node: (road_graph.nodes[node]["x"], -road_graph.nodes[node]["y"]))
    destination = max(graph_nodes,
                      key=lambda node: (road_graph.nodes[node]["x"], -road_graph.nodes[node]["y"]))
    node_path = nx.shortest_path(road_graph, origin, destination, weight="travel_time")
    sumo_route, diagnostics = translator.translate(node_path)
    if len(sumo_route) < 2:
        print("[fail] translation produced no route")
        return 1

    route_file = config.SUMO_DIR / "smoke.rou.xml"
    route_file.write_text(
        '<routes>\n'
        f'  <vType id="car" vClass="passenger" emissionClass="{EMISSION_CLASS}"/>\n'
        '  <vehicle id="ego" type="car" depart="0">\n'
        f'    <route edges="{" ".join(sumo_route)}"/>\n'
        '  </vehicle>\n'
        '</routes>\n', encoding="utf-8")

    demand_file = config.DEMAND_ROU
    if with_demand:
        if not demand_file.is_file():
            raise SystemExit(f"No {demand_file.name} -- run src/sim/gen_demand.py")
        route_files = f"{route_file},{demand_file}"
    else:
        route_files = str(route_file)

    run_tag = "withdemand" if with_demand else "empty"
    tripinfo_path = config.SUMO_DIR / f"smoke.{run_tag}.tripinfo.xml"
    command = [config.sumo_bin("sumo"), "-n", str(config.NET_XML), "-r", route_files,
               "--tripinfo-output", str(tripinfo_path),
               "--device.emissions.probability", "1",
               "--no-step-log", "true", "--duration-log.statistics", "true",
               "--time-to-teleport", "-1", "--end", "7200"]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr[-2000:])
        return 1

    tripinfo_root = ET.parse(tripinfo_path).getroot()
    ego_tripinfo = next((entry for entry in tripinfo_root.findall("tripinfo")
                         if entry.get("id") == "ego"), None)
    if ego_tripinfo is None:
        print("[fail] ego vehicle never completed the trip")
        return 1

    return 0


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-demand", action="store_true",
                        help="include calibrated background traffic")
    sys.exit(main(parser.parse_args().with_demand))
