from __future__ import annotations

import sys
from pathlib import Path

import networkx as nx
import osmnx as ox

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config


def _is_drivable(highway_tag) -> bool:
    if isinstance(highway_tag, list):
        return any(tag in config.DRIVABLE_HIGHWAY_CLASSES for tag in highway_tag)
    return highway_tag in config.DRIVABLE_HIGHWAY_CLASSES


def build() -> nx.MultiDiGraph:
    if not config.RAW_OSM.is_file():
        raise SystemExit("No extract found. Run: python src/graph/fetch_osm.py")

    road_graph = ox.graph_from_xml(config.RAW_OSM, bidirectional=False, simplify=True, retain_all=True)

    non_drivable_edges = [(from_node, to_node, edge_key)
                          for from_node, to_node, edge_key, edge_data
                          in road_graph.edges(keys=True, data=True)
                          if not _is_drivable(edge_data.get("highway"))]
    road_graph.remove_edges_from(non_drivable_edges)
    road_graph.remove_nodes_from(list(nx.isolates(road_graph)))
    road_graph = ox.truncate.largest_component(road_graph, strongly=True)
    road_graph = ox.add_edge_speeds(road_graph)
    road_graph = ox.add_edge_travel_times(road_graph)

    ox.save_graphml(road_graph, config.GRAPHML)

    return road_graph


if __name__ == "__main__":
    build()
