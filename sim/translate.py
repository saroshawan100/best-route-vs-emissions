from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import networkx as nx
import osmnx as ox
import sumolib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
def way_id_of(sumo_edge_id: str) -> str | None:
    if sumo_edge_id.startswith(":"):
        return None
    way_id = sumo_edge_id[1:] if sumo_edge_id.startswith("-") else sumo_edge_id
    way_id = way_id.split("#")[0]
    return way_id or None


def build_way_to_edges(sumo_net: sumolib.net.Net,
                       net_xml_path: Path | None = None) -> dict[str, list[str]]:
    way_to_edges: dict[str, set[str]] = defaultdict(set)
    for sumo_edge in sumo_net.getEdges():
        way_id = way_id_of(sumo_edge.getID())
        if way_id:
            way_to_edges[way_id].add(sumo_edge.getID())

    if net_xml_path and net_xml_path.is_file():
        import xml.etree.ElementTree as ET
        for _, element in ET.iterparse(str(net_xml_path), events=("end",)):
            if element.tag != "edge":
                continue
            edge_id = element.get("id", "")
            if not edge_id.startswith(":"):
                for lane in element.findall("lane"):
                    for param in lane.findall("param"):
                        if param.get("key") == "origId":
                            for way_id in (param.get("value") or "").split():
                                way_to_edges[way_id].add(edge_id)
            element.clear()

    return {way_id: sorted(edge_ids) for way_id, edge_ids in way_to_edges.items()}

def _bearing(from_x, from_y, to_x, to_y) -> float:
    return math.degrees(math.atan2(to_y - from_y, to_x - from_x)) % 360.0


def _bearing_delta(bearing_a: float, bearing_b: float) -> float:
    delta = abs(bearing_a - bearing_b) % 360.0
    return min(delta, 360.0 - delta)


def _edge_endpoints(sumo_edge):
    shape = sumo_edge.getShape()
    start_x, start_y = shape[0]
    end_x, end_y = shape[-1]
    return start_x, start_y, end_x, end_y, ((start_x + end_x) / 2, (start_y + end_y) / 2)


class Translator:
    def __init__(self, sumo_net: sumolib.net.Net, road_graph: nx.MultiDiGraph,
                 way_to_edges: dict[str, list[str]]):
        self.sumo_net, self.road_graph, self.way_to_edges = sumo_net, road_graph, way_to_edges

    def _xy(self, node_id):
        node_data = self.road_graph.nodes[node_id]
        return self.sumo_net.convertLonLat2XY(node_data["x"], node_data["y"])

    def match_edges(self, from_node, to_node, edge_data) -> list[str]:
        osmid_value = edge_data.get("osmid")
        osm_way_ids = osmid_value if isinstance(osmid_value, list) else [osmid_value]
        candidate_edge_ids: list[str] = []
        for osm_way_id in osm_way_ids:
            candidate_edge_ids.extend(self.way_to_edges.get(str(osm_way_id), []))
        if not candidate_edge_ids:
            return []

        from_x, from_y = self._xy(from_node)
        to_x, to_y = self._xy(to_node)
        axis_x, axis_y = to_x - from_x, to_y - from_y
        segment_length = math.hypot(axis_x, axis_y)
        if segment_length < 1e-6:
            return []
        target_bearing = _bearing(from_x, from_y, to_x, to_y)
        lateral_tolerance = max(100.0, 0.5 * segment_length)

        scored_edges: list[tuple[float, str]] = []
        for edge_id in set(candidate_edge_ids):
            sumo_edge = self.sumo_net.getEdge(edge_id)
            start_x, start_y, end_x, end_y, midpoint = _edge_endpoints(sumo_edge)
            if _bearing_delta(target_bearing, _bearing(start_x, start_y, end_x, end_y)) > 90:
                continue
            along_axis = ((midpoint[0] - from_x) * axis_x
                          + (midpoint[1] - from_y) * axis_y) / (segment_length ** 2)
            lateral_offset = abs((midpoint[0] - from_x) * (-axis_y)
                                 + (midpoint[1] - from_y) * axis_x) / segment_length
            if -0.15 <= along_axis <= 1.15 and lateral_offset <= lateral_tolerance:
                scored_edges.append((along_axis, edge_id))

        scored_edges.sort()
        return [edge_id for _, edge_id in scored_edges]

    def _corridor(self, node_path: list[int]) -> tuple[set[str], int]:
        corridor_edge_ids: set[str] = set()
        n_unmatched = 0
        for from_node, to_node in zip(node_path, node_path[1:]):
            edge_data = min(self.road_graph.get_edge_data(from_node, to_node).values(),
                            key=lambda candidate: candidate.get("length", 0))
            osmid_value = edge_data.get("osmid")
            osm_way_ids = osmid_value if isinstance(osmid_value, list) else [osmid_value]
            matched = False
            for osm_way_id in osm_way_ids:
                for edge_id in self.way_to_edges.get(str(osm_way_id), []):
                    corridor_edge_ids.add(edge_id)
                    matched = True
            if not matched:
                n_unmatched += 1
        return corridor_edge_ids, n_unmatched
    OFF_CORRIDOR_PENALTY = 8.0

    def _dijkstra(self, start_edge_id: str, goal_edge_id: str,
                  corridor_edge_ids: set[str]) -> list[str] | None:
        if start_edge_id == goal_edge_id:
            return [start_edge_id]
        import heapq
        best_distance = {start_edge_id: 0.0}
        came_from: dict[str, str] = {}
        frontier = [(0.0, start_edge_id)]
        while frontier:
            distance, edge_id = heapq.heappop(frontier)
            if edge_id == goal_edge_id:
                edge_sequence = [edge_id]
                while edge_sequence[-1] != start_edge_id:
                    edge_sequence.append(came_from[edge_sequence[-1]])
                return edge_sequence[::-1]
            if distance > best_distance.get(edge_id, float("inf")):
                continue
            for next_edge in self.sumo_net.getEdge(edge_id).getOutgoing():
                next_edge_id = next_edge.getID()
                penalty = 1.0 if next_edge_id in corridor_edge_ids else self.OFF_CORRIDOR_PENALTY
                next_distance = distance + next_edge.getLength() * penalty
                if next_distance < best_distance.get(next_edge_id, float("inf")):
                    best_distance[next_edge_id] = next_distance
                    came_from[next_edge_id] = edge_id
                    heapq.heappush(frontier, (next_distance, next_edge_id))
        return None

    def _endpoint(self, node_path: list[int], at_start: bool) -> str | None:
        node_pairs = list(zip(node_path, node_path[1:]))
        if not at_start:
            node_pairs.reverse()
        for from_node, to_node in node_pairs[:10]:
            edge_data = min(self.road_graph.get_edge_data(from_node, to_node).values(),
                            key=lambda candidate: candidate.get("length", 0))
            edge_ids = self.match_edges(from_node, to_node, edge_data)
            if edge_ids:
                return edge_ids[0] if at_start else edge_ids[-1]
        return None

    def translate(self, node_path: list[int]) -> tuple[list[str], dict]:
        corridor_edge_ids, n_unmatched = self._corridor(node_path)
        start_edge = self._endpoint(node_path, True)
        goal_edge = self._endpoint(node_path, False)
        if start_edge is None or goal_edge is None:
            return [], {"unmatched_osmnx_edges": n_unmatched, "escaped_corridor": 0,
                        "route_edges": 0, "note": "no endpoint match"}

        sumo_route = self._dijkstra(start_edge, goal_edge, corridor_edge_ids)
        if sumo_route is None:
            return [], {"unmatched_osmnx_edges": n_unmatched, "off_corridor_frac": 1.0,
                        "route_edges": 0, "note": "unreachable in SUMO net"}
        off_corridor_m = sum(self.sumo_net.getEdge(edge_id).getLength()
                             for edge_id in sumo_route
                             if edge_id not in corridor_edge_ids)
        total_m = sum(self.sumo_net.getEdge(edge_id).getLength()
                      for edge_id in sumo_route) or 1.0

        return sumo_route, {"unmatched_osmnx_edges": n_unmatched,
                            "off_corridor_frac": off_corridor_m / total_m,
                            "route_edges": len(sumo_route)}

def _junction_length_m(sumo_net: sumolib.net.Net, sumo_route: list[str]) -> float:
    total = 0.0
    for edge_id, next_edge_id in zip(sumo_route, sumo_route[1:]):
        edge, next_edge = sumo_net.getEdge(edge_id), sumo_net.getEdge(next_edge_id)
        longest_via = 0.0
        for connection in edge.getConnections(next_edge):
            via_lane_id = connection.getViaLaneID()
            if via_lane_id:
                try:
                    longest_via = max(longest_via, sumo_net.getLane(via_lane_id).getLength())
                except KeyError:
                    pass
        total += longest_via
    return total


def validate(n_routes: int = 25, seed: int = 0) -> bool:
    if not config.NET_XML.is_file():
        raise SystemExit("No SUMO net. Run: python src/sim/build_net.py")
    if not config.GRAPHML.is_file():
        raise SystemExit("No routing graph. Run: python src/graph/build_graph.py")

    sumo_net = sumolib.net.readNet(str(config.NET_XML), withInternal=True)
    road_graph = ox.load_graphml(config.GRAPHML)
    way_to_edges = build_way_to_edges(sumo_net, config.NET_XML)

    graph_way_ids = set()
    for _, _, edge_data in road_graph.edges(data=True):
        osmid_value = edge_data.get("osmid")
        graph_way_ids.update(str(way_id) for way_id in
                             (osmid_value if isinstance(osmid_value, list) else [osmid_value]))
    n_covered = sum(1 for way_id in graph_way_ids if way_id in way_to_edges)
    coverage_fraction = n_covered / max(len(graph_way_ids), 1)

    translator = Translator(sumo_net, road_graph, way_to_edges)
    rng = random.Random(seed)
    graph_nodes = list(road_graph.nodes)

    n_passed = n_failed = 0
    n_attempts = 0
    while n_passed + n_failed < n_routes and n_attempts < n_routes * 20:
        n_attempts += 1
        origin, destination = rng.sample(graph_nodes, 2)
        try:
            node_path = nx.shortest_path(road_graph, origin, destination, weight="travel_time")
        except nx.NetworkXNoPath:
            continue
        if len(node_path) < 8:
            continue

        sumo_route, diagnostics = translator.translate(node_path)
        if len(sumo_route) < 2:
            n_failed += 1
            continue

        osm_length_m = sum(
            min(parallel_edge.get("length", 0)
                for parallel_edge in road_graph.get_edge_data(from_node, to_node).values())
            for from_node, to_node in zip(node_path, node_path[1:]))
        sumo_length_m = (sum(sumo_net.getEdge(edge_id).getLength() for edge_id in sumo_route)
                         + _junction_length_m(sumo_net, sumo_route))
        length_ratio = sumo_length_m / osm_length_m if osm_length_m else float("inf")
        if 0.85 <= length_ratio <= 1.15 and diagnostics["off_corridor_frac"] <= 0.15:
            n_passed += 1
        else:
            n_failed += 1
            print(f"  [!] len ratio {length_ratio:.2f}  unmatched={diagnostics['unmatched_osmnx_edges']}"
                  f"/{len(node_path)-1}  off_corridor={diagnostics['off_corridor_frac']:.1%}")

    config.EDGE_MAP.write_text(json.dumps(way_to_edges, indent=0), encoding="utf-8")

    is_trustworthy = n_passed >= 0.8 * max(n_passed + n_failed, 1) and coverage_fraction >= 0.5
    return is_trustworthy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    sys.exit(0 if validate(args.n, args.seed) else 1)
