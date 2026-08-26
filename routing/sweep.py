from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import math
import sys
from pathlib import Path

import networkx as nx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from routing.costs import EdgeCostModel

ROUTES_JSON = config.RESULTS / "routes.json"
ALPHA_VALUES = [round(step * 0.1, 1) for step in range(11)]

PLACES = config.PLACES
OD_PAIRS = config.OD_PAIRS

MAX_SPEED_MS = 110 / 3.6


def _great_circle_m(road_graph, from_node, to_node) -> float:
    from_lat, from_lon = road_graph.nodes[from_node]["y"], road_graph.nodes[from_node]["x"]
    to_lat, to_lon = road_graph.nodes[to_node]["y"], road_graph.nodes[to_node]["x"]
    deg_to_rad = math.pi / 180
    haversine = (0.5 - math.cos((to_lat - from_lat) * deg_to_rad) / 2
                 + math.cos(from_lat * deg_to_rad) * math.cos(to_lat * deg_to_rad)
                 * (1 - math.cos((to_lon - from_lon) * deg_to_rad)) / 2)
    return 12742000 * math.asin(math.sqrt(haversine))


def astar_time_dependent(cost_model: EdgeCostModel, origin, destination, alpha: float,
                         time_norm: float, co2_norm: float) -> list | None:
    road_graph = cost_model.road_graph
    heuristic_scale = alpha / (MAX_SPEED_MS * time_norm) if time_norm > 0 else 0.0
    start_heuristic = _great_circle_m(road_graph, origin, destination) * heuristic_scale
    best_cost: dict = {origin: 0.0}
    elapsed_s: dict = {origin: 0.0}
    came_from: dict = {}
    frontier = [(start_heuristic, 0.0, origin)]
    while frontier:
        priority, cost_so_far, node = heapq.heappop(frontier)
        if node == destination:
            path = [node]
            while path[-1] != origin:
                path.append(came_from[path[-1]])
            return path[::-1]
        if cost_so_far > best_cost.get(node, float("inf")):
            continue
        arrival_s = elapsed_s[node]
        for _, next_node, edge_key in road_graph.out_edges(node, keys=True):
            graph_edge = (node, next_node, edge_key)
            edge_time_s = cost_model.travel_time_s(graph_edge, arrival_s)
            edge_co2_g = cost_model.co2_grams(graph_edge, arrival_s)
            next_cost = (cost_so_far + alpha * edge_time_s / time_norm
                         + (1 - alpha) * edge_co2_g / co2_norm)
            if next_cost < best_cost.get(next_node, float("inf")) - 1e-12:
                best_cost[next_node] = next_cost
                elapsed_s[next_node] = arrival_s + edge_time_s
                came_from[next_node] = node
                heapq.heappush(frontier,
                               (next_cost + _great_circle_m(road_graph, next_node, destination)
                                * heuristic_scale, next_cost, next_node))
    return None


def compute_route_id(node_path: list) -> str:
    return hashlib.sha1(",".join(map(str, node_path)).encode()).hexdigest()[:12]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="am_peak")
    parser.add_argument("--departure-s", type=int, default=1800,
                        help="seconds after 07:00 (1800 = 07:30)")
    args = parser.parse_args()

    import osmnx as ox
    cost_model = EdgeCostModel(scenario=args.scenario, departure_s=args.departure_s)
    road_graph = cost_model.road_graph

    node_by_place = {name: ox.distance.nearest_nodes(road_graph, X=lon, Y=lat)
                     for name, (lat, lon) in PLACES.items()}

    records = []
    for origin_name, destination_name in OD_PAIRS:
        origin, destination = node_by_place[origin_name], node_by_place[destination_name]
        od_label = f"{origin_name}->{destination_name}"

        fastest_path = astar_time_dependent(cost_model, origin, destination, 1.0, 1.0, 1.0)
        cleanest_path = astar_time_dependent(cost_model, origin, destination, 0.0, 1.0, 1.0)
        if fastest_path is None or cleanest_path is None:
            print(f"[!!] {od_label}: unreachable, skipped")
            continue
        best_time_s = cost_model.route_totals(fastest_path)[0]
        best_co2_g = cost_model.route_totals(cleanest_path)[1]

        for alpha in ALPHA_VALUES:
            node_path = astar_time_dependent(cost_model, origin, destination, alpha,
                                             best_time_s, best_co2_g)
            if node_path is None:
                continue
            route_id = compute_route_id(node_path)
            pred_time_s, pred_co2_g, length_m = cost_model.route_totals(node_path)
            records.append({
                "od": od_label, "kind": "sweep", "alpha": alpha, "route_id": route_id,
                "pred_time_s": round(pred_time_s, 1), "pred_co2_g": round(pred_co2_g, 1),
                "length_m": round(length_m, 0), "node_path": node_path,
            })

        for baseline_kind, weight_attribute in (("shortest_dist", "length"),
                                                ("fastest_static", "travel_time")):
            node_path = nx.shortest_path(road_graph, origin, destination, weight=weight_attribute)
            pred_time_s, pred_co2_g, length_m = cost_model.route_totals(node_path)
            records.append({
                "od": od_label, "kind": baseline_kind, "alpha": None,
                "route_id": compute_route_id(node_path),
                "pred_time_s": round(pred_time_s, 1), "pred_co2_g": round(pred_co2_g, 1),
                "length_m": round(length_m, 0), "node_path": node_path,
            })

    ROUTES_JSON.write_text(json.dumps({
        "scenario": args.scenario, "departure_s": args.departure_s,
        "routes": records}, indent=1), encoding="utf-8")

    alpha_changes_route = any(
        {record["route_id"] for record in records
         if record["od"] == od_label and record["alpha"] == 0.0}
        != {record["route_id"] for record in records
            if record["od"] == od_label and record["alpha"] == 1.0}
        for od_label in {record["od"] for record in records})
    if not alpha_changes_route:
        print("[WARN] alpha=0 and alpha=1 give identical routes on EVERY O-D "
              "pair -- the cost function is not trading time against emissions")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
