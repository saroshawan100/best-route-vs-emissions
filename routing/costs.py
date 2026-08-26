from __future__ import annotations

import math
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

EDGE_STATE_CSV = config.RESULTS / "emissions_training_data.csv"
EMISSIONS_MODEL_JOBLIB = config.RESULTS / "emissions_model.joblib"
FORECAST_SPEEDS_CSV = config.RESULTS / "forecast_speeds.csv"
SENSOR_COVERAGE_CSV = config.RESULTS / "sensor_coverage.csv"

SIM_START_HOUR = 7
INTERVAL_S = 900
N_TIME_SLOTS = 8
FREEWAY_CLASSES = {"motorway", "motorway_link", "trunk", "trunk_link"}

NETCONVERT_PRIORITY = {"motorway": 13, "motorway_link": 12, "trunk": 11, "trunk_link": 10,
                       "primary": 9, "primary_link": 8, "secondary": 7,
                       "secondary_link": 6, "tertiary": 6, "tertiary_link": 5,
                       "unclassified": 4, "residential": 3, "living_street": 2}
DEFAULT_LANES_BY_CLASS = {"motorway": 4, "trunk": 3, "primary": 2, "secondary": 2}


def _first_value(tag_value):
    return tag_value[0] if isinstance(tag_value, list) else tag_value


def _compass_direction(bearing: float) -> str:
    return "NESW"[int(((bearing + 45) % 360) // 90)]


class EdgeCostModel:

    def __init__(self, scenario: str = "am_peak", departure_s: int = 1800):
        for required_path, how_to_build in (
                (EDGE_STATE_CSV, "src/emissions_model/collect_edge_data.py"),
                (EMISSIONS_MODEL_JOBLIB, "src/emissions_model/train_emissions_model.py"),
                (FORECAST_SPEEDS_CSV, "src/forecast/predict.py")):
            if not required_path.is_file():
                raise SystemExit(f"missing {required_path.name} -- run {how_to_build} first")
        import joblib
        import osmnx as ox
        import sumolib
        from sim.translate import Translator, build_way_to_edges

        self.departure_s = departure_s
        self.road_graph = ox.load_graphml(config.GRAPHML)
        sumo_net = sumolib.net.readNet(str(config.NET_XML), withInternal=True)
        translator = Translator(sumo_net, self.road_graph,
                                build_way_to_edges(sumo_net, config.NET_XML))

        bundle = joblib.load(EMISSIONS_MODEL_JOBLIB)
        self._model = bundle["model"]
        self._feature_names = bundle["features"]
        self._categorical_features = bundle["categorical"]

        edge_state = pd.read_csv(EDGE_STATE_CSV)
        edge_state["slot"] = (edge_state["interval"] // INTERVAL_S).astype(int)
        self._state_by_edge_slot = edge_state.set_index(["edge", "slot"])[
            ["speed", "density", "occupancy"]].sort_index()
        self._median_state_by_edge = edge_state.groupby("edge")[
            ["speed", "density", "occupancy"]].median()
        self._global_median_state = edge_state[["density", "occupancy"]].median()

        sys.path.insert(0, str(config.ROOT / "src"))
        from emissions_model.collect_edge_data import static_features
        self._static_features = static_features().set_index("edge")

        self._speed_by_sensor = self._load_forecast(scenario)
        self._build_profiles(translator)

    def _load_forecast(self, scenario: str) -> dict:
        forecast_rows = pd.read_csv(FORECAST_SPEEDS_CSV)
        forecast_rows = forecast_rows[forecast_rows["scenario"] == scenario]
        if forecast_rows.empty:
            raise SystemExit(f"no rows for scenario '{scenario}' in {FORECAST_SPEEDS_CSV.name}")
        speed_by_sensor: dict[int, np.ndarray] = {}
        departure_slot = self.departure_s // INTERVAL_S
        for sensor_id, sensor_rows in forecast_rows.groupby("sensor"):
            sensor_rows = sensor_rows.sort_values("minutes_ahead")
            step_speeds_ms = sensor_rows["speed_kmh"].to_numpy() / 3.6
            slot_speeds = np.empty(N_TIME_SLOTS)
            for slot in range(N_TIME_SLOTS):
                step_index = max(0, min(slot - departure_slot, len(step_speeds_ms) - 1))
                slot_speeds[slot] = step_speeds_ms[step_index]
            speed_by_sensor[int(sensor_id)] = slot_speeds
        return speed_by_sensor

    def _sensor_assignment(self) -> dict:
        import osmnx as ox
        if not SENSOR_COVERAGE_CSV.is_file():
            raise SystemExit("missing sensor_coverage.csv -- run "
                             "src/data/check_sensor_coverage.py")
        sensor_meta = pd.read_csv(SENSOR_COVERAGE_CSV)
        sensor_meta = sensor_meta[sensor_meta["ID"].isin(self._speed_by_sensor)]
        freeway_edges, edge_directions = [], []
        for from_node, to_node, edge_key, edge_data in self.road_graph.edges(keys=True, data=True):
            if _first_value(edge_data.get("highway")) in FREEWAY_CLASSES:
                freeway_edges.append((from_node, to_node, edge_key))
                bearing = math.degrees(math.atan2(
                    self.road_graph.nodes[to_node]["x"] - self.road_graph.nodes[from_node]["x"],
                    self.road_graph.nodes[to_node]["y"] - self.road_graph.nodes[from_node]["y"])) % 360
                edge_directions.append(_compass_direction(bearing))
        if not freeway_edges:
            return {}
        sensor_by_edge = {}
        for direction in "NESW":
            direction_sensors = sensor_meta[sensor_meta["Direction"] == direction]
            direction_edges = [graph_edge for graph_edge, edge_direction
                               in zip(freeway_edges, edge_directions)
                               if edge_direction == direction]
            if direction_sensors.empty or not direction_edges:
                continue
            midpoints_x = [(self.road_graph.nodes[from_node]["x"] + self.road_graph.nodes[to_node]["x"]) / 2
                           for from_node, to_node, _ in direction_edges]
            midpoints_y = [(self.road_graph.nodes[from_node]["y"] + self.road_graph.nodes[to_node]["y"]) / 2
                           for from_node, to_node, _ in direction_edges]
            sensor_lon, sensor_lat = (direction_sensors["Lng"].to_numpy(),
                                      direction_sensors["Lat"].to_numpy())
            sensor_ids = direction_sensors["ID"].to_numpy()
            for graph_edge, edge_x, edge_y in zip(direction_edges, midpoints_x, midpoints_y):
                squared_distance = (sensor_lon - edge_x) ** 2 + (sensor_lat - edge_y) ** 2
                nearest = int(np.argmin(squared_distance))
                if squared_distance[nearest] < (0.015) ** 2:
                    sensor_by_edge[graph_edge] = int(sensor_ids[nearest])
        return sensor_by_edge

    def _build_profiles(self, translator) -> None:
        sensor_by_edge = self._sensor_assignment()
        feature_rows, edge_meta = [], []
        for from_node, to_node, edge_key, edge_data in self.road_graph.edges(keys=True, data=True):
            length = float(edge_data.get("length", 0.0)) or 1.0
            highway_class = _first_value(edge_data.get("highway")) or "unclassified"
            free_flow_ms = float(edge_data.get("speed_kph", 50.0)) / 3.6

            matched_sumo_ids = translator.match_edges(from_node, to_node, edge_data)
            sumo_edge_id = matched_sumo_ids[0] if matched_sumo_ids else None
            if sumo_edge_id is not None and sumo_edge_id in self._static_features.index:
                static_row = self._static_features.loc[sumo_edge_id]
                lanes = float(static_row["lanes"]); priority = float(static_row["priority"])
                speed_limit = float(static_row["speed_limit"])
                road_class = static_row["road_class"]; n_outgoing = float(static_row["n_outgoing"])
                signalised = int(static_row["signalised"])
            else:
                try:
                    lanes = float(_first_value(edge_data.get("lanes"))
                                  or DEFAULT_LANES_BY_CLASS.get(highway_class, 1))
                except (TypeError, ValueError):
                    lanes = DEFAULT_LANES_BY_CLASS.get(highway_class, 1)
                priority = NETCONVERT_PRIORITY.get(highway_class, 4)
                speed_limit = free_flow_ms
                road_class = f"highway.{highway_class}"
                n_outgoing = float(self.road_graph.out_degree(to_node))
                signalised = int(self.road_graph.nodes[to_node].get("highway") == "traffic_signals")

            slot_speeds = np.full(N_TIME_SLOTS, free_flow_ms)
            speed_source = "maxspeed"
            if (from_node, to_node, edge_key) in sensor_by_edge:
                slot_speeds = self._speed_by_sensor[sensor_by_edge[(from_node, to_node, edge_key)]].copy()
                speed_source = "forecast"
            elif sumo_edge_id is not None:
                found_any = False
                for slot in range(N_TIME_SLOTS):
                    try:
                        slot_speeds[slot] = self._state_by_edge_slot.loc[(sumo_edge_id, slot), "speed"]
                        found_any = True
                    except KeyError:
                        pass
                if found_any:
                    speed_source = "simulated"
            slot_speeds = np.clip(slot_speeds, 1.0, max(free_flow_ms, slot_speeds.max()))

            density_by_slot = np.full(N_TIME_SLOTS, self._global_median_state["density"])
            occupancy_by_slot = np.full(N_TIME_SLOTS, self._global_median_state["occupancy"])
            if sumo_edge_id is not None and sumo_edge_id in self._median_state_by_edge.index:
                density_by_slot[:] = self._median_state_by_edge.loc[sumo_edge_id, "density"]
                occupancy_by_slot[:] = self._median_state_by_edge.loc[sumo_edge_id, "occupancy"]
                for slot in range(N_TIME_SLOTS):
                    try:
                        state_row = self._state_by_edge_slot.loc[(sumo_edge_id, slot)]
                        density_by_slot[slot], occupancy_by_slot[slot] = (state_row["density"],
                                                                          state_row["occupancy"])
                    except KeyError:
                        pass

            for slot in range(N_TIME_SLOTS):
                feature_rows.append({
                    "length": length, "lanes": lanes, "speed_limit": speed_limit,
                    "priority": priority, "n_outgoing": n_outgoing,
                    "signalised": signalised, "speed": slot_speeds[slot],
                    "density": density_by_slot[slot], "occupancy": occupancy_by_slot[slot],
                    "speed_ratio": slot_speeds[slot] / max(speed_limit, 0.1),
                    "road_class": road_class,
                })
            edge_meta.append(((from_node, to_node, edge_key), length, slot_speeds, speed_source))

        feature_frame = pd.DataFrame(feature_rows)[self._feature_names]
        for column in self._categorical_features:
            feature_frame[column] = feature_frame[column].astype("category")
        co2_predictions = np.maximum(self._model.predict(feature_frame), 0.05)

        self.time_s: dict = {}
        self.co2_g: dict = {}
        self.speed_source: dict = {}
        for index, (graph_edge, length, slot_speeds, speed_source) in enumerate(edge_meta):
            self.time_s[graph_edge] = length / slot_speeds
            self.co2_g[graph_edge] = co2_predictions[index * N_TIME_SLOTS:(index + 1) * N_TIME_SLOTS]
            self.speed_source[graph_edge] = speed_source

    def _slot_index(self, offset_s: float) -> int:
        absolute_s = self.departure_s + offset_s
        return int(min(max(absolute_s // INTERVAL_S, 0), N_TIME_SLOTS - 1))

    def travel_time_s(self, graph_edge, offset_s: float) -> float:
        return float(self.time_s[graph_edge][self._slot_index(offset_s)])

    def co2_grams(self, graph_edge, offset_s: float) -> float:
        return float(self.co2_g[graph_edge][self._slot_index(offset_s)])

    def route_totals(self, node_path: list) -> tuple[float, float, float]:
        total_time_s = total_co2_g = total_length_m = 0.0
        for from_node, to_node in zip(node_path, node_path[1:]):
            edge_key = min(self.road_graph[from_node][to_node],
                           key=lambda candidate_key:
                           self.road_graph[from_node][to_node][candidate_key].get("length", 0))
            edge_time_s = self.travel_time_s((from_node, to_node, edge_key), total_time_s)
            total_co2_g += self.co2_grams((from_node, to_node, edge_key), total_time_s)
            total_length_m += float(self.road_graph[from_node][to_node][edge_key].get("length", 0))
            total_time_s += edge_time_s
        return total_time_s, total_co2_g, total_length_m
