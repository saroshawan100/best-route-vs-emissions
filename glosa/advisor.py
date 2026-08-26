from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

ROUTES_JSON = config.RESULTS / "routes.json"
OUTPUT_CSV = config.RESULTS / "glosa_comparison.csv"
GLOSA_DIR = config.GLOSA_DIR

EMISSION_CLASS = "HBEFA3/PC_G_EU4"
EGO_ID = "ego_glosa"
MIN_ADVISORY_SPEED_MS = 3.0
MIN_ADVISORY_DISTANCE_M = 25.0
MAX_LOOKAHEAD_M = 600.0
DEPART_TIME_S = 1800


def green_window(tls_id, link_index, now: float, traci) -> tuple[float, float] | None:
    program = traci.trafficlight.getAllProgramLogics(tls_id)[0]
    phases = program.phases
    current_phase = traci.trafficlight.getPhase(tls_id)
    remaining = traci.trafficlight.getNextSwitch(tls_id) - now

    offset_s = 0.0
    for step in range(len(phases) * 2 + 1):
        phase_index = (current_phase + step) % len(phases)
        duration = remaining if step == 0 else phases[phase_index].duration
        state = phases[phase_index].state
        if link_index < len(state) and state[link_index] in "gG":
            green_start = offset_s
            green_end = offset_s + duration
            lookahead = 1
            while True:
                next_phase = phases[(phase_index + lookahead) % len(phases)]
                if link_index < len(next_phase.state) and next_phase.state[link_index] in "gG":
                    green_end += next_phase.duration
                    lookahead += 1
                else:
                    break
            return green_start, green_end
        offset_s += duration
    return None


def advise(traci) -> None:
    upcoming = traci.vehicle.getNextTLS(EGO_ID)
    if not upcoming:
        traci.vehicle.setSpeed(EGO_ID, -1)
        return
    tls_id, link_index, distance, state = upcoming[0]
    if distance < MIN_ADVISORY_DISTANCE_M or distance > MAX_LOOKAHEAD_M:
        traci.vehicle.setSpeed(EGO_ID, -1)
        return

    now = traci.simulation.getTime()
    window = green_window(tls_id, link_index, now, traci)
    if window is None:
        traci.vehicle.setSpeed(EGO_ID, -1)
        return
    green_start, green_end = window
    allowed_speed = traci.vehicle.getAllowedSpeed(EGO_ID)

    if green_start <= 0.1:
        if distance / max(allowed_speed, 0.1) <= green_end:
            traci.vehicle.setSpeed(EGO_ID, -1)
            return
        next_window = green_window(tls_id, link_index, now + green_end + 0.1, traci)
        if next_window is None:
            traci.vehicle.setSpeed(EGO_ID, -1)
            return
        green_start, green_end = (next_window[0] + green_end + 0.1,
                                  next_window[1] + green_end + 0.1)

    speed_to_arrive_at_start = distance / max(green_start, 0.1)
    speed_to_arrive_at_end = distance / max(green_end - 1.0, 0.2)
    min_speed, max_speed = (max(speed_to_arrive_at_end, MIN_ADVISORY_SPEED_MS),
                            min(speed_to_arrive_at_start, allowed_speed))
    if min_speed > max_speed:
        traci.vehicle.setSpeed(EGO_ID, -1)
        return
    current_speed = traci.vehicle.getSpeed(EGO_ID)
    traci.vehicle.setSpeed(EGO_ID, min(max(current_speed, min_speed), max_speed))


def run_once(route_edges: list[str], seed: int, advisory: bool) -> dict | None:
    import traci
    config.sumo_home()
    run_tag = f"s{seed}_{'on' if advisory else 'off'}"
    route_file = GLOSA_DIR / f"ego_{run_tag}.rou.xml"
    route_file.write_text(
        '<routes>\n'
        f'  <vType id="glosa_car" vClass="passenger" emissionClass="{EMISSION_CLASS}" '
        'scale="1"/>\n'
        f'  <vehicle id="{EGO_ID}" type="glosa_car" depart="{DEPART_TIME_S}">\n'
        f'    <route edges="{" ".join(route_edges)}"/>\n'
        '  </vehicle>\n'
        '</routes>\n', encoding="utf-8")
    tripinfo_path = GLOSA_DIR / f"tripinfo_{run_tag}.xml"

    traci.start([config.sumo_bin("sumo"), "-n", str(config.NET_XML),
                 "-r", f"{route_file},{config.DEMAND_ROU}",
                 "--tripinfo-output", str(tripinfo_path),
                 "--device.emissions.explicit", EGO_ID,
                 "--seed", str(seed), "--no-step-log", "true",
                 "--time-to-teleport", "300", "--end", "12000"])
    crossed: dict[str, str] = {}
    pending: tuple[str, str] | None = None
    departed = False
    try:
        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()
            if traci.simulation.getTime() >= 12000:
                break
            if EGO_ID not in traci.vehicle.getIDList():
                if pending and pending[0] not in crossed:
                    crossed[pending[0]] = pending[1]
                pending = None
                if departed:
                    break
                continue
            departed = True
            upcoming = traci.vehicle.getNextTLS(EGO_ID)
            next_signal = upcoming[0] if upcoming else None
            if pending and (next_signal is None or next_signal[0] != pending[0]):
                crossed[pending[0]] = pending[1]
            pending = (next_signal[0], next_signal[3]) if next_signal else None
            if advisory:
                advise(traci)
    finally:
        traci.close()

    tripinfo_root = ET.parse(tripinfo_path).getroot()
    tripinfo = next((entry for entry in tripinfo_root.findall("tripinfo")
                     if entry.get("id") == EGO_ID), None)
    if tripinfo is None:
        print(f"  [!] {run_tag}: ego never finished")
        return None
    emissions = tripinfo.find("emissions")
    n_signals_crossed = len(crossed)
    n_green_crossings = sum(1 for signal_state in crossed.values() if signal_state in "gG")
    return {
        "seed": seed, "advisory": advisory,
        "duration_s": float(tripinfo.get("duration")),
        "stops": int(tripinfo.get("waitingCount")),
        "idle_s": float(tripinfo.get("waitingTime")),
        "timeloss_s": float(tripinfo.get("timeLoss")),
        "co2_g": float(emissions.get("CO2_abs", 0)) / 1000.0,
        "fuel_g": float(emissions.get("fuel_abs", 0)) / 1000.0,
        "tls_crossed": n_signals_crossed,
        "green_hit_rate": n_green_crossings / n_signals_crossed if n_signals_crossed else float("nan"),
    }


def pick_route(records: list[dict], route_id: str | None) -> dict:
    if route_id:
        for record in records:
            if record["route_id"] == route_id:
                return record
        raise SystemExit(f"route_id {route_id} not in routes.json")
    cleanest_routes = [record for record in records
                       if record["kind"] == "sweep" and record["alpha"] == 0.0]
    return max(cleanest_routes, key=lambda record: record["length_m"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--route-id", default=None)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--seed0", type=int, default=300)
    args = parser.parse_args()

    if not ROUTES_JSON.is_file():
        raise SystemExit("No routes.json -- run src/routing/sweep.py first")
    GLOSA_DIR.mkdir(parents=True, exist_ok=True)

    import osmnx as ox
    import sumolib
    from sim.translate import Translator, build_way_to_edges

    payload = json.loads(ROUTES_JSON.read_text(encoding="utf-8"))
    record = pick_route(payload["routes"], args.route_id)

    sumo_net = sumolib.net.readNet(str(config.NET_XML), withInternal=True)
    road_graph = ox.load_graphml(config.GRAPHML)
    translator = Translator(sumo_net, road_graph,
                            build_way_to_edges(sumo_net, config.NET_XML))
    edge_ids, diagnostics = translator.translate(record["node_path"])
    if len(edge_ids) < 2:
        raise SystemExit("route translation failed")

    rows = []
    for index in range(args.seeds):
        seed = args.seed0 + index
        for advisory in (False, True):
            row = run_once(edge_ids, seed, advisory)
            if row:
                rows.append(row)

    comparison = pd.DataFrame(rows)
    comparison.to_csv(OUTPUT_CSV, index=False)

    return 0


if __name__ == "__main__":
    sys.exit(main())
