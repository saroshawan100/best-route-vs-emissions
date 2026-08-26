from __future__ import annotations

import argparse
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

ROUTES_JSON = config.RESULTS / "routes.json"
METRICS_CSV = config.RESULTS / "sumo_metrics.csv"
EVAL_DIR = config.EVAL_DIR

EMISSION_CLASS = "HBEFA3/PC_G_EU4"


def translate_routes(records: list[dict]) -> dict[str, list[str]]:
    import osmnx as ox
    import sumolib
    from sim.translate import Translator, build_way_to_edges

    sumo_net = sumolib.net.readNet(str(config.NET_XML), withInternal=True)
    road_graph = ox.load_graphml(config.GRAPHML)
    translator = Translator(sumo_net, road_graph,
                            build_way_to_edges(sumo_net, config.NET_XML))

    routes_by_id: dict[str, list[str]] = {}
    for record in records:
        route_id = record["route_id"]
        if route_id in routes_by_id:
            continue
        sumo_route, diagnostics = translator.translate(record["node_path"])
        if len(sumo_route) < 2:
            print(f"[!!] {route_id} ({record['od']}): translation failed, skipped")
            continue
        if diagnostics["off_corridor_frac"] > 0.15:
            print(f"[!] {route_id}: off-corridor {diagnostics['off_corridor_frac']:.1%} "
                  "(> 15%), keeping but flagging")
        routes_by_id[route_id] = sumo_route
    return routes_by_id


def write_route_file(routes: dict[str, list[str]], depart_s: int) -> Path:
    lines = ["<routes>",
             f'  <vType id="ego_car" vClass="passenger" emissionClass="{EMISSION_CLASS}" '
             'scale="1"/>']
    for route_id, edge_ids in sorted(routes.items()):
        lines.append(f'  <vehicle id="ego_{route_id}" type="ego_car" '
                     f'depart="{depart_s}" departLane="best" departSpeed="max">')
        lines.append(f'    <route edges="{" ".join(edge_ids)}"/>')
        lines.append('  </vehicle>')
    lines.append("</routes>")
    path = EVAL_DIR / "egos.rou.xml"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def run_seed(ego_route_file: Path, routes: dict, seed: int, scale: float) -> list[dict]:
    tripinfo_path = EVAL_DIR / f"tripinfo_s{seed}_x{scale}.xml"
    ego_vehicle_ids = ",".join(f"ego_{route_id}" for route_id in sorted(routes))
    command = [config.sumo_bin("sumo"), "-n", str(config.NET_XML),
               "-r", f"{ego_route_file},{config.DEMAND_ROU}",
               "--scale", str(scale),
               "--tripinfo-output", str(tripinfo_path),
               "--device.emissions.explicit", ego_vehicle_ids,
               "--seed", str(seed), "--no-step-log", "true",
               "--time-to-teleport", "300", "--end", "12000"]
    result = subprocess.run(command, capture_output=True, text=True, cwd=str(EVAL_DIR))
    if result.returncode != 0:
        print(result.stderr[-2000:])
        return []

    rows = []
    for tripinfo in ET.parse(tripinfo_path).getroot().findall("tripinfo"):
        if not tripinfo.get("id", "").startswith("ego_"):
            continue
        emissions = tripinfo.find("emissions")
        if emissions is None:
            continue
        rows.append({
            "route_id": tripinfo.get("id")[4:], "seed": seed, "scale": scale,
            "duration_s": float(tripinfo.get("duration")),
            "route_length_m": float(tripinfo.get("routeLength")),
            "waiting_s": float(tripinfo.get("waitingTime")),
            "stops": int(tripinfo.get("waitingCount")),
            "timeloss_s": float(tripinfo.get("timeLoss")),
            "co2_g": float(emissions.get("CO2_abs", 0)) / 1000.0,
            "nox_g": float(emissions.get("NOx_abs", 0)) / 1000.0,
            "pmx_g": float(emissions.get("PMx_abs", 0)) / 1000.0,
            "fuel_g": float(emissions.get("fuel_abs", 0)) / 1000.0,
        })
    measured_ids = {row["route_id"] for row in rows}
    missing = set(routes) - measured_ids
    if missing:
        print(f"  [!] seed {seed}: {len(missing)} ego(s) never finished: "
              f"{sorted(missing)[:3]}...")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--seed0", type=int, default=100)
    parser.add_argument("--scale", type=float, default=1.0)
    args = parser.parse_args()

    if not ROUTES_JSON.is_file():
        raise SystemExit("No routes.json -- run src/routing/sweep.py first")
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    payload = json.loads(ROUTES_JSON.read_text(encoding="utf-8"))
    routes = translate_routes(payload["routes"])
    if not routes:
        return 1
    ego_route_file = write_route_file(routes, payload.get("departure_s", 1800))

    all_rows = []
    for index in range(args.seeds):
        seed = args.seed0 + index
        all_rows.extend(run_seed(ego_route_file, routes, seed, args.scale))

    if not all_rows:
        print("[fail] no ego completed in any seed")
        return 1

    measured = pd.DataFrame(all_rows)
    if METRICS_CSV.is_file():
        previous = pd.read_csv(METRICS_CSV)
        previous = previous[~((previous["scale"] == args.scale)
                              & (previous["seed"].between(args.seed0,
                                                          args.seed0 + args.seeds - 1)))]
        measured = pd.concat([previous, measured], ignore_index=True)
    measured.to_csv(METRICS_CSV, index=False)

    return 0


if __name__ == "__main__":
    sys.exit(main())
