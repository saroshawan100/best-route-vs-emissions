from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

NETCONVERT_FLAGS = [
    "--output.original-names",
    "--roundabouts.guess",
    "--ramps.guess",
    "--junctions.join",
    "--tls.guess-signals",
    "--tls.discard-simple",
    "--tls.join",
    "--tls.default-type", "static",
    "--remove-edges.isolated",
    "--keep-edges.by-vclass", "passenger",
    "--no-turnarounds", "true",
]


def build(force: bool = False) -> Path:
    if not config.RAW_OSM.is_file():
        raise SystemExit("No extract found. Run: python src/graph/fetch_osm.py")
    if config.NET_XML.is_file() and not force:
        return config.NET_XML

    log_path = config.SUMO_DIR / f"{config.CORRIDOR_NAME}_netconvert.log"
    command = [config.sumo_bin("netconvert"), "--osm-files", str(config.RAW_OSM),
               "-o", str(config.NET_XML), "--log", str(log_path)] + NETCONVERT_FLAGS
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stdout[-3000:])
        print(result.stderr[-3000:])
        raise SystemExit(f"netconvert failed (exit {result.returncode}); see {log_path}")

    import sumolib
    sumo_net = sumolib.net.readNet(str(config.NET_XML))
    traffic_lights = sumo_net.getTrafficLights()
    if len(traffic_lights) == 0:
        print("[warn] ZERO traffic lights -- GLOSA and green-wave metrics are "
              "impossible. Check --tls.guess-signals and the extract.")
    return config.NET_XML


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    build(parser.parse_args().force)
