from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config


def fetch(force: bool = False) -> Path:
    if config.RAW_OSM.is_file() and not force:
        return config.RAW_OSM

    west, south, east, north = config.BBOX
    osm_get_tool = config.sumo_tool("osmGet.py")
    command = [
        sys.executable, str(osm_get_tool),
        f"--bbox={west},{south},{east},{north}",
        "--prefix", config.CORRIDOR_NAME,
        "--output-dir", str(config.OSM_DIR),
    ]
    subprocess.run(command, check=True)

    downloaded_files = sorted(config.OSM_DIR.glob(f"{config.CORRIDOR_NAME}*.osm.xml"))
    if not downloaded_files:
        raise RuntimeError(f"osmGet.py produced no .osm.xml in {config.OSM_DIR}")
    if downloaded_files[0] != config.RAW_OSM:
        shutil.move(str(downloaded_files[0]), str(config.RAW_OSM))

    size_mb = config.RAW_OSM.stat().st_size / 1e6
    if size_mb < 1.0:
        print("[warn] extract looks small -- check the bbox is not empty ocean")
    return config.RAW_OSM


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    fetch(parser.parse_args().force)
