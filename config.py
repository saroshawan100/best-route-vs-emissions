from __future__ import annotations

import os
import sys
from pathlib import Path
CORRIDORS = {
    "pa_mv": {
        "label": "Palo Alto to Mountain View",
        "bbox": (-122.160, 37.370, -122.060, 37.450),
        "places": {
            "pa_downtown": (37.4459, -122.1600),
            "pa_midtown": (37.4270, -122.1310),
            "mv_downtown": (37.3894, -122.0819),
            "mv_whisman": (37.4010, -122.0680),
        },
        "od_pairs": [
            ("pa_downtown", "mv_downtown"), ("mv_downtown", "pa_downtown"),
            ("pa_downtown", "mv_whisman"), ("mv_whisman", "pa_downtown"),
            ("pa_midtown", "mv_downtown"), ("mv_downtown", "pa_midtown"),
        ],
        "arterial_name": "El Camino Real",
        "arterial_route": "82",
        "arterial_aadt": 41_300,
        "aadt_note": "2019 Traffic Census corridor mean for SR-82 (AADT "
                     "41,300, Castro St – University Ave)",
    },
    "eb_i80": {
        "label": "Berkeley to El Cerrito",
        "bbox": (-122.335, 37.845, -122.260, 37.915),
        "places": {
            "berkeley_university": (37.8659, -122.2921),
            "berkeley_gilman": (37.8802, -122.2955),
            "albany_solano": (37.8891, -122.2976),
            "elcerrito_plaza": (37.9027, -122.2996),
        },
        "od_pairs": [
            ("berkeley_university", "elcerrito_plaza"),
            ("elcerrito_plaza", "berkeley_university"),
            ("berkeley_university", "albany_solano"),
            ("albany_solano", "berkeley_university"),
            ("berkeley_gilman", "elcerrito_plaza"),
            ("elcerrito_plaza", "berkeley_gilman"),
        ],
        "arterial_name": "San Pablo Avenue",
        "arterial_route": "123",
        "arterial_aadt": 23_100,
        "aadt_note": "2019 Traffic Census corridor mean for SR-123 (AADT "
                     "23,100, University Ave – El Cerrito Central Ave)",
    },
}

CORRIDOR_NAME = os.environ.get("CORRIDOR", "pa_mv")
if CORRIDOR_NAME not in CORRIDORS:
    raise SystemExit(f"CORRIDOR={CORRIDOR_NAME!r} unknown; "
                     f"choose from {sorted(CORRIDORS)}")
_selected_corridor = CORRIDORS[CORRIDOR_NAME]
CORRIDOR_LABEL = _selected_corridor["label"]
BBOX = _selected_corridor["bbox"]
PLACES = _selected_corridor["places"]
OD_PAIRS = _selected_corridor["od_pairs"]
ARTERIAL_NAME = _selected_corridor["arterial_name"]
ARTERIAL_ROUTE = _selected_corridor["arterial_route"]
ARTERIAL_AADT = _selected_corridor["arterial_aadt"]
AADT_NOTE = _selected_corridor["aadt_note"]

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OSM_DIR = DATA / "osm"
SUMO_DIR = DATA / "sumo"
LARGEST_DIR = DATA / "largest"
RESULTS = ROOT / "results" if CORRIDOR_NAME == "pa_mv" \
    else ROOT / "results" / CORRIDOR_NAME

for _required_directory in (OSM_DIR, SUMO_DIR, LARGEST_DIR, RESULTS):
    _required_directory.mkdir(parents=True, exist_ok=True)

RAW_OSM = OSM_DIR / f"{CORRIDOR_NAME}_bbox.osm.xml"
GRAPHML = OSM_DIR / f"{CORRIDOR_NAME}_drive.graphml"
NET_XML = SUMO_DIR / f"{CORRIDOR_NAME}.net.xml"
EDGE_MAP = SUMO_DIR / f"{CORRIDOR_NAME}_edgemap.json"

SUMO_PREFIX = "" if CORRIDOR_NAME == "pa_mv" else f"{CORRIDOR_NAME}_"
COUNTS_XML = SUMO_DIR / f"{SUMO_PREFIX}counts.xml"
CANDIDATES_ROU = SUMO_DIR / f"{SUMO_PREFIX}candidates.rou.xml"
DEMAND_ROU = SUMO_DIR / f"{SUMO_PREFIX}demand.rou.xml"
EVAL_DIR = SUMO_DIR / f"{SUMO_PREFIX}eval"
GLOSA_DIR = SUMO_DIR / f"{SUMO_PREFIX}glosa"

DRIVABLE_HIGHWAY_CLASSES = {
    "motorway", "motorway_link", "trunk", "trunk_link",
    "primary", "primary_link", "secondary", "secondary_link",
    "tertiary", "tertiary_link", "unclassified", "residential",
    "living_street",
}


def sumo_home() -> Path:
    if os.environ.get("SUMO_HOME"):
        return Path(os.environ["SUMO_HOME"])

    import sumo
    sumo_home_dir = Path(sumo.__file__).parent
    if not (sumo_home_dir / "tools").is_dir():
        for path_entry in sys.path:
            candidate_dir = Path(path_entry) / "sumo"
            if (candidate_dir / "tools").is_dir():
                sumo_home_dir = candidate_dir
                break
        else:
            raise RuntimeError(
                f"`import sumo` resolved to {sumo_home_dir}, which has no tools/ dir. "
                "A local module named 'sumo' is shadowing the eclipse-sumo "
                "package. Rename it, or set SUMO_HOME explicitly.")
    os.environ["SUMO_HOME"] = str(sumo_home_dir)
    return sumo_home_dir


def sumo_tool(tool_name: str) -> Path:
    tool_path = sumo_home() / "tools" / tool_name
    if not tool_path.is_file():
        raise FileNotFoundError(f"SUMO tool not found: {tool_path}")
    return tool_path


def sumo_bin(binary_name: str) -> str:
    executable = binary_name + (".exe" if sys.platform == "win32" else "")
    candidate_path = Path(sys.executable).parent / executable
    if candidate_path.is_file():
        return str(candidate_path)
    candidate_path = sumo_home() / "bin" / executable
    if candidate_path.is_file():
        return str(candidate_path)
    return binary_name
