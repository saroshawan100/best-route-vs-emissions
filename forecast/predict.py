from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

LARGEST_REPO = config.ROOT / "external" / "LargeST"
GBA_DATA_DIR = LARGEST_REPO / "data" / "gba"
CHECKPOINT = LARGEST_REPO / "experiments" / "gwnet" / "GBA-doubletransition-1" / "final_model_s2023.pt"
OUTPUT_CSV = config.RESULTS / "forecast_speeds.csv"

INPUT_STEPS, FORECAST_STEPS = 12, 12
MINUTES_PER_STEP = 15
FREE_FLOW_KMH = 104.6
CAPACITY_VEH_PER_HOUR_PER_LANE = 2200.0

DEPARTURE_SCENARIOS = {
    "am_peak": "07:30", "midday": "12:00", "pm_peak": "17:30",
    "evening": "20:00", "off_peak": "05:00",
}


def flow_to_speed(flow_15min: np.ndarray, lanes: np.ndarray) -> np.ndarray:
    flow_per_hour_per_lane = np.maximum(flow_15min, 0.0) * 4.0 / np.maximum(lanes, 1)
    return FREE_FLOW_KMH / (1.0 + 0.15 * (flow_per_hour_per_lane
                                          / CAPACITY_VEH_PER_HOUR_PER_LANE) ** 4)


def build_model(device):
    import torch
    sys.path.insert(0, str(LARGEST_REPO))
    from src.models.gwnet import GWNET
    from src.utils.graph_algo import normalize_adj_mx

    adjacency = np.load(GBA_DATA_DIR / "gba_rn_adj.npy")
    supports = [torch.tensor(matrix).to(device)
                for matrix in normalize_adj_mx(adjacency, "doubletransition")]
    model = GWNET(node_num=2352, input_dim=3, output_dim=1, supports=supports,
                  adp_adj=1, dropout=0.3, residual_channels=32,
                  dilation_channels=32, skip_channels=256, end_channels=512,
                  seq_len=INPUT_STEPS, horizon=FORECAST_STEPS)
    state_dict = torch.load(CHECKPOINT, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device).eval()
    return model


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="2019-11-05",
                        help="weekday inside the test split")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    if not CHECKPOINT.is_file():
        raise SystemExit(f"No checkpoint at {CHECKPOINT} -- train gwnet first")

    import torch
    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))

    archive = np.load(GBA_DATA_DIR / "2019" / "his.npz")
    data, mean, std = archive["data"], float(archive["mean"]), float(archive["std"])
    idx_test = np.load(GBA_DATA_DIR / "2019" / "idx_test.npy")
    series_start = pd.Timestamp("2019-01-01 00:00")

    meta = pd.read_csv(GBA_DATA_DIR / "gba_meta.csv")
    lanes = meta["Lanes"].to_numpy(dtype=float)

    model = build_model(device)
    date = pd.Timestamp(args.date)
    if date.dayofweek >= 5:
        raise SystemExit(f"{args.date} is a weekend -- pick a weekday")

    rows = []
    for scenario_name, clock_time in DEPARTURE_SCENARIOS.items():
        departure_ts = pd.Timestamp(f"{args.date} {clock_time}")
        step_index = int((departure_ts - series_start) / pd.Timedelta(minutes=MINUTES_PER_STEP))
        if step_index not in idx_test:
            raise SystemExit(f"{departure_ts} (index {step_index}) is not in the test split "
                             f"[{idx_test[0]}..{idx_test[-1]}] -- pick a later date")

        input_window = data[step_index - INPUT_STEPS + 1: step_index + 1][None, ...]
        with torch.no_grad():
            prediction = model(torch.tensor(input_window, dtype=torch.float32, device=device))
        flow = (prediction.squeeze().cpu().numpy() * std) + mean
        flow = np.maximum(flow, 0.0)
        actual_flow = (data[step_index + 1: step_index + FORECAST_STEPS + 1, :, 0] * std) + mean

        speed = flow_to_speed(flow, lanes[None, :])
        for horizon_step in range(FORECAST_STEPS):
            for sensor_index, sensor_id in enumerate(meta["ID"]):
                rows.append({
                    "scenario": scenario_name, "departure": str(departure_ts),
                    "minutes_ahead": (horizon_step + 1) * MINUTES_PER_STEP, "sensor": sensor_id,
                    "pred_flow_15min": round(float(flow[horizon_step, sensor_index]), 1),
                    "true_flow_15min": round(float(actual_flow[horizon_step, sensor_index]), 1),
                    "speed_kmh": round(float(speed[horizon_step, sensor_index]), 2),
                })

    forecast_rows = pd.DataFrame(rows)
    forecast_rows.to_csv(OUTPUT_CSV, index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
