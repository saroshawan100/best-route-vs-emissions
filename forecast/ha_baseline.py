from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

GBA_DATA_DIR = config.ROOT / "external" / "LargeST" / "data" / "gba"
GWNET_LOG = (config.ROOT / "external" / "LargeST" / "experiments" / "gwnet"
             / "GBA-doubletransition-1" / "record_s2023.log")
OUTPUT_CSV = config.RESULTS / "forecast_metrics.csv"

INPUT_STEPS, FORECAST_STEPS, MINUTES_PER_STEP = 12, 12, 15
REPORTED_HORIZONS_MIN = {1: 15, 2: 30, 4: 60}


def masked(error: np.ndarray, actual: np.ndarray, metric_fn) -> float:
    observed = actual > 0
    return float(metric_fn(error[observed], actual[observed]))


def main() -> int:
    archive = np.load(GBA_DATA_DIR / "2019" / "his.npz")
    data, mean, std = archive["data"], float(archive["mean"]), float(archive["std"])
    flow = (data[..., 0] * std) + mean
    idx_train = np.load(GBA_DATA_DIR / "2019" / "idx_train.npy")
    idx_test = np.load(GBA_DATA_DIR / "2019" / "idx_test.npy")

    series_start = pd.Timestamp("2019-01-01 00:00")
    times = series_start + pd.to_timedelta(np.arange(flow.shape[0]) * MINUTES_PER_STEP, unit="m")
    slot = (times.dayofweek.to_numpy() * 96
            + times.hour.to_numpy() * 4 + times.minute.to_numpy() // MINUTES_PER_STEP)

    train_end = idx_train[-1]
    average_by_slot = np.zeros((7 * 96, flow.shape[1]), dtype=np.float64)
    for slot_index in range(7 * 96):
        slot_rows = flow[: train_end + 1][slot[: train_end + 1] == slot_index]
        average_by_slot[slot_index] = np.nanmean(
            np.where(slot_rows > 0, slot_rows, np.nan), axis=0)
    average_by_slot = np.nan_to_num(average_by_slot, nan=float(mean))

    metric_rows = []
    for step, horizon_min in REPORTED_HORIZONS_MIN.items():
        target_index = idx_test + step
        target_index = target_index[target_index < flow.shape[0]]
        actual = flow[target_index]
        predicted = average_by_slot[slot[target_index]]
        error = predicted - actual
        mae = masked(error, actual, lambda e, y: np.abs(e).mean())
        rmse = masked(error, actual, lambda e, y: np.sqrt((e ** 2).mean()))
        mape = masked(error, actual, lambda e, y: (np.abs(e) / y).mean() * 100)
        metric_rows.append({"model": "historical_average", "horizon_min": horizon_min,
                            "MAE": round(mae, 2), "RMSE": round(rmse, 2),
                            "MAPE%": round(mape, 2)})

    if GWNET_LOG.is_file():
        pattern = re.compile(r"Horizon (\d+), Test MAE: ([\d.]+), Test RMSE: "
                             r"([\d.]+), Test MAPE: ([\d.]+)")
        gwnet_metrics = {}
        for match in pattern.finditer(GWNET_LOG.read_text(encoding="utf-8", errors="ignore")):
            gwnet_metrics[int(match.group(1))] = (float(match.group(2)), float(match.group(3)),
                                                 float(match.group(4)))
        for step, horizon_min in REPORTED_HORIZONS_MIN.items():
            if step in gwnet_metrics:
                mae, rmse, mape = gwnet_metrics[step]
                metric_rows.append({"model": "gwnet", "horizon_min": horizon_min,
                                    "MAE": mae, "RMSE": rmse,
                                    "MAPE%": round(mape * 100, 2)})
        if not gwnet_metrics:
            print("[GW]  no test metrics in the gwnet log yet (still training?)")

    pd.DataFrame(metric_rows).to_csv(OUTPUT_CSV, index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
