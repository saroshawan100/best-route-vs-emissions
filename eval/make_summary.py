from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

sys.stdout.reconfigure(encoding="utf-8")

ROUTES_JSON = config.RESULTS / "routes.json"
METRICS_CSV = config.RESULTS / "sumo_metrics.csv"
MODEL_REPORT_TXT = config.RESULTS / "emissions_model_report.txt"
SUMMARY_MD = config.RESULTS / "summary.md"


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    routes = pd.DataFrame(json.loads(ROUTES_JSON.read_text(encoding="utf-8"))["routes"])
    measured = pd.read_csv(METRICS_CSV)
    baseline_scale_rows = measured[measured["scale"] == 1.0]
    per_route = baseline_scale_rows.groupby("route_id").agg(
        time_s=("duration_s", "mean"), time_sd=("duration_s", "std"),
        co2_g=("co2_g", "mean"), co2_sd=("co2_g", "std"),
        nox_g=("nox_g", "mean"), stops=("stops", "mean"),
        idle_s=("waiting_s", "mean"), meas_length_m=("route_length_m", "mean"),
        n=("seed", "count")).reset_index()
    joined = routes.merge(per_route, on="route_id", how="inner")
    missing = set(routes["route_id"]) - set(per_route["route_id"])
    if missing:
        print(f"[!] {len(missing)} routes never measured in SUMO: "
              f"{sorted(missing)[:4]}...")
    return joined, measured


def per_alpha(joined: pd.DataFrame) -> pd.DataFrame:
    sweep = joined[joined["kind"] == "sweep"]
    by_alpha = sweep.groupby("alpha").agg(
        time_s=("time_s", "mean"), co2_g=("co2_g", "mean"),
        time_sem=("time_s", "sem"), co2_sem=("co2_g", "sem"),
        stops=("stops", "mean"), idle_s=("idle_s", "mean"),
        n_od=("od", "nunique")).reset_index()
    return by_alpha.sort_values("alpha", ascending=False)


def knee_point(alpha_points: pd.DataFrame) -> pd.Series:
    times = alpha_points["time_s"].to_numpy()
    co2 = alpha_points["co2_g"].to_numpy()
    time_norm = (times - times.min()) / max(times.max() - times.min(), 1e-9)
    co2_norm = (co2 - co2.min()) / max(co2.max() - co2.min(), 1e-9)
    return alpha_points.iloc[int(np.argmin(np.hypot(time_norm, co2_norm)))]


def main() -> int:
    for required_path in (ROUTES_JSON, METRICS_CSV):
        if not required_path.is_file():
            raise SystemExit(f"missing {required_path.name} -- run the earlier stages first")
    joined, measured = load()
    alpha_points = per_alpha(joined)
    if len(alpha_points) < 2:
        raise SystemExit("fewer than 2 alpha points measured -- nothing to plot")

    knee = knee_point(alpha_points)
    fastest = alpha_points[alpha_points["alpha"] == 1.0].iloc[0]
    cleanest = alpha_points[alpha_points["alpha"] == 0.0].iloc[0]

    knee_time_pct = (knee["time_s"] - fastest["time_s"]) / fastest["time_s"] * 100
    knee_co2_pct = (fastest["co2_g"] - knee["co2_g"]) / fastest["co2_g"] * 100
    cleanest_time_pct = (cleanest["time_s"] - fastest["time_s"]) / fastest["time_s"] * 100
    cleanest_co2_pct = (fastest["co2_g"] - cleanest["co2_g"]) / fastest["co2_g"] * 100

    sweep = joined[joined["kind"] == "sweep"]
    corr_stops = sweep["co2_g"].corr(sweep["stops"])
    corr_idle = sweep["co2_g"].corr(sweep["idle_s"])

    grams_per_km = None
    if MODEL_REPORT_TXT.is_file():
        for line in MODEL_REPORT_TXT.read_text(encoding="utf-8").splitlines():
            if line.startswith("baseline constant:"):
                grams_per_km = float(line.split(":")[1].split("g")[0])

    lines = [
        "# Results summary — Best Route vs. Best Emissions",
        "",
        f"AM peak (07:00–09:00), depart 07:30, {config.CORRIDOR_LABEL}, "
        f"{sweep['od'].nunique()} O-D pairs, "
        f"{int(measured[measured['scale'] == 1.0]['seed'].nunique())} seeds "
        f"per route, HBEFA3/PC_G_EU4.",
        "",
        "## Headline",
        "",
        f"* **Fastest route (α=1):** {fastest['time_s']/60:.1f} min, "
        f"{fastest['co2_g']:.0f} g CO₂ (mean across O-D pairs)",
        f"* **Cleanest route (α=0):** {cleanest['time_s']/60:.1f} min "
        f"(+{cleanest_time_pct:.1f}%), {cleanest['co2_g']:.0f} g CO₂ (−{cleanest_co2_pct:.1f}%)",
        (f"* **Knee (α={knee['alpha']:g}):** +{knee_time_pct:.1f}% travel time buys "
         f"−{knee_co2_pct:.1f}% CO₂ — **{knee_co2_pct / knee_time_pct:.2f}% emissions "
         f"saved per 1% extra travel time**") if knee_time_pct > 0.1 else
        (f"* **Knee (α={knee['alpha']:g}):** dominates the pure-time route — "
         f"{-knee_time_pct:.1f}% *faster* and {knee_co2_pct:.1f}% cleaner as measured. "
         "The predicted-fastest route underperformed in simulation on at "
         "least one O-D pair (see below), so slightly emissions-weighted "
         "routing won on both axes."),
        "",
        "## Mechanism check",
        "",
        f"* CO₂ vs number of stops: r = {corr_stops:.2f}",
        f"* CO₂ vs idle time:       r = {corr_idle:.2f}",
        "  (stop-and-go, not distance, is where the grams come from — the "
        "project's premise)",
    ]

    slow_ods = []
    for od_label, group in sweep.groupby("od"):
        alpha1_time_s = group[group["alpha"] == 1.0]["time_s"].mean()
        best_time_s = group["time_s"].min()
        if alpha1_time_s > best_time_s * 1.05:
            slow_ods.append(f"{od_label} (+{(alpha1_time_s / best_time_s - 1) * 100:.0f}%)")
    if slow_ods:
        lines += ["",
                  "* Predicted-fastest (α=1) measured >5% slower than a "
                  "cleaner sweep route on: " + ", ".join(slow_ods) +
                  " — forecast-driven route choice is not infallible under "
                  "microsimulation, which is itself a result."]
    if grams_per_km:
        estimated_co2 = sweep["meas_length_m"] / 1000.0 * grams_per_km
        error_pct = (estimated_co2 - sweep["co2_g"]).abs() / sweep["co2_g"] * 100
        lines += ["",
                  f"* Constant-g/km estimate ({grams_per_km:.0f} g/km) is off by a "
                  f"median {error_pct.median():.0f}% per route — why the "
                  "fine-grained model is worth it."]

    scales = sorted(measured["scale"].unique())
    if len(scales) > 1:
        lines += ["", "## Demand sensitivity", ""]
        for scale in scales:
            scale_rows = measured[measured["scale"] == scale].merge(
                joined[joined["kind"] == "sweep"][["route_id", "alpha"]].drop_duplicates(),
                on="route_id")
            fastest_rows = scale_rows[scale_rows["alpha"] == 1.0]
            cleanest_rows = scale_rows[scale_rows["alpha"] == 0.0]
            if fastest_rows.empty or cleanest_rows.empty:
                continue
            saving_pct = ((fastest_rows["co2_g"].mean() - cleanest_rows["co2_g"].mean())
                          / fastest_rows["co2_g"].mean() * 100)
            lines.append(f"* scale {scale:g}: cleanest saves {saving_pct:.1f}% CO₂ vs fastest "
                         f"({'sign holds' if saving_pct > 0 else 'SIGN FLIPS'})")

    lines += ["", "## Limitations (state these)", "",
              "* Emissions are model-estimated (HBEFA3), not measured; claims "
              "are relative, not absolute grams.",
              "* Forecasts cover freeway edges only; arterial speeds come from "
              "the calibrated simulation (LargeST has zero arterial sensors).",
              f"* Signals are fixed-time; arterial demand uses the Caltrans "
              f"{config.AADT_NOTE} with an 8% peak-hour share.",
              "* Forecast horizons are 15-min steps (LargeST convention): "
              "steps 1/2/4 = 15/30/60 min."]

    SUMMARY_MD.write_text("\n".join(lines), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
