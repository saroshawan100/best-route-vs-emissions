import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

sys.stdout.reconfigure(encoding="utf-8")

RESULTS_DIR = config.RESULTS
FIGURES_DIR = RESULTS_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)
CORRIDOR_LABEL = config.CORRIDOR_LABEL

COLOR_OURS = "#2a78d6"
COLOR_BASELINE = "#898781"
COLOR_SAVING = "#0ca30c"
COLOR_COST = "#d03b3b"
COLOR_INK = "#0b0b0b"
COLOR_INK_MUTED = "#52514e"
COLOR_GRID = "#e1e0d9"
COLOR_AXIS = "#c3c2b7"

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "font.size": 12,
    "font.family": "sans-serif",
    "text.color": COLOR_INK,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.labelcolor": COLOR_INK_MUTED,
    "axes.edgecolor": COLOR_AXIS,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": COLOR_GRID,
    "grid.linewidth": 0.8,
    "xtick.color": COLOR_INK_MUTED,
    "ytick.color": COLOR_INK_MUTED,
    "figure.facecolor": "white",
})


def save(figure, filename):
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / filename, bbox_inches="tight")
    plt.close(figure)


forecast_metrics = pd.read_csv(RESULTS_DIR / "forecast_metrics.csv")
glosa_comparison = pd.read_csv(RESULTS_DIR / "glosa_comparison.csv")
sumo_metrics = pd.read_csv(RESULTS_DIR / "sumo_metrics.csv")
routes = pd.DataFrame([
    {key: record.get(key) for key in ("route_id", "od", "kind", "alpha")}
    for record in json.load(open(RESULTS_DIR / "routes.json"))["routes"]
])
measured = sumo_metrics.merge(routes, on="route_id")

ha_mae = forecast_metrics[forecast_metrics.model == "historical_average"].MAE.mean()
gwnet_mae = forecast_metrics[forecast_metrics.model == "gwnet"].MAE.mean()

figure, axis = plt.subplots(figsize=(5.4, 4))
bars = axis.bar(["Historical\naverage", "Graph WaveNet\n(ours)"], [ha_mae, gwnet_mae],
                color=[COLOR_BASELINE, COLOR_OURS], width=0.55)
axis.bar_label(bars, fmt="%.0f", padding=4, fontweight="bold", fontsize=14)
axis.set_ylabel("Forecast error MAE at every 15 min")
axis.set_title(f"Our forecaster vs. Historical Average")
axis.set_ylim(0, ha_mae * 1.25)
axis.grid(axis="x", visible=False)
save(figure, "fig1_forecaster.png")

sweep_means = (measured[(measured.kind == "sweep") & (measured.scale == 1.0)]
               .groupby(["alpha", "od"])[["duration_s", "co2_g"]].mean()
               .groupby("alpha").mean().reset_index())
sweep_means["min"] = sweep_means.duration_s / 60
fastest = sweep_means[sweep_means.alpha == 1.0].iloc[0]
cleanest = sweep_means[sweep_means.alpha == 0.0].iloc[0]
score = (sweep_means["min"] / sweep_means["min"].min()
         + sweep_means.co2_g / sweep_means.co2_g.min())
knee = sweep_means.loc[score.idxmin()]
knee_is_cleanest = knee.alpha == 0.0
other_points = sweep_means[~sweep_means.alpha.isin(
    {1.0, knee.alpha} | (set() if knee_is_cleanest else {0.0}))]

figure, axis = plt.subplots(figsize=(6.8, 4.4))
axis.scatter(other_points["min"], other_points.co2_g, s=30, color=COLOR_BASELINE,
             alpha=0.55, zorder=2)
axis.scatter([fastest["min"]], [fastest.co2_g], s=70, color=COLOR_INK_MUTED, zorder=3)
axis.annotate("predicted fastest", (fastest["min"], fastest.co2_g),
              textcoords="offset points", xytext=(-8, 6), ha="right", color=COLOR_INK_MUTED)
if not knee_is_cleanest:
    axis.scatter([cleanest["min"]], [cleanest.co2_g], s=70, color=COLOR_INK_MUTED, zorder=3)
    axis.annotate("cleanest", (cleanest["min"], cleanest.co2_g),
                  textcoords="offset points", xytext=(0, -18), ha="center",
                  color=COLOR_INK_MUTED)
axis.scatter([knee["min"]], [knee.co2_g], s=110, color=COLOR_OURS, zorder=4)
knee_label = ("cleanest and fastest in practice" if knee_is_cleanest
              else "best balance:\nfaster AND cleaner")
axis.annotate(knee_label, (knee["min"], knee.co2_g),
              textcoords="offset points", xytext=(14, 8), ha="left",
              fontweight="bold", color=COLOR_OURS)
axis.set_xlabel("Minutes travel time per trip")
axis.set_ylabel("CO₂ (grams per trip)")
axis.set_title(f"Every route choice for {CORRIDOR_LABEL}")
axis.set_xlim(sweep_means["min"].min() - 0.5, sweep_means["min"].max() + 0.5)
save(figure, "fig2_tradeoff.png")

baseline_means = (measured[measured.scale == 1.0].assign(
    label=lambda frame: frame.apply(
        lambda row: {"shortest_dist": "Shortest\ndistance",
                     "fastest_static": "Static\nfastest"}.get(row["kind"])
        or ("Ours" if row["alpha"] == knee.alpha else None),
        axis=1))
    .dropna(subset=["label"])
    .groupby(["label", "od"])[["duration_s", "co2_g"]].mean()
    .groupby("label").mean()
    .loc[["Shortest\ndistance", "Static\nfastest", "Ours"]])
bar_colors = [COLOR_BASELINE, COLOR_BASELINE, COLOR_OURS]

figure, panels = plt.subplots(1, 2, figsize=(9, 4))
for panel, column, unit_label, panel_title in [
        (panels[0], "duration_s", "minutes", "Travel time"),
        (panels[1], "co2_g", "grams", "CO₂ per trip")]:
    values = baseline_means[column] / (60 if column == "duration_s" else 1)
    bars = panel.bar(range(3), values, color=bar_colors, width=0.55)
    panel.bar_label(bars, fmt="%.0f", padding=3, fontweight="bold", fontsize=13)
    panel.set_xticks(range(3), baseline_means.index)
    panel.set_ylabel(unit_label)
    panel.set_title(panel_title, fontsize=12, color=COLOR_INK_MUTED)
    panel.set_ylim(0, values.max() * 1.18)
    panel.grid(axis="x", visible=False)
figure.suptitle(f"Forecast driven routing beats both baselines {CORRIDOR_LABEL}",
                fontweight="bold", fontsize=14)
save(figure, "fig3_baselines.png")

sensitivity = (measured[measured.kind == "sweep"].groupby(["scale", "alpha"])["co2_g"].mean()
               .unstack("alpha"))
saving = (1 - sensitivity[0.0] / sensitivity[1.0]) * 100

figure, axis = plt.subplots(figsize=(6, 4))
bar_colors = [COLOR_COST if value < 0 else COLOR_SAVING for value in saving.values]
bars = axis.bar([f"{scale:g}× traffic" for scale in saving.index], saving.values,
                color=bar_colors, width=0.55)
axis.bar_label(bars, fmt="%+.0f%%", padding=4, fontweight="bold", fontsize=13)
axis.axhline(0, color=COLOR_INK, lw=0.8)
axis.set_ylabel("CO₂ saved by cleanest route %")
if (saving.values < 0).any():
    axis.set_title(f"The trade-off only exists in congestion — {CORRIDOR_LABEL}")
    worst_index = int(np.argmin(saving.values))
    axis.text(worst_index, saving.values.min() - 3.5, "fastest is already cleanest",
              ha="center", va="top", fontsize=10, color=COLOR_INK_MUTED)
    axis.set_ylim(saving.values.min() * 1.6, saving.values.max() * 1.3)
else:
    axis.set_title(f"Cleanest route wins at every traffic level {CORRIDOR_LABEL}")
    axis.set_ylim(0, saving.values.max() * 1.3)
axis.grid(axis="x", visible=False)
save(figure, "fig4_sensitivity.png")

glosa_means = glosa_comparison.groupby("advisory")[
    ["stops", "idle_s", "co2_g", "duration_s"]].mean()
advisory_off, advisory_on = glosa_means.loc[False], glosa_means.loc[True]
pct_change = ((advisory_on - advisory_off) / advisory_off * 100)[
    ["duration_s", "co2_g", "idle_s", "stops"]]
metric_labels = ["Travel time", "CO₂", "Idle time", "Stops"]
duration_pct_change = pct_change["duration_s"]

figure, axis = plt.subplots(figsize=(6.8, 3.8))
bar_colors = [COLOR_SAVING if value < 0 else COLOR_BASELINE for value in pct_change.values]
bars = axis.barh(metric_labels, pct_change.values, color=bar_colors, height=0.55)
axis.bar_label(bars, fmt="%+.0f%%", padding=4, fontweight="bold", fontsize=13)
axis.axvline(0, color=COLOR_INK, lw=0.8)
axis.set_xlabel("Change with green-light advisory (%)")
axis.set_title(f"GLOSA gives less stop and go {CORRIDOR_LABEL}")
axis.set_xlim(pct_change.values.min() * 1.25, 14)
axis.grid(axis="y", visible=False)
save(figure, "fig5_glosa.png")

sweep_rows = measured[(measured.kind == "sweep") & (measured.scale == 1.0)]
correlation = np.corrcoef(sweep_rows.waiting_s, sweep_rows.co2_g)[0, 1]

figure, axis = plt.subplots(figsize=(6, 4.2))
axis.scatter(sweep_rows.waiting_s / 60, sweep_rows.co2_g, s=16, alpha=0.45,
             color=COLOR_OURS, edgecolors="none")
fit = np.polyfit(sweep_rows.waiting_s / 60, sweep_rows.co2_g, 1)
x_line = np.linspace(sweep_rows.waiting_s.min() / 60, sweep_rows.waiting_s.max() / 60, 50)
axis.plot(x_line, np.polyval(fit, x_line), color=COLOR_INK_MUTED, lw=2)
axis.set_xlabel("Minutes stuck idling per trip")
axis.set_ylabel("CO₂ (grams per trip)")
axis.set_title(f"Emissions tracking for stop and go driving for {CORRIDOR_LABEL}")
save(figure, "fig6_mechanism.png")
