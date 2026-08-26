from __future__ import annotations
from pathlib import Path

import sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

TRAINING_DATA_CSV = config.RESULTS / "emissions_training_data.csv"
MODEL_JOBLIB = config.RESULTS / "emissions_model.joblib"
REPORT_TXT = config.RESULTS / "emissions_model_report.txt"

NUMERIC_FEATURES = ["length", "lanes", "speed_limit", "priority", "n_outgoing",
                    "signalised", "speed", "density", "occupancy", "speed_ratio"]
CATEGORICAL_FEATURES = ["road_class"]
TARGET_COLUMN = "co2_per_veh"


def metrics(actual, predicted) -> dict[str, float]:
    error = predicted - actual
    residual_sum_squares = float((error ** 2).sum())
    total_sum_squares = float(((actual - actual.mean()) ** 2).sum())
    return {
        "MAE": float(np.abs(error).mean()),
        "RMSE": float(np.sqrt((error ** 2).mean())),
        "R2": 1.0 - residual_sum_squares / total_sum_squares if total_sum_squares > 0 else float("nan"),
        "MAPE%": float((np.abs(error) / np.maximum(np.abs(actual), 1e-6)).mean() * 100),
    }


def main() -> int:
    if not TRAINING_DATA_CSV.is_file():
        raise SystemExit("Run src/emissions_model/collect_edge_data.py first")
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.inspection import permutation_importance
    import joblib

    samples = pd.read_csv(TRAINING_DATA_CSV)
    samples = samples.dropna(subset=NUMERIC_FEATURES + [TARGET_COLUMN])

    rng = np.random.default_rng(0)
    edge_ids = samples["edge"].unique()
    rng.shuffle(edge_ids)
    n_test_edges = max(1, int(0.25 * len(edge_ids)))
    test_edge_ids = set(edge_ids[:n_test_edges])
    test_rows = samples[samples["edge"].isin(test_edge_ids)]
    train_rows = samples[~samples["edge"].isin(test_edge_ids)]

    feature_names = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    train_features = train_rows[feature_names].copy()
    test_features = test_rows[feature_names].copy()
    for column in CATEGORICAL_FEATURES:
        train_features[column] = train_features[column].astype("category")
        test_features[column] = pd.Categorical(test_features[column],
                                               categories=train_features[column].cat.categories)
    train_target = train_rows[TARGET_COLUMN].to_numpy()
    test_target = test_rows[TARGET_COLUMN].to_numpy()

    model = HistGradientBoostingRegressor(
        max_iter=400, learning_rate=0.06, max_depth=None,
        min_samples_leaf=25, l2_regularization=1.0,
        categorical_features=[feature_names.index(column) for column in CATEGORICAL_FEATURES],
        early_stopping=True, validation_fraction=0.15, random_state=0)
    model.fit(train_features, train_target)
    predictions = model.predict(test_features)
    model_metrics = metrics(test_target, predictions)
    grams_per_km = float((train_rows[TARGET_COLUMN] / (train_rows["length"] / 1000.0)).median())
    proxy_predictions = grams_per_km * (test_rows["length"].to_numpy() / 1000.0)
    proxy_metrics = metrics(test_target, proxy_predictions)

    lines = [
        "Emissions model -- edge features -> CO2 grams per vehicle",
        f"samples {len(samples):,} | edges {samples['edge'].nunique():,} | "
        f"held-out edges {len(test_edge_ids):,} (disjoint from training)",
        "",
        f"{'model':<28}{'MAE':>9}{'RMSE':>9}{'R2':>8}{'MAPE%':>9}",
        f"{'constant g/km proxy':<28}{proxy_metrics['MAE']:>9.2f}{proxy_metrics['RMSE']:>9.2f}"
        f"{proxy_metrics['R2']:>8.3f}{proxy_metrics['MAPE%']:>9.1f}",
        f"{'learned model':<28}{model_metrics['MAE']:>9.2f}{model_metrics['RMSE']:>9.2f}"
        f"{model_metrics['R2']:>8.3f}{model_metrics['MAPE%']:>9.1f}",
        "",
        f"MAE improvement over proxy: {(1 - model_metrics['MAE']/proxy_metrics['MAE'])*100:.1f}%",
        f"baseline constant: {grams_per_km:.1f} g CO2 per vehicle-km",
        "",
    ]

    importance = permutation_importance(model, test_features, test_target, n_repeats=5,
                                        random_state=0,
                                        scoring="neg_mean_absolute_error")
    ranked = np.argsort(importance.importances_mean)[::-1]
    lines.append("permutation importance (MAE degradation when shuffled):")
    for index in ranked:
        lines.append(f"  {feature_names[index]:<16}{importance.importances_mean[index]:>8.3f}"
                     f" +/- {importance.importances_std[index]:.3f}")

    report_text = "\n".join(lines)
    REPORT_TXT.write_text(report_text, encoding="utf-8")
    joblib.dump({"model": model, "features": feature_names,
                 "categorical": CATEGORICAL_FEATURES, "target": TARGET_COLUMN}, MODEL_JOBLIB)

    if model_metrics["MAE"] >= proxy_metrics["MAE"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
