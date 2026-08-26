from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

LARGEST_REPO = config.ROOT / "external" / "LargeST"
CHECKPOINT = LARGEST_REPO / "experiments" / "gwnet" / "GBA-doubletransition-1" / "final_model_s2023.pt"
PYTHON = sys.executable


def run(command: list[str], cwd: Path = config.ROOT) -> None:
    result = subprocess.run([str(part) for part in command], cwd=str(cwd))
    if result.returncode != 0:
        raise SystemExit(f"step failed (exit {result.returncode}) -- fix and re-run; "
                         "completed steps will be skipped")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true",
                        help="re-run steps even when outputs exist")
    parser.add_argument("--epochs", type=int, default=100,
                        help="gwnet max epochs (patience 30 may stop earlier)")
    parser.add_argument("--skip-train", action="store_true",
                        help="fail fast instead of training if no checkpoint")
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--bs", type=int, default=32)
    args = parser.parse_args()

    def need(path: Path) -> bool:
        return args.force or not path.is_file()

    if need(config.RESULTS / "emissions_training_data.csv"):
        run([PYTHON, "src/emissions_model/collect_edge_data.py", "--hours", "7", "9"])
    if need(config.RESULTS / "emissions_model.joblib"):
        run([PYTHON, "src/emissions_model/train_emissions_model.py"])

    if need(CHECKPOINT):
        if args.skip_train:
            raise SystemExit(f"no checkpoint at {CHECKPOINT} and --skip-train given")
        run([PYTHON, "experiments/gwnet/main.py", "--device", "cuda:0",
             "--dataset", "GBA", "--years", "2019", "--model_name", "gwnet",
             "--seed", "2023", "--bs", str(args.bs),
             "--max_epochs", str(args.epochs)], cwd=LARGEST_REPO)

    if need(config.RESULTS / "forecast_speeds.csv"):
        run([PYTHON, "src/forecast/predict.py"])
    run([PYTHON, "src/forecast/ha_baseline.py"])

    if need(config.RESULTS / "routes.json"):
        run([PYTHON, "src/routing/sweep.py"])

    if need(config.RESULTS / "sumo_metrics.csv"):
        run([PYTHON, "src/eval/measure_routes.py", "--seeds", str(args.seeds)])
        run([PYTHON, "src/eval/measure_routes.py", "--seeds", "3",
             "--seed0", "200", "--scale", "0.8"])
        run([PYTHON, "src/eval/measure_routes.py", "--seeds", "3",
             "--seed0", "210", "--scale", "1.2"])

    run([PYTHON, "src/eval/make_summary.py"])

    return 0


if __name__ == "__main__":
    sys.exit(main())
