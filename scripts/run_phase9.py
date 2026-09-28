"""Run one or all Phase 9 experiments on REAL validated feature data only."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import get_settings
from app.experiments.registry import EXPERIMENTS
from app.experiments.runner import Phase9BlockedError, run_experiment, write_result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="all", choices=["all"] + [x.experiment_id for x in EXPERIMENTS])
    args = parser.parse_args()

    output_dir = Path("experiments/results/official")
    specs = EXPERIMENTS if args.experiment == "all" else [x for x in EXPERIMENTS if x.experiment_id == args.experiment]

    for spec in specs:
        try:
            result = run_experiment(spec, settings=get_settings())
        except Phase9BlockedError as exc:
            print("PHASE 9 BLOCKED — no real experiment was executed.")
            print(str(exc))
            return 2
        path = write_result(result, output_dir)
        print(f"{spec.experiment_id}: completed -> {path}")
        print(json.dumps(result["test"], indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
