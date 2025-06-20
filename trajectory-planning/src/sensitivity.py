# trajectory-planning/src/sensitivity.py

import itertools
import time
import json
from pathlib import Path
import numpy as np
import pandas as pd
from planners.planner_chapter2 import RecedingHorizonController
from utils.map_loader import MapLoader


def run_experiment(map_path, param_grid):
    # Load map data
    with open(map_path, 'r') as f:
        map_data = json.load(f)

    map_boundary = map_data['map_boundary']
    obstacles = map_data['obstacles']
    start_point = map_data['start_point']
    end_point = map_data['end_point']

    results = []

    for N, Ne, tau, umax in param_grid:
        print(f"Running: N={N}, Ne={Ne}, tau={tau}, umax={umax}")

        # Initialize controller
        controller = RecedingHorizonController(
            map_boundary=map_boundary,
            obstacles=obstacles,
            start_point=start_point,
            end_point=end_point,
            N=N,
            Ne=Ne,
            tau=tau,
            umax=umax,
            use_visualizer=False
        )

        try:
            start_time = time.time()
            controller.plan_and_execute()
            elapsed = time.time() - start_time

            # Collect metrics
            result = {
                "N": N,
                "Ne": Ne,
                "tau": tau,
                "umax": umax,
                "computation_time": elapsed,
                "arrival_time": controller.arrival_time,
                "penalty_start": controller.penalty_values[0],
                "penalty_end": controller.penalty_values[-1],
                "penalty_rate": (
                    (controller.penalty_values[0] - controller.penalty_values[-1])
                    / len(controller.penalty_values)
                    if len(controller.penalty_values) > 1 else 0
                ),
                "successful": True
            }

        except Exception as e:
            print(f"Error: {e}")
            result = {
                "N": N,
                "Ne": Ne,
                "tau": tau,
                "umax": umax,
                "computation_time": None,
                "arrival_time": None,
                "penalty_start": None,
                "penalty_end": None,
                "penalty_rate": None,
                "successful": False
            }

        results.append(result)

    return pd.DataFrame(results)


def main():
    # Parameter values to test
    N_vals = [12, 18, 30]
    Ne_vals = [1, 2, 3]
    tau_vals = [0.2, 0.5, 0.75]
    umax_vals = [0.5, 1.0, 2.0]

    # Create parameter grid
    param_grid = list(itertools.product(N_vals, Ne_vals, tau_vals, umax_vals))

    # Path to fixed benchmark map
    maps_dir = Path(__file__).parent.parent / 'maps'
    map_path = maps_dir / 'scenarios' / 'sensitivity analysis'/ 'baseline_map_sa.json'

    results_df = run_experiment(map_path, param_grid)

    # Save results
    out_path = Path(__file__).parent / 'sensitivity_results.csv'
    results_df.to_csv(out_path, index=False)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
