# trajectory-planning/src/sensitivity.py

import time
import json
from pathlib import Path
import numpy as np
import pandas as pd
from planners.planner_chapter2 import RecedingHorizonController

# Select variable to test and fixed values
VARY = "tau"  # Choose from: "N", "Ne", "tau", "umax"

param_options = {
    "N": [12, 18, 24, 30, 36],
    "Ne": [1, 2, 3, 4, 5],
    "tau": [0.1, 0.2, 0.3, 0.5, 0.75, 1.0],
    "umax": [0.5, 1.0, 1.5, 2.0]
}

fixed_params = {
    "N": 30,
    "Ne": 3,
    "tau": 0.5,
    "umax": 1.0
}

def run_experiments(vary_key, vary_values, fixed_params, map_path):
    with open(map_path, 'r') as f:
        map_data = json.load(f)

    results = []
    for val in vary_values:
        params = fixed_params.copy()
        params[vary_key] = val
        print(f"Running: {params}")

        controller = RecedingHorizonController(
            map_boundary=map_data['map_boundary'],
            obstacles=map_data['obstacles'],
            start_point=map_data['start_point'],
            end_point=map_data['end_point'],
            N=params["N"],
            Ne=params["Ne"],
            tau=params["tau"],
            umax=params["umax"],
            use_visualizer=False
        )

        try:
            start_time = time.time()
            controller.plan_and_execute()
            distance_progress = controller.distance_history
            stuck = False
            if len(distance_progress) > 5:
                recent_deltas = np.diff(distance_progress[-5:])
                stuck = np.all(np.abs(recent_deltas) < 1e-2)  # Threshold can be adjusted

            elapsed = time.time() - start_time

            successful = controller.arrival_time is not None


            result = {
                "varied_param": vary_key,
                vary_key: val,
                "N": params["N"],
                "Ne": params["Ne"],
                "tau": params["tau"],
                "umax": params["umax"],
                "computation_time": elapsed,
                "arrival_time": controller.arrival_time,
                "penalty_start": controller.penalty_values[0] if controller.penalty_values else None,
                "penalty_end": controller.penalty_values[-1] if controller.penalty_values else None,
                "penalty_rate": (
                    (controller.penalty_values[0] - controller.penalty_values[-1])
                    / len(controller.penalty_values)
                    if len(controller.penalty_values) > 1 else 0
                ),
                "successful": controller.arrival_time is not None,
                "stuck_near_end": stuck
        }           

        except Exception as e:
            print(f"Error: {e}")
            result = {
                "varied_param": vary_key,
                vary_key: val,
                "N": params["N"],
                "Ne": params["Ne"],
                "tau": params["tau"],
                "umax": params["umax"],
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
    vary_key = VARY
    vary_values = param_options[vary_key]

    # Maps
    map_names = ["easy_map_sa", "baseline_map_sa", "hard_map_sa"]
    maps_dir = Path(__file__).parent.parent / 'maps' / 'scenarios'/ 'sensitivity analysis'

    all_results = []

    for map_name in map_names:
        map_path = maps_dir / f"{map_name}.json"
        df = run_experiments(vary_key, vary_values, fixed_params, map_path)
        df["map"] = map_name 
        all_results.append(df)

    # Save results
    combined_df = pd.concat(all_results, ignore_index=True)

    results_dir = Path(__file__).parent / "Sensitivity results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"sensitivity_{vary_key}_with_maps.csv"
    combined_df.to_csv(out_path, index=False)

    print(f"Saved combined results to {out_path}")

## Test for map with concavities (tau = 0.75 needed)
def run_single_test_on_new_map():
    map_name = "complex_map"
    maps_dir = Path(__file__).parent.parent / 'maps' / 'scenarios' / 'sensitivity analysis'
    map_path = maps_dir / f"{map_name}.json"

    with open(map_path, 'r') as f:
        map_data = json.load(f)

    print(f"Running single evaluation on {map_name}...")
    controller = RecedingHorizonController(
        map_boundary=map_data['map_boundary'],
        obstacles=map_data['obstacles'],
        start_point=map_data['start_point'],
        end_point=map_data['end_point'],
        N=fixed_params["N"],
        Ne=fixed_params["Ne"],
        tau= 0.75,
        umax=fixed_params["umax"],
        use_visualizer=True
    )

    try:
        start_time = time.time()
        controller.plan_and_execute()
        distance_progress = controller.distance_history
        stuck = False
        if len(distance_progress) > 5:
            recent_deltas = np.diff(distance_progress[-5:])
            stuck = np.all(np.abs(recent_deltas) < 1e-2)

        elapsed = time.time() - start_time

        result = {
            "map": map_name,
            "N": fixed_params["N"],
            "Ne": fixed_params["Ne"],
            "tau": 0.75,
            "umax": fixed_params["umax"],
            "computation_time": elapsed,
            "arrival_time": controller.arrival_time,
            "penalty_start": controller.penalty_values[0] if controller.penalty_values else None,
            "penalty_end": controller.penalty_values[-1] if controller.penalty_values else None,
            "penalty_rate": (
                (controller.penalty_values[0] - controller.penalty_values[-1]) / len(controller.penalty_values)
                if len(controller.penalty_values) > 1 else 0
            ),
            "successful": controller.arrival_time is not None,
            "stuck_near_end": stuck,
            "varied_param": "none",
            "notes": "concave map test"
        }

        # Append to a dedicated file
        results_dir = Path(__file__).parent / "Sensitivity results"
        results_dir.mkdir(parents=True, exist_ok=True)
        out_path = results_dir / "sensitivity_single_run.csv"

        if out_path.exists():
            existing = pd.read_csv(out_path)
            updated = pd.concat([existing, pd.DataFrame([result])], ignore_index=True)
        else:
            updated = pd.DataFrame([result])

        updated.to_csv(out_path, index=False)
        print(f"Saved result to {out_path}")

    except Exception as e:
        print(f"Error in single run: {e}")


if __name__ == "__main__":
    # Choose if you want to run the full sensitivity analysis or just a single test on a specific map
    mode = "single"  # or "full"

    if mode == "full":
        main()
    elif mode == "single":
        run_single_test_on_new_map()


