# Trajectory Planning with MILP

This project implements a trajectory planning system using Mixed Integer Linear Programming (MILP) to handle dynamic feasibility constraints. The system is designed to plan trajectories in environments with obstacles, ensuring that the planned paths are both efficient and safe.

## Project Structure

```
trajectory-planning
├── README.md
├── requirements.txt
├── examples/
│   └── simple_planning.py
├── maps/
│   └── scenarios/
│       ├── basic_map.json
│       ├── paper_validation.json
│       └── sensitivity analysis/
│           ├── complex_map.json
│           ├── easy_map_sa.json
│           └── hard_map_sa.json
├── src/
│   ├── main.py
│   ├── plot_sensitivity.py
│   ├── sensitivity.py
│   ├── figures/
│   ├── planners/
│   │   ├── __init__.py
│   │   ├── base_planner.py
│   │   ├── planner_chapter2.py
│   │   └── planner_chapter3.py
│   ├── Sensitivity results/
│   └── utils/
│       ├── __init__.py
│       ├── geometry.py
│       ├── map_loader.py
│       ├── obstacles.py
│       └── visualization.py
└── tests/
    ├── __init__.py
    ├── conftest.py
    └── test_planner_chapter2.py
```

## Installation

To set up the project, clone the repository and install the required dependencies:

```bash
git clone <repository-url>
cd trajectory-planning
pip install -r requirements.txt
```

## Usage

To run the trajectory planning application, execute the main script:

```bash
python src/main.py
```

Collecting workspace information# Trajectory Planning with MILP

This project implements a trajectory planning system using Mixed Integer Linear Programming (MILP) to handle dynamic feasibility constraints. The system is designed to plan trajectories in environments with obstacles, ensuring that the planned paths are both efficient and safe.

## Project Structure

```
trajectory-planning
├── README.md
├── requirements.txt
├── examples/
│   └── simple_planning.py
├── maps/
│   └── scenarios/
│       ├── basic_map.json
│       ├── paper_validation.json
│       └── sensitivity analysis/
│           ├── complex_map.json
│           ├── easy_map_sa.json
│           └── hard_map_sa.json
├── src/
│   ├── main.py
│   ├── plot_sensitivity.py
│   ├── sensitivity.py
│   ├── figures/
│   ├── planners/
│   │   ├── __init__.py
│   │   ├── base_planner.py
│   │   ├── planner_chapter2.py
│   │   └── planner_chapter3.py
│   ├── Sensitivity results/
│   └── utils/
│       ├── __init__.py
│       ├── geometry.py
│       ├── map_loader.py
│       ├── obstacles.py
│       └── visualization.py
└── tests/
    ├── __init__.py
    ├── conftest.py
    └── test_planner_chapter2.py
```

## Installation

To set up the project, clone the repository and install the required dependencies:

```bash
git clone <repository-url>
cd trajectory-planning
pip install -r requirements.txt
```

## Usage

To run the trajectory planning application, execute the main script:

```bash
python src/main.py
```

The application will load a map configuration, run the receding horizon controller, and visualize the environment, including obstacles and the planned trajectory.

## Sensitivity Analysis

The project includes functionality for sensitivity analysis of various planning parameters:

```bash
python src/sensitivity.py
```

This will run experiments with different parameter values (N, Ne, tau, umax) and save the results. You can then visualize the results using:

```bash
python src/plot_sensitivity.py
```

## Maps

The `maps/scenarios/` directory contains several JSON map configurations:
- basic_map.json: A simple environment with minimal obstacles
- paper_validation.json: Complex environment for validation
- `sensitivity analysis/`: Maps specifically designed for sensitivity analysis

## Testing

Unit tests are provided to ensure the functionality of the planners and geometric utilities. To run the tests, use:

```bash
pytest trajectory-planning/tests/
```

## License

To be chosen
