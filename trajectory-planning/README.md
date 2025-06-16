# Trajectory Planning with MILP

This project implements a trajectory planning system using Mixed Integer Linear Programming (MILP) to handle dynamic feasibility constraints. The system is designed to plan trajectories in environments with obstacles, ensuring that the planned paths are both efficient and safe.

## Project Structure

```
trajectory-planning
├── src
│   ├── planners
│   │   ├── __init__.py
│   │   ├── base_planner.py
│   │   └── milp_planner.py
│   ├── utils
│   │   ├── __init__.py
│   │   ├── geometry.py
│   │   ├── visualization.py
│   │   └── obstacles.py
│   └── main.py
├── tests
│   ├── __init__.py
│   ├── test_milp_planner.py
│   └── test_geometry.py
├── examples
│   └── simple_planning.py
├── requirements.txt
└── README.md
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

You will be prompted to choose the MILP planner for trajectory planning. The application will visualize the environment, including obstacles and the planned trajectory.

## Testing

Unit tests are provided to ensure the functionality of the MILP planner and geometric utilities. To run the tests, use:

```bash
pytest tests/
```

## Examples

An example of how to use the trajectory planning functionality can be found in the `examples/simple_planning.py` file. This script demonstrates a simple planning scenario with predefined obstacles and a start and end point.

## Contributing

Contributions are welcome! Please submit a pull request or open an issue for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.