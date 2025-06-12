import json
import numpy as np
from pathlib import Path


def generate_static_headings(num_headings=16):
    """Generates evenly spaced unit vectors."""
    headings = []
    for i in range(num_headings):
        angle = 2 * np.pi * i / num_headings
        # Unit vector [cos(angle), sin(angle)]
        headings.append([np.cos(angle), np.sin(angle)])
    return headings

if __name__ == '__main__':
    static_headings = generate_static_headings(num_headings=16)
    
    # Save to a JSON file
    dynamics_path = Path(__file__).parent.parent / 'vehicle_dynamics' / 'terminal_headings.json'
    # Create directory if it doesn't exist
    dynamics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dynamics_path, 'w') as f:
        json.dump(static_headings, f, indent=4)
    print(f"Saved {len(static_headings)} headings to terminal_headings.json")