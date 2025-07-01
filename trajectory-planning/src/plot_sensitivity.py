# trajectory-planning/src/plot_sensitivity.py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

# Set which variable you're analyzing
VARY = "umax"  # Change to: "N", "Ne", or "umax" as needed

# Define the paths
ROOT = Path(__file__).parent
csv_path = ROOT / "Sensitivity results" / f"sensitivity_{VARY}_with_maps.csv"

# Output folder for figures
figures_dir = ROOT / "figures"
figures_dir.mkdir(parents=True, exist_ok=True)

# Load the data
df = pd.read_csv(csv_path)

# Define metrics and map order
metrics = ["computation_time", "arrival_time", "penalty_start", "penalty_rate"]
map_order = ["easy_map_sa", "baseline_map_sa", "hard_map_sa"]

# Plot the results
sns.set(style="whitegrid")

for metric in metrics:
    plt.figure(figsize=(8, 5))
    sns.barplot(
        data=df,
        x=VARY,
        y=metric,
        hue="map",
        order=sorted(df[VARY].unique()),
        hue_order=map_order,
        errorbar="sd"
    )
    plt.title(f"{metric.replace('_', ' ').title()} vs {VARY}")
    plt.xlabel(VARY.title())
    plt.ylabel(metric.replace('_', ' ').title())
    plt.legend(title="Map")
    plt.tight_layout()
    
    # Save to the figures folder
    fig_path = figures_dir / f"{VARY}_{metric}_bars.pdf"
    plt.savefig(fig_path)
    print(f"Saved: {fig_path}")
    plt.show()
