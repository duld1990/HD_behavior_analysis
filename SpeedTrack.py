# Animal speed track
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

# Load OPF mouse data
file_path = r"\test_data.txt"  # Directory

# Extract coordinates
with open(file_path, "r") as f:
    content = f.read()
pattern = r"(\d+\.\d+)\s+(\d+\.\d+)"
matches = re.findall(pattern, content)
coordinates = pd.DataFrame(matches, columns=["X", "Y"]).astype(float)

# Scale field to 1m x 1m
x_range = coordinates["X"].max() - coordinates["X"].min()
y_range = coordinates["Y"].max() - coordinates["Y"].min()
side_length = np.mean([x_range, y_range])
scale = 1.0 / side_length
coordinates_scaled = coordinates * scale

# Normalize coordinates
coordinates_normalized = coordinates_scaled.copy()
coordinates_normalized["X"] -= coordinates_scaled["X"].min()
coordinates_normalized["Y"] -= coordinates_scaled["Y"].min()

# Calculate speed (Euclidean distance between points × 60Hz)
dx = coordinates_normalized["X"].diff()
dy = coordinates_normalized["Y"].diff()
speed = np.sqrt(dx**2 + dy**2) * 60  # speed in m/s
coordinates_normalized["speed"] = speed.fillna(0)

# Middle square mask (0.25–0.75 m)
middle_mask = (
    (coordinates_normalized["X"] >= 0.25) & (coordinates_normalized["X"] <= 0.75) &
    (coordinates_normalized["Y"] >= 0.25) & (coordinates_normalized["Y"] <= 0.75)
)

# Average speed in the middle square
if middle_mask.sum() > 0:
    average_middle_speed = coordinates_normalized.loc[middle_mask, "speed"].mean()
else:
    average_middle_speed = 0.0

# Draw field
def draw_squares(ax):
    outer = plt.Rectangle((0, 0), 1, 1, fill=False, color='black', linewidth=1.5, linestyle='--')
    middle = plt.Rectangle((0.25, 0.25), 0.5, 0.5, fill=False, color='red', linewidth=1.5, linestyle=':')
    ax.add_patch(outer)
    ax.add_patch(middle)

# Plot speed heatmap
fig, ax = plt.subplots(figsize=(6, 6))
sc = ax.scatter(coordinates_normalized["X"], coordinates_normalized["Y"],
                c=coordinates_normalized["speed"], cmap="viridis", s=3)
draw_squares(ax)
ax.set_title("Speed Heatmap - WT Mouse")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_aspect('equal')
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.grid(True)

# Add colorbar
cbar = plt.colorbar(sc, ax=ax, orientation='vertical')
cbar.set_label("Speed (m/s)")

plt.tight_layout()
plt.show()

# Output dwell time
print(f"Dwell time in middle square: {dwell_time_s:.2f} seconds")
