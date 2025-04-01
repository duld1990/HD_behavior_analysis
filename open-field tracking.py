# Libraries for open-field analysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

# Load OPF mouse data
file_path = "/test_data.txt"  # Directory to the coordinates file in *.TXT format

# Extract coordinates
with open(file_path, "r") as f:
    content = f.read()
pattern = r"(\d+\.\d+)\s+(\d+\.\d+)"
matches = re.findall(pattern, content)
coordinates = pd.DataFrame(matches, columns=["X", "Y"]).astype(float)

# Estimate the side length of the square open field
x_range = coordinates["X"].max() - coordinates["X"].min()
y_range = coordinates["Y"].max() - coordinates["Y"].min()
side_length = np.mean([x_range, y_range])

# Scale coordinates so that the open-field is 1x1 meter
scale = 1.0 / side_length
coordinates_scaled = coordinates * scale

# Normalize coordinates to fit in [0, 1] x [0, 1]
coordinates_normalized = coordinates_scaled.copy()
coordinates_normalized["X"] -= coordinates_scaled["X"].min()
coordinates_normalized["Y"] -= coordinates_scaled["Y"].min()

# Travel distance (in meters)
distances = np.sqrt(np.diff(coordinates_scaled["X"])**2 + np.diff(coordinates_scaled["Y"])**2)
total_distance_m = distances.sum()

# Average speed in 60Hz frame rate
sampling_rate = 60
total_time_s = len(coordinates_scaled) / sampling_rate
average_speed_mps = total_distance_m / total_time_s

# Draw open-field
def draw_field(ax):
    field = plt.Rectangle((0, 0), 1, 1, fill=False, color='black', linewidth=1.5, linestyle='--')
    ax.add_patch(field)

# Print results
print(f"Total distance traveled: {total_distance_m:.2f} meters")
print(f"Average speed: {average_speed_mps:.2f} m/s")

# Plot animal track
fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(coordinates_normalized["X"], coordinates_normalized["Y"], color='blue', linewidth=1)
draw_field(ax)
ax.set_title("Your_title_here") # Define your plot title here
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_aspect('equal')
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.grid(True)

plt.tight_layout()
plt.show()
