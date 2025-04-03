# Lida Du wrote this script to analyze and plot tracked animal movement coordinates in a 2D space
# Contact: ldu13@jh.edu
# Libraries for open-field analysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

# Load OPF mouse data
file_path = "G:/OPF_VIDEOS/20250403_143-1-track.txt"  # Directory to the coordinates file in *.TXT format

# Extract coordinates
with open(file_path, "r") as f: 
    content = f.read()
pattern = r"(\d+\.\d+)\s+(\d+\.\d+)"
matches = re.findall(pattern, content)
coordinates = pd.DataFrame(matches, columns=["X", "Y"]).astype(float)

# Scale the side length of the square open field
x_range = coordinates["X"].max() - coordinates["X"].min()
y_range = coordinates["Y"].max() - coordinates["Y"].min()
side_length = np.mean([x_range, y_range])
scale = 40 / side_length
coordinates_scaled = coordinates * scale

# Normalize coordinates
coordinates_normalized = coordinates_scaled.copy()
coordinates_normalized["X"] -= coordinates_scaled["X"].min()
coordinates_normalized["Y"] -= coordinates_scaled["Y"].min()

# Travel distance (in meters)
distances = np.sqrt(np.diff(coordinates_scaled["X"])**2 + np.diff(coordinates_scaled["Y"])**2)
total_distance_cm = distances.sum()

# Average speed in 60Hz frame rate (Check your camera setting, we take 60FPS as an example here)
sampling_rate = 60
total_time_s = len(coordinates_scaled) / sampling_rate
average_speed_mps = total_distance_cm / total_time_s

# Define the area of the middle zone(s)
middle_mask = (
    (coordinates_normalized["X"] >= 10) & (coordinates_normalized["X"] <= 30) &
    (coordinates_normalized["Y"] >= 10) & (coordinates_normalized["Y"] <= 30)
)
time_middle_square_s = middle_mask.sum() / sampling_rate

# Count entries into the middle zone
middle_entries = (middle_mask & ~middle_mask.shift(1, fill_value=False)).sum()

# Average time per visit
dwell_times = []
in_zone = False
counter = 0
for point in middle_mask:
    if point:
        counter += 1
        in_zone = True
    elif in_zone:
        dwell_times.append(counter / sampling_rate)
        counter = 0
        in_zone = False
# Handle final stay time if it ends inside the middle
if in_zone and counter > 0:
    dwell_times.append(counter / sampling_rate)

# Draw open-field
def draw_field(ax):
    outer = plt.Rectangle((0, 0), 40, 40, fill=False, color='black', linewidth=1.5, linestyle='--')
    middle = plt.Rectangle((10, 10), 20, 20, fill=False, color='red', linewidth=1.5, linestyle=':')
    ax.add_patch(outer)
    ax.add_patch(middle)

# Print results
print(f"Total distance traveled: {total_distance_m:.2f} meters")
print(f"Average speed: {average_speed_mps:.2f} m/s")
print(f"Dwell time in middle square: {time_middle_square_s:.2f} s")
print(f"Number of entries into middle square: {middle_entries}")
print(f"Dwell times in middle square (s): {np.round(dwell_times, 2).tolist()}")

# Print tab-separated values for Excel
print("\nCopy and paste into Excel:")
print("\t".join(metrics))
print("\t".join(map(str, values)))

# Unmark this to save summary to CSV file
# summary_df.to_csv("mouse_summary.csv", index=False)

# Plot animal track
fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(coordinates_normalized["X"], coordinates_normalized["Y"], color='blue', linewidth=1)
draw_field(ax)
ax.set_title("Your_title_here") # Define your plot title here
ax.set_xlim(0, 40)
ax.set_ylim(0, 40)
ax.set_aspect('equal')
ax.set_xlabel("X (cm)")
ax.set_ylabel("Y (cm)")
ax.grid(True)

plt.tight_layout()
plt.show()
