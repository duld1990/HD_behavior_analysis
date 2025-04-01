# Novel object recognition within the open field
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

# Define regex pattern
pattern = r"(\d+\.\d+)\s+(\d+\.\d+)"

# File paths and estimated side lengths (used for 1m normalization)
file_data = {
    "Mouse 1 (DN)": ("/DN1.txt", 167.96),
    "Mouse 2 (WT1)": ("/WT1.txt", 175.90),
    "Mouse 3 (WT2)": ("/WT2.txt", 95.85),
    "Mouse 4 (WT3)": ("/WT3.txt", 173.33)
}

# Load and scale coordinate data
tracks_scaled = []
titles = []
for label, (path, side_length) in file_data.items():
    with open(path, "r") as f:
        content = f.read()
    matches = re.findall(pattern, content)
    coords = pd.DataFrame(matches, columns=["X", "Y"]).astype(float)
    scale = 1.0 / side_length
    coords_scaled = coords * scale
    tracks_scaled.append(coords_scaled)
    titles.append(label)

# Define a middle square spans from (0.25, 0.25) to (0.75, 0.75)
object_1 = (0.25, 0.75)  # Identical object
object_2 = (0.75, 0.25)  # Identical/new object
radius = 0.15  

# Calculate exploration time for each object per mouse
exploration_times = {}
fig, axs = plt.subplots(2, 2, figsize=(12, 12))

for ax, track, title in zip(axs.flatten(), normalized_tracks, titles):
    # Distance to each object
    dist_o1 = np.sqrt((track["X"] - object_1[0])**2 + (track["Y"] - object_1[1])**2)
    dist_o2 = np.sqrt((track["X"] - object_2[0])**2 + (track["Y"] - object_2[1])**2)

    # Points within 15cm of each object
    in_o1 = dist_o1 <= radius
    in_o2 = dist_o2 <= radius

    # Exploration time (in seconds)
    time_o1 = round(in_o1.sum() / 60, 2)
    time_o2 = round(in_o2.sum() / 60, 2)
    exploration_times[title] = {"Object 1": time_o1, "Object 2": time_o2}

    # Plot track
    ax.plot(track["X"], track["Y"], color='blue', linewidth=1)
    draw_squares(ax)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(title)
    ax.grid(True)

    # Mark AOIs
    o1_circle = plt.Circle(object_1, radius, color='green', fill=False, linestyle='--', linewidth=2, label='Object 1 AOI')
    o2_circle = plt.Circle(object_2, radius, color='orange', fill=False, linestyle='--', linewidth=2, label='Object 2 AOI')
    ax.add_patch(o1_circle)
    ax.add_patch(o2_circle)

plt.tight_layout()
plt.show()

exploration_times
