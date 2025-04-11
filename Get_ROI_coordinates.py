# -*- coding: utf-8 -*-
"""
Created on Tue Apr  1 16:11:19 2025

@author: duanlab
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np

# Load first frame from your video
video_path = "G:\OPF_VIDEOS/DN_20250328_16_48_52_Pro.mp4"
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
cap.release()

# Convert to grayscale for clarity
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Show the frame and let user click corners
plt.figure(figsize=(8, 8))
plt.imshow(gray, cmap='gray')
plt.title("Click 4 corners of the open field (in order), then press ENTER")
clicked_pts = plt.ginput(4, timeout=0)  # Wait for 4 clicks
plt.close()

# Show what was clicked
roi_coords = [(int(x), int(y)) for x, y in clicked_pts]
print("Your manually selected ROI coordinates:")
print(roi_coords)