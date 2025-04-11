# -*- coding: utf-8 -*-
"""
Created on Mon Apr  7 15:51:17 2025

@author: Lida Du
ldu13@jh.edu
"""

import os
import cv2
import numpy as np
%matplotlib qt
import matplotlib.pyplot as plt

# Directories and camera settings
video_path = r"G:\OPF_VIDEOS\WIN_20250403_13_54_53_Pro.mp4"  # Full path to your video file
output_dir = r"G:\OPF_VIDEOS"  # Directory where results will be saved
sampling_rate = 60  # Desired frame sampling rate (frames per second)
live_preview = False  # Set to True to enable live tracking preview

os.makedirs(output_dir, exist_ok=True)

def select_roi(frame):
    """
    Let the user select the region-of-interest (ROI) by clicking 4 corners (in clockwise order).
    Returns the ROI as a numpy array of coordinates.
    """
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    plt.figure(figsize=(8, 8))
    plt.imshow(gray_frame, cmap='gray')
    plt.title("Click 4 corners of the open field (clockwise), then press ENTER")
    clicked_pts = plt.ginput(4, timeout=0)
    plt.close()
    roi_coords = [(int(x), int(y)) for x, y in clicked_pts]
    roi = np.array(roi_coords, dtype=np.int32)
    print("Selected ROI Coordinates:", roi_coords)
    return roi

def create_mask(shape, roi):
    """
    Create a binary mask for the ROI.
    """
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(mask, [roi], 255)
    return mask

def threshold_preview(gray_sample, mask):
    """
    Display an interactive threshold preview.
    Adjust the threshold via a trackbar and let the user select the mouse blob with a mouse click.
    Returns the chosen threshold value and click coordinates.
    """
    clicked_point = None

    def click_callback(event, x, y, flags, param):
        nonlocal clicked_point
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked_point = (x, y)
            print(f"User clicked at: {clicked_point}")

    
    def click_callback(event, x, y, flags, param):
        nonlocal clicked_point
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked_point = (x, y)
            print(f"User clicked at: {clicked_point}")

    def on_trackbar(val):
        masked_sample = cv2.bitwise_and(gray_sample, gray_sample, mask=mask)
        # Use THRESH_BINARY_INV so the dark mouse becomes white
        _, thresh = cv2.threshold(masked_sample, val, 255, cv2.THRESH_BINARY_INV)
        preview = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 100:
                cv2.drawContours(preview, [cnt], -1, (255, 255, 255), 1)
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    cv2.circle(preview, (cx, cy), 3, (0, 255, 0), -1)
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(preview, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.imshow("Threshold Preview", preview)

    cv2.namedWindow("Threshold Preview")
    cv2.createTrackbar("Threshold", "Threshold Preview", 50, 255, on_trackbar)
    on_trackbar(50)
    cv2.setMouseCallback("Threshold Preview", click_callback)

    print("Adjust threshold using the trackbar, then click on the mouse blob to select it.")
    while clicked_point is None:
        cv2.waitKey(10)

    selected_thresh = cv2.getTrackbarPos("Threshold", "Threshold Preview")
    cv2.destroyWindow("Threshold Preview")
    print(f"Selected threshold: {selected_thresh}")
    print(f"Selected tracking click point: {clicked_point}")
    return selected_thresh, clicked_point

def track_mouse(video_path, mask, selected_thresh, clicked_point, sampling_rate, live_preview=False):
    """
    Track the mouse over the video using the selected threshold and click point.
    The sampling_rate determines how many frames per second are processed by skipping frames as needed.
    For each processed frame, the contour closest to the clicked point is chosen,
    and both the center and a nose estimate (via ellipse fit) are computed.
    If live_preview is True, the tracking is shown in real time.
    """
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video FPS: {video_fps}")
    # Calculate frame interval based on the desired sampling rate.
    if sampling_rate < video_fps:
        frame_interval = int(round(video_fps / sampling_rate))
    else:
        frame_interval = 1

    max_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_frame = 10  # starting from frame 10
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    tracked_points = []
    frame_idx = start_frame

    while frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        # Process only frames that match the sampling interval.
        if (frame_idx - start_frame) % frame_interval != 0:
            frame_idx += 1
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        masked_gray = cv2.bitwise_and(gray, gray, mask=mask)
        # Use THRESH_BINARY_INV so that the dark mouse becomes white
        _, thresh = cv2.threshold(masked_gray, selected_thresh, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 100]
        if not valid_contours:
            tracked_points.append({"center": None, "nose": None})
            frame_idx += 1
            continue

        # Use absolute distance to select the contour closest to the clicked point.
        mouse_contour = min(valid_contours, key=lambda cnt: abs(cv2.pointPolygonTest(cnt, clicked_point, True)))
        M = cv2.moments(mouse_contour)
        if M["m00"] == 0:
            tracked_points.append({"center": None, "nose": None})
            frame_idx += 1
            continue

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        center = (cx, cy)

        # Estimate nose location using an ellipse fit if enough points are present.
        if len(mouse_contour) >= 5:
            (x, y), (MA, ma), angle = cv2.fitEllipse(mouse_contour)
            vec = np.array([np.cos(np.radians(angle)), np.sin(np.radians(angle))])
            nose_est = (int(cx + vec[0] * ma / 2), int(cy + vec[1] * ma / 2))
        else:
            nose_est = center

        tracked_points.append({"center": center, "nose": nose_est})

        # Optionally show a live preview of tracking.
        if live_preview:
            frame_preview = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            cv2.drawContours(frame_preview, [mouse_contour], -1, (255, 255, 255), 1)
            cv2.circle(frame_preview, center, 4, (0, 255, 0), -1)
            cv2.circle(frame_preview, nose_est, 3, (0, 0, 255), -1)
            x, y, w, h = cv2.boundingRect(mouse_contour)
            cv2.rectangle(frame_preview, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.imshow("Live Tracking", frame_preview)
            if cv2.waitKey(1) & 0xFF == 27:  # Exit live preview on pressing Esc
                break

        frame_idx += 1

    cap.release()
    if live_preview:
        cv2.destroyWindow("Live Tracking")
    return tracked_points

def save_tracking_results(output_dir, tracked_points):
    """
    Save the tracked nose and body (center) coordinates into text files.
    The files are named using the base name of the processed video.
    """
    video_basename = os.path.splitext(os.path.basename(video_path))[0]
    nose_points = [f"{pt['nose'][0]:.5f} {pt['nose'][1]:.5f}" for pt in tracked_points if pt["nose"]]
    body_points = [f"{pt['center'][0]:.5f} {pt['center'][1]:.5f}" for pt in tracked_points if pt["center"]]

    nose_file = os.path.join(output_dir, f"nose_track_{video_basename}.txt")
    body_file = os.path.join(output_dir, f"body_track_{video_basename}.txt")
    with open(nose_file, 'w') as f:
        f.write("\n".join(nose_points))
    with open(body_file, 'w') as f:
        f.write("\n".join(body_points))
    print(f"Tracking complete for {video_basename}. Results saved to:")
    print(f"  {nose_file}\n  {body_file}")

def tracking_preview(gray_frame, tracked_points, roi):
    """
    Display a final preview of the tracking overlaid on the first frame.
    """
    preview = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)
    for pt in tracked_points[:300:20]:
        if pt["center"]:
            cv2.circle(preview, pt["center"], 4, (0, 255, 0), -1)
        if pt["nose"]:
            cv2.circle(preview, pt["nose"], 3, (0, 0, 255), -1)
    cv2.polylines(preview, [roi], isClosed=True, color=(255, 255, 0), thickness=2)
    plt.imshow(preview)
    plt.title("Tracking Preview (Green: Body, Red: Nose, Yellow: ROI)")
    plt.axis("off")
    plt.show()

def process_video():
    """
    Process the video file: let the user select ROI and threshold interactively,
    track the mouse, save the results, and show a tracking preview.
    """
    print(f"Processing video: {video_path}")

    # Load the first frame to select ROI.
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if not ret:
        print(f"Error: Could not read from video {video_path}")
        return
    cap.release()

    # Let the user select the ROI.
    roi = select_roi(frame)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mask = create_mask(gray_frame.shape, roi)

    # Allow interactive threshold preview and blob selection.
    gray_sample = gray_frame.copy()
    selected_thresh, clicked_point = threshold_preview(gray_sample, mask)

    # Warn if the clicked point is outside the ROI.
    if mask[clicked_point[1], clicked_point[0]] == 0:
        print("Warning: Clicked point is outside the selected ROI.")

    # Track the mouse over the video.
    tracked_points = track_mouse(video_path, mask, selected_thresh, clicked_point,
                                 sampling_rate, live_preview=live_preview)

    # Save tracking results.
    save_tracking_results(output_dir, tracked_points)

    # Show a final preview.
    tracking_preview(gray_frame, tracked_points, roi)

def main():
    process_video()

if __name__ == "__main__":
    main()
