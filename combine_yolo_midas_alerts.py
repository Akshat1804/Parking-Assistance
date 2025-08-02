import cv2
import os
import numpy as np
import torch
from ultralytics import YOLO
import pyttsx3
import time

# Paths
frames_path = "data/frames/2011_09_26/2011_09_26_drive_0005_sync/image_00/data"
yolo_model_path = "yolov8n.pt"
depth_maps_path = "data/midas_results"
output_path = "data/final_alerts"
os.makedirs(output_path, exist_ok=True)

# Load YOLO
model = YOLO(yolo_model_path)

# Audio engine
engine = pyttsx3.init()
engine.setProperty("rate", 150)

# Depth thresholds
CLOSE_THRESHOLD = 100
MEDIUM_THRESHOLD = 200

# Cooldown
last_alert_time = 0
ALERT_COOLDOWN = 5  # seconds

# Banner settings
BANNER_HEIGHT = 50
BANNER_COLOR = (0, 0, 255)  # Red
TEXT_COLOR = (255, 255, 255)

# Process images
for img_name in sorted(os.listdir(frames_path)):
    frame = cv2.imread(os.path.join(frames_path, img_name))
    depth_map = cv2.imread(os.path.join(depth_maps_path, img_name), cv2.IMREAD_GRAYSCALE)

    if frame is None or depth_map is None:
        print(f"⚠ Skipping {img_name}, missing frame or depth map.")
        continue

    # YOLO detection
    results = model.predict(frame, conf=0.4)
    annotated_frame = results[0].plot()

    close_objects = []

    # Iterate over detections
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cls = int(box.cls[0].item())
        label = model.names[cls]

        # Extract depth region for this object
        object_depth = depth_map[y1:y2, x1:x2]
        avg_depth = np.mean(object_depth) if object_depth.size > 0 else 0

        # Classify proximity
        if avg_depth < CLOSE_THRESHOLD:
            proximity = "Close"
            color = (0, 0, 255)
            close_objects.append(label)
        elif avg_depth < MEDIUM_THRESHOLD:
            proximity = "Medium"
            color = (0, 255, 255)
        else:
            proximity = "Far"
            color = (0, 255, 0)

        # Annotate with proximity
        cv2.putText(
            annotated_frame,
            f"{label} - {proximity} ({avg_depth:.1f})",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )

    # Draw visual banner if there are close objects
    if close_objects:
        cv2.rectangle(
            annotated_frame,
            (0, 0),
            (annotated_frame.shape[1], BANNER_HEIGHT),
            BANNER_COLOR,
            -1
        )
        cv2.putText(
            annotated_frame,
            f"WARNING: {', '.join(set(close_objects))} VERY CLOSE!",
            (10, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            TEXT_COLOR,
            2
        )

    # Audio alerts with cooldown
    current_time = time.time()
    if close_objects and (current_time - last_alert_time >= ALERT_COOLDOWN):
        engine.say(f"Warning: {', '.join(set(close_objects))} very close")
        engine.runAndWait()
        last_alert_time = current_time

    # Save annotated frame
    cv2.imwrite(os.path.join(output_path, img_name), annotated_frame)

print(f"✅ Alert-based annotated images with banner saved in: {output_path}")