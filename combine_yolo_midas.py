import cv2
import os
import numpy as np
import torch
from ultralytics import YOLO

# Paths
frames_path = "data/frames/2011_09_26/2011_09_26_drive_0005_sync/image_00/data"
yolo_model_path = "yolov8n.pt"
depth_maps_path = "data/midas_results"
output_path = "data/final_results"
os.makedirs(output_path, exist_ok=True)

# Load YOLO
model = YOLO(yolo_model_path)

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

    # Iterate over detections
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cls = int(box.cls[0].item())
        label = model.names[cls]

        # Extract depth region for this object
        object_depth = depth_map[y1:y2, x1:x2]

        if object_depth.size > 0:
            avg_depth = np.mean(object_depth)
        else:
            avg_depth = 0

        # Annotate with distance
        cv2.putText(
            annotated_frame,
            f"{label} ~ {avg_depth:.1f} depth",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    cv2.imwrite(os.path.join(output_path, img_name), annotated_frame)

print(f"✅ Final annotated images saved in: {output_path}")