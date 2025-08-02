import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO
import pyttsx3
import time

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load YOLO (auto-uses GPU if available)
yolo_model = YOLO("runs/torchscript/yolov8n.torchscript", task="detect")

# Load MiDaS Small for speed
model_type = "MiDaS_small"
midas = torch.hub.load("intel-isl/MiDaS", model_type)
midas.to(device)
midas.eval()

# Transform
transform = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform

# Audio
engine = pyttsx3.init()
last_alert_time = 0
ALERT_COOLDOWN = 2

# Thresholds
CLOSE_THRESHOLD = 100
MEDIUM_THRESHOLD = 200

#Recording Setup\

recording = False
video_writer = None
alert_buffer_seconds = 5
last_close_time = 0
output_dir = "data/recorded_videos"
os.makedirs(output_dir, exist_ok=True)

# Webcam
cap = cv2.VideoCapture("http://192.168.1.9:4747/video")
frame_count = 0
depth_norm = None  # Store last depth map

while True:
    start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        break

    frame_resized = cv2.resize(frame, (640, 360))
    img_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

    # Depth estimation every 2 frames
    frame_count += 1
    if frame_count % 2 == 0 or depth_norm is None:
        input_batch = transform(img_rgb).to(device)
        with torch.no_grad():
            prediction = midas(input_batch)
        depth_map = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=frame_resized.shape[:2],
            mode="bicubic",
            align_corners=False
        ).squeeze().cpu().numpy()
        depth_norm = (depth_map * 255 / depth_map.max()).astype(np.uint8)

    # YOLO detection
    results = yolo_model.predict(frame_resized, conf=0.4, device=0)
    annotated_frame = results[0].plot()

    close_objects = []

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cls = int(box.cls[0].item())
        label = yolo_model.names[cls]

        object_depth = depth_norm[y1:y2, x1:x2]
        avg_depth = np.mean(object_depth) if object_depth.size > 0 else 0

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

        cv2.putText(
            annotated_frame,
            f"{label} - {proximity} ({avg_depth:.1f})",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )

    # Recording logic based on presence of close objects
    current_time = time.time()

    if close_objects:
        last_close_time = current_time
        if not recording:
            # Start recording
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            out_path = os.path.join(output_dir, f"alert_{timestamp}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(out_path, fourcc, 10, (annotated_frame.shape[1], annotated_frame.shape[0]))
            recording = True
            print(f"🔴 Recording started: {out_path}")

    elif recording and (current_time - last_close_time > alert_buffer_seconds):
        # Stop recording if buffer time has passed
        video_writer.release()
        video_writer = None
        recording = False
        print("🟢 Recording stopped (buffer timeout).")

    # Visual banner
    if close_objects:
        cv2.rectangle(annotated_frame, (0, 0), (annotated_frame.shape[1], 50), (0, 0, 255), -1)
        cv2.putText(annotated_frame, f"WARNING: {', '.join(set(close_objects))} VERY CLOSE!",
                    (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)    

    # Audio alert
    current_time = time.time()
    if close_objects and (current_time - last_alert_time >= ALERT_COOLDOWN):
        engine.say(f"Warning: {', '.join(set(close_objects))} very close")
        engine.runAndWait()
        last_alert_time = current_time

    # Calculate and show FPS
    fps = 1.0 / (time.time() - start_time)
    cv2.putText(
        annotated_frame,
        f"FPS: {fps:.2f}",
        (10, annotated_frame.shape[0] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),
        2,
    )   

    # Show live feed
    cv2.imshow("Live Depth Alert", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()