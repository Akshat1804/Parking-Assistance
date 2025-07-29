from ultralytics import YOLO
import cv2
import os

# Load YOLOv8 model (nano for speed)
model = YOLO("yolov8n.pt")  

# Paths
input_path = "data/frames/2011_09_26/2011_09_26_drive_0005_sync/image_00/data"
output_path = "data/yolo_results"
os.makedirs(output_path, exist_ok=True)

# Loop through frames
image_files = sorted(os.listdir(input_path))

for img_name in image_files:
    img_path = os.path.join(input_path, img_name)

    # Run YOLO inference
    results = model.predict(source=img_path, conf=0.4)

    # Save result image
    result_img = results[0].plot()
    cv2.imwrite(os.path.join(output_path, img_name), result_img)

print(f"✅ Detection completed. Results saved in: {output_path}")