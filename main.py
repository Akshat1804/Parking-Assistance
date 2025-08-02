# from ultralytics import YOLO
# import cv2
# import os

# # Load YOLOv8 model (nano for speed)
# model = YOLO("runs/torchscript/yolov8n.torchscript", task="detect")

# # Paths
# input_path = "data/frames/2011_09_26/2011_09_26_drive_0005_sync/image_01/data"
# output_path = "data/yolo_results"
# os.makedirs(output_path, exist_ok=True)

# # Loop through frames
# image_files = sorted(os.listdir(input_path))

# for img_name in image_files:
#     img_path = os.path.join(input_path, img_name)

#     # Run YOLO inference
#     results = model.predict(source=img_path, conf=0.4)

#     # Save result image
#     result_img = results[0].plot()
#     cv2.imwrite(os.path.join(output_path, img_name), result_img)

# print(f"✅ Detection completed. Results saved in: {output_path}")

# import torch
# import cv2
# import numpy as np
# import os

# # Load MiDaS model and transform
# model_type = "DPT_Hybrid"
# model = torch.hub.load("intel-isl/MiDaS", model_type, source="github")
# model.eval()

# midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", source="github")
# transform = midas_transforms.dpt_transform

# # Paths
# input_path = "data/frames/2011_09_26/2011_09_26_drive_0005_sync/image_01/data"
# output_path = "data/midas_results"
# os.makedirs(output_path, exist_ok=True)

# # Loop through images
# for img_name in sorted(os.listdir(input_path)):
#     img_path = os.path.join(input_path, img_name)

#     # Load image
#     img = cv2.imread(img_path)
#     if img is None:
#         print(f"⚠ Skipping {img_name}, cannot read file.")
#         continue

#     img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     img_rgb = img_rgb.astype(np.float32) / 255.0

#     # Transform (no unsqueeze)
#     input_batch = transform(img_rgb)

#     with torch.no_grad():
#         prediction = model(input_batch)

#     # Resize depth map to original image size
#     prediction = torch.nn.functional.interpolate(
#         prediction.unsqueeze(1),
#         size=img.shape[:2],
#         mode="bicubic",
#         align_corners=False,
#     ).squeeze()

#     depth_map = prediction.cpu().numpy()

#     # Normalize for saving
#     depth_normalized = (depth_map * 255 / depth_map.max()).astype("uint8")
#     cv2.imwrite(os.path.join(output_path, img_name), depth_normalized)

# print(f"✅ All depth maps saved in: {output_path}")

# import cv2
# import os
# import numpy as np
# import torch
# from ultralytics import YOLO

# # Paths
# frames_path = "data/frames/2011_09_26/2011_09_26_drive_0005_sync/image_01/data"
# yolo_model_path = "yolov8n.pt"
# depth_maps_path = "data/midas_results"
# output_path = "data/final_results"
# os.makedirs(output_path, exist_ok=True)

# # Load YOLO
# model = YOLO(yolo_model_path)

# # Process images
# for img_name in sorted(os.listdir(frames_path)):
#     frame = cv2.imread(os.path.join(frames_path, img_name))
#     depth_map = cv2.imread(os.path.join(depth_maps_path, img_name), cv2.IMREAD_GRAYSCALE)

#     if frame is None or depth_map is None:
#         print(f"⚠ Skipping {img_name}, missing frame or depth map.")
#         continue

#     # YOLO detection
#     results = model.predict(frame, conf=0.4)
#     annotated_frame = results[0].plot()

#     # Iterate over detections
#     for box in results[0].boxes:
#         x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
#         cls = int(box.cls[0].item())
#         label = model.names[cls]

#         # Extract depth region for this object
#         object_depth = depth_map[y1:y2, x1:x2]

#         if object_depth.size > 0:
#             avg_depth = np.mean(object_depth)
#         else:
#             avg_depth = 0

#         # Annotate with distance
#         cv2.putText(
#             annotated_frame,
#             f"{label} ~ {avg_depth:.1f} depth",
#             (x1, y1 - 10),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             0.6,
#             (0, 255, 0),
#             2,
#         )

#     cv2.imwrite(os.path.join(output_path, img_name), annotated_frame)

# print(f"✅ Final annotated images saved in: {output_path}")

# import cv2
# import os

# # Paths
# input_path = "data/final_results"
# output_video_path = "data/final_results_video/alert_output.mp4"
# os.makedirs(os.path.dirname(output_video_path), exist_ok=True)

# # Get list of images
# images = sorted([img for img in os.listdir(input_path) if img.endswith(('.png', '.jpg'))])

# # Read first image to get frame size
# first_frame = cv2.imread(os.path.join(input_path, images[0]))
# height, width, layers = first_frame.shape

# # Video writer
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# fps = 10  # Adjust based on your dataset
# video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# # Write frames to video
# for img_name in images:
#     frame = cv2.imread(os.path.join(input_path, img_name))
#     if frame is not None:
#         video_writer.write(frame)

# video_writer.release()
# print(f"✅ Video generated and saved at: {output_video_path}")


import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO

# Paths
frames_path = "data/frames/2011_09_26/2011_09_26_drive_0005_sync/image_01/data"
output_video_path = "data/final_results_video/alert_output.mp4"
os.makedirs(os.path.dirname(output_video_path), exist_ok=True)

# Load YOLO TorchScript model
yolo_model = YOLO("runs/torchscript/yolov8n.torchscript", task="detect")

# Load MiDaS PyTorch model
midas_model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
midas_model.eval()

# MiDaS transform
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.small_transform

# Get frame list
image_files = sorted([img for img in os.listdir(frames_path) if img.endswith(('.png', '.jpg'))])

# Read first frame to get video size
first_frame = cv2.imread(os.path.join(frames_path, image_files[0]))
height, width, layers = first_frame.shape

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = 10
video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

for img_name in image_files:
    img_path = os.path.join(frames_path, img_name)
    frame = cv2.imread(img_path)

    # YOLO inference
    yolo_results = yolo_model.predict(frame, conf=0.4)
    annotated_frame = yolo_results[0].plot()

    # MiDaS depth estimation
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_rgb = img_rgb.astype(np.float32) / 255.0
    input_batch = transform(img_rgb)

    with torch.no_grad():
        depth_pred = midas_model(input_batch)

    depth_map = torch.nn.functional.interpolate(
        depth_pred.unsqueeze(1),
        size=frame.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze().cpu().numpy()

    depth_normalized = (depth_map * 255 / depth_map.max()).astype("uint8")

    # Combine YOLO detections with depth info
    for box in yolo_results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cls = int(box.cls[0].item())
        label = yolo_model.names[cls]

        object_depth = depth_normalized[y1:y2, x1:x2]
        avg_depth = np.mean(object_depth) if object_depth.size > 0 else 0

        cv2.putText(
            annotated_frame,
            f"{label} ~ {avg_depth:.1f} depth",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    video_writer.write(annotated_frame)

video_writer.release()
print(f"✅ Final video generated and saved at: {output_video_path}")