import cv2
import os

# Paths
input_path = "data/final_alerts"
output_video_path = "data/final_alerts_video/alert_output.mp4"
os.makedirs(os.path.dirname(output_video_path), exist_ok=True)

# Get list of images
images = sorted([img for img in os.listdir(input_path) if img.endswith(('.png', '.jpg'))])

# Read first image to get frame size
first_frame = cv2.imread(os.path.join(input_path, images[0]))
height, width, layers = first_frame.shape

# Video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = 10  # Adjust based on your dataset
video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# Write frames to video
for img_name in images:
    frame = cv2.imread(os.path.join(input_path, img_name))
    if frame is not None:
        video_writer.write(frame)

video_writer.release()
print(f"✅ Video generated and saved at: {output_video_path}")