import torch
import cv2
import numpy as np
import os

# Load TorchScript MiDaS
midas = torch.jit.load("midas_small.torchscript")
midas.eval()

# Transform
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.small_transform

# Paths
input_path = "data/frames/2011_09_26/2011_09_26_drive_0005_sync/image_00/data"
output_path = "data/midas_results"
os.makedirs(output_path, exist_ok=True)

# Loop through images
for img_name in sorted(os.listdir(input_path)):
    img_path = os.path.join(input_path, img_name)
    img = cv2.imread(img_path)
    if img is None:
        print(f"⚠ Skipping {img_name}, cannot read file.")
        continue

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_rgb = img_rgb.astype(np.float32) / 255.0

    input_batch = transform(img_rgb)

    with torch.no_grad():
        prediction = midas(input_batch)

    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=img.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze()

    depth_map = prediction.cpu().numpy()
    depth_normalized = (depth_map * 255 / depth_map.max()).astype("uint8")
    cv2.imwrite(os.path.join(output_path, img_name), depth_normalized)

print(f"✅ All depth maps saved in: {output_path}")