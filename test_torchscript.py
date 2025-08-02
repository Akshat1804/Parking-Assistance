import torch
import cv2
from ultralytics import YOLO

# Load TorchScript model
model = YOLO("runs/torchscript/yolov8n.torchscript")

# Test on an image
img = cv2.imread("data/frames/2011_09_26/2011_09_26_drive_0005_sync/image_01/data/0000000000.png")
results = model(img)

# Show results
annotated = results[0].plot()
cv2.imshow("YOLO TorchScript Output", annotated)
cv2.waitKey(0)
cv2.destroyAllWindows()