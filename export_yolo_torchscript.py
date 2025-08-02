from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Export to TorchScript
model.export(format="torchscript")

print("✅ YOLO model successfully exported to TorchScript.")