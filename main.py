from ultralytics import YOLO
import cv2

# Load the YOLO model
model = YOLO("yolov8l.pt")

# load an image
image_path = "data/input.jpg"
image = cv2.imread(image_path)
image = cv2.resize(image, (640, 640))  

# Perform inference
results = model.predict(source=image, conf=0.1)

# Print results

annonated_frame = results[0].plot()

#Show the result
cv2.imshow("YOLOv8 Detection", annonated_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()

for r in results:
    for box in r.boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id]
        confidence = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        print(f"Detected {label} with confidence {confidence:.2f} at [{x1}, {y1}, {x2}, {y2}]")
    