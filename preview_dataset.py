import cv2
import matplotlib.pyplot as plt
import os

# Path to your frames
frames_path = "data/frames/2011_09_26/2011_09_26_drive_0005_sync/image_00/data"
image_files = sorted(os.listdir(frames_path))

# Preview first 5 images
for img_name in image_files[:5]:
    img = cv2.imread(os.path.join(frames_path, img_name))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.imshow(img_rgb)
    plt.title(img_name)
    plt.axis('off')
    plt.show()