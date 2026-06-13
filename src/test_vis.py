import cv2
import numpy as np
import os

# Create a white background
img = np.ones((480, 640, 3), dtype=np.uint8) * 255
# Draw a large red circle in the center
cv2.circle(img, (320, 240), 50, (0, 0, 255), -1)

output_path = "data/processed_keypoints/visualizations/TEST_VISIBILITY.png"
cv2.imwrite(output_path, img)
print(f"Created test image at: {os.path.abspath(output_path)}")
