# ===============================
# 🛠️ Required setup
# ===============================
# Make sure you have OpenCV and NumPy installed. If not, run:
# pip install opencv-python numpy
# ===============================

import cv2
import numpy as np
import time

# ===============================
# 1️⃣ Open webcam
# ===============================
cap = cv2.VideoCapture(0)  # Open default camera (0)
if not cap.isOpened():
    print("❌ Cannot open camera")
    exit()
time.sleep(2)  # Small delay to allow camera to initialize

print("👉 Adjust yourself in front of the camera, then press ENTER to capture the background!")

# ===============================
# 2️⃣ Capture the background frame
# ===============================
# The background is captured without the cloak. Later, we will replace the cloak pixels with this background.
while True:
    ret, frame = cap.read()
    if not ret:
        continue  # Skip if frame not captured

    frame = np.flip(frame, axis=1)  # Mirror effect for natural movement
    cv2.imshow("Background Setup - Press ENTER", frame)

    key = cv2.waitKey(10) & 0xFF
    if key == 13:  # ENTER key pressed
        background = frame.copy()  # Save background
        print("✅ Background captured!")
        cv2.destroyWindow("Background Setup - Press ENTER")
        break
    elif key == ord('q'):  # Quit if 'q' pressed
        cap.release()
        cv2.destroyAllWindows()
        exit()

# ===============================
# 3️⃣ Main loop: Ultra-Magical Smooth Cloak
# ===============================
prev_mask = None
temporal_alpha = 0.35  # Lower value → faster mask update, ghost effect
min_area_ratio = 0.02  # Minimum size of cloak area to consider

height, width = background.shape[:2]
frame_area = height * width

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        continue
    img = np.flip(img, axis=1)  # Mirror the frame
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # Convert to HSV color space

    # ===============================
    # 3a️⃣ Detect red color (cloak)
    # ===============================
    lower_red1 = np.array([0, 100, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 100, 50])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)  # Combine two red ranges

    # ===============================
    # 3b️⃣ Morphological operations to clean mask
    # ===============================
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)   # Remove small noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)  # Close small holes
    mask = cv2.dilate(mask, kernel, iterations=3)  # Expand mask slightly
    mask = cv2.erode(mask, kernel, iterations=1)   # Smooth edges

    # ===============================
    # 3c️⃣ Keep only large connected components
    # ===============================
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask_filtered = np.zeros_like(mask)
    for cnt in contours:
        if cv2.contourArea(cnt) > min_area_ratio * frame_area:
            cv2.drawContours(mask_filtered, [cnt], -1, 255, -1)
    mask = mask_filtered

    # ===============================
    # 3d️⃣ Temporal smoothing for silky effect
    # ===============================
    mask_float = mask.astype(float)/255.0
    if prev_mask is not None and np.sum(mask_float) > 0:
        mask_float = temporal_alpha * prev_mask + (1 - temporal_alpha) * mask_float
    prev_mask = mask_float.copy()

    # ===============================
    # 3e️⃣ Edge softening (multi-scale blur)
    # ===============================
    edges = cv2.Canny((mask_float*255).astype(np.uint8), 50, 150)
    edges = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=1)
    edge_blur_small = cv2.GaussianBlur(edges.astype(float)/255.0, (11,11), 0)
    edge_blur_large = cv2.GaussianBlur(edges.astype(float)/255.0, (21,21), 0)
    mask_float = np.clip(mask_float + edge_blur_small*0.6 + edge_blur_large*0.4, 0, 1)
    mask_inv_float = 1.0 - mask_float

    # ===============================
    # 3f️⃣ Blend cloak with background
    # ===============================
    final_output = (img * mask_inv_float[:,:,None] + background * mask_float[:,:,None]).astype(np.uint8)

    # Final silky smooth blur
    final_output = cv2.GaussianBlur(final_output, (5,5), 0)

    # ===============================
    # 3g️⃣ Display
    # ===============================
    cv2.imshow("Ultra-Magical Red Cloak - Press 'q' to Exit", final_output)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ===============================
# 4️⃣ Release resources
# ===============================
cap.release()
cv2.destroyAllWindows()
