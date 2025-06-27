import cv2
import os
import time
from ultralytics import YOLO
from datetime import datetime

# Load the trained YOLOv8 model
model = YOLO("G:/LuncherAi-Yolo/runs/detect/train/weights/best.pt")

# Open the webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Folder to save detected images
save_dir = "detected_rice"
os.makedirs(save_dir, exist_ok=True)

# Settings
last_detection_time = 0
DETECTION_INTERVAL = 300  # 5 minutes
CONFIDENCE_THRESHOLD = 0.70  # Minimum confidence to save

print("🟢 YOLOv8 rice detection started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Failed to read frame.")
        continue

    # Predict using YOLOv8
    results = model.predict(source=frame, conf=0.3, classes=0, verbose=False)
    annotated_frame = results[0].plot()

    # Show webcam feed
    cv2.imshow("🍚 YOLOv8 Rice Detection", annotated_frame)

    current_time = time.time()

    # Check for rice with ≥ 70% confidence
    high_confidence_detected = False
    if results[0].boxes is not None:
        for box in results[0].boxes:
            conf = float(box.conf[0])
            if conf >= CONFIDENCE_THRESHOLD:
                high_confidence_detected = True
                break

    if high_confidence_detected:
        if current_time - last_detection_time >= DETECTION_INTERVAL:
            print("✅ Rice detected with ≥70% confidence! Saving image...")

            # Save the annotated frame
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"{save_dir}/rice_{timestamp}.jpg"
            cv2.imwrite(filename, annotated_frame)

            last_detection_time = current_time
        else:
            remaining = int(DETECTION_INTERVAL - (current_time - last_detection_time))
            print(f"⏳ Waiting {remaining}s before saving another image.")
    else:
        print("❌ No rice detected with ≥70% confidence.")

    # Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("👋 Exiting...")
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
