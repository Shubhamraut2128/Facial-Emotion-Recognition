# scripts/live_detect.py
import cv2
import os
import pandas as pd
from datetime import datetime
from ultralytics import YOLO

# Paths
model_path = "../model/best.pt"
output_folder = "../output"
log_file = os.path.join(output_folder, "emotion_log_live.csv")
video_output_path = os.path.join(output_folder, "live_output.avi")

# Create output folder if not exists
os.makedirs(output_folder, exist_ok=True)

# Load model
model = YOLO(model_path)

# Open webcam
cap = cv2.VideoCapture(0)

# Define video writer (for saving live detection)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(video_output_path, fourcc, 20.0, (640, 480))

# Store detection results
emotion_records = []

print("🎥 Starting live detection... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO prediction
    results = model.predict(source=frame, imgsz=640, conf=0.5, verbose=False)

    # Draw boxes and labels on frame
    annotated_frame = results[0].plot()

    # Save detections
    for r in results:
        if r.boxes and hasattr(r.names, '__getitem__'):
            for box in r.boxes:
                cls_id = int(box.cls)
                conf = float(box.conf)
                label = r.names[cls_id]
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                emotion_records.append({
                    "timestamp": timestamp,
                    "emotion": label,
                    "confidence": conf
                })

    # Write frame to video
    out.write(annotated_frame)

    # Display
    cv2.imshow("Emotion Detection - Live", annotated_frame)

    # Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

# Save detections to CSV
if emotion_records:
    df = pd.DataFrame(emotion_records)
    df.to_csv(log_file, index=False)
    print(f"✅ Emotion log saved to: {log_file}")
    print(f"✅ Video saved to: {video_output_path}")
else:
    print("⚠️ No emotions detected during session.")
