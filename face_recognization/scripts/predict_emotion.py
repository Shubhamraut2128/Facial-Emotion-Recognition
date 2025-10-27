# scripts/predict_emotion.py
from ultralytics import YOLO
import os
import cv2

# Load your trained YOLO model
model = YOLO("../model/best.pt")


# Folder paths
input_folder = "../input"
output_folder = "../output"
os.makedirs(output_folder, exist_ok=True)

# Loop through all images
for file_name in os.listdir(input_folder):
    if file_name.lower().endswith(('.jpg', '.png', '.jpeg')):
        img_path = os.path.join(input_folder, file_name)
        results = model.predict(source=img_path, save=False)
        for r in results:
            annotated = r.plot()
            save_path = os.path.join(output_folder, file_name)
            cv2.imwrite(save_path, annotated)
            print(f"✅ Processed: {file_name}")

print("🎉 All predictions completed!")
