# scripts/emotion_analysis.py
import os
import pandas as pd
import matplotlib.pyplot as plt

# Example results folder containing prediction logs
results_file = "outputs/emotion_log.csv"

# If you’ve saved detection results as CSV (from YOLO or manually)
if os.path.exists(results_file):
    df = pd.read_csv(results_file)

    # Count each emotion
    emotion_counts = df['emotion'].value_counts()

    # Bar chart of emotions
    plt.figure(figsize=(7,5))
    emotion_counts.plot(kind='bar')
    plt.title("Customer Emotion Distribution")
    plt.xlabel("Emotion")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()

else:
    print("⚠️ No emotion log file found. Run prediction first.")
