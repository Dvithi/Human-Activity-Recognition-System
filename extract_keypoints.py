import mediapipe as mp
import cv2
import pandas as pd
import os

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

def extract_keypoints(image_path):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = pose.process(img_rgb)
    if result.pose_landmarks:
        keypoints = []
        for lm in result.pose_landmarks.landmark:
            keypoints.extend([lm.x, lm.y, lm.z])  # 33 landmarks × 3 = 99 values
        return keypoints
    return None

# Build dataset
data = []
activities = ["walking", "sitting", "standing", "bending"]

for activity in activities:
    frames_folder = f"frames/{activity}"
    print(f"Processing {activity}...")
    for img_file in os.listdir(frames_folder):
        if img_file.endswith(".jpg"):
            kp = extract_keypoints(f"{frames_folder}/{img_file}")
            if kp:
                data.append(kp + [activity])  # keypoints + label

# Save to CSV
columns = [f"kp_{i}" for i in range(99)] + ["label"]
df = pd.DataFrame(data, columns=columns)
df.to_csv("activity_dataset.csv", index=False)
print(f"Dataset saved: {len(df)} samples")
print(df["label"].value_counts())