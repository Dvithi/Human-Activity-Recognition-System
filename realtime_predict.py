import cv2
import mediapipe as mp
import numpy as np
import pickle

model = pickle.load(open("har_model.pkl", "rb"))
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.6)

COLORS = {
    "walking":  (0, 201, 167),
    "sitting":  (255, 199, 95),
    "standing": (249, 248, 113),
    "bending":  (255, 111, 145),
}

cap = cv2.VideoCapture(0)  # 0 = webcam
print("Press Q to quit")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(rgb)

    label = "No person detected"
    color = (200, 200, 200)

    if result.pose_landmarks:
        mp_draw.draw_landmarks(frame, result.pose_landmarks,
                               mp_pose.POSE_CONNECTIONS)
        kp = []
        for lm in result.pose_landmarks.landmark:
            kp.extend([lm.x, lm.y, lm.z])

        prediction = model.predict([kp])[0]
        confidence = model.predict_proba([kp]).max() * 100
        label = f"{prediction.upper()}  {confidence:.1f}%"
        color = COLORS.get(prediction, (255, 255, 255))

    # Display label box
    cv2.rectangle(frame, (0, 0), (400, 50), (30, 30, 30), -1)
    cv2.putText(frame, label, (10, 35), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, color, 2)

    cv2.imshow("Human Activity Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
