import cv2
import mediapipe as mp
import json
import os

#From chat 
# c -> saves points to a json 
# q -> exists program 

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Open webcam (adjust index if needed)
cap = cv2.VideoCapture(1)

# Counter for saved frames
capture_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert BGR to RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame
    results = pose.process(rgb)

    # Draw landmarks if detected
    drawing_spec = mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2)
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=drawing_spec
        )

    # Show frame
    cv2.imshow("Pose", frame)

    # Wait for key press
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break
    elif key == ord("c") and results.pose_landmarks:
        # Save landmarks to JSON
        pose_data = []
        for lm in results.pose_landmarks.landmark:
            pose_data.append({
                "x": lm.x,
                "y": lm.y,
                "z": lm.z,
                "visibility": lm.visibility
            })

        filename = f"pose_capture_{capture_count}.json"
        with open(filename, "w") as f:
            json.dump(pose_data, f, indent=2)

        print(f"Pose landmarks saved to {filename}")
        capture_count += 1

cap.release()
cv2.destroyAllWindows()
