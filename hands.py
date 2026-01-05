import cv2
import mediapipe as mp
import json

#same as live_camera.py but for hands
#Pose -> Hands

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
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

    # Flip frame for a mirror view (optional)
    frame = cv2.flip(frame, 1)

    # Convert BGR to RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame
    results = hands.process(rgb)

    # Draw landmarks if detected
    drawing_spec = mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                landmark_drawing_spec=drawing_spec
            )

    # Show frame
    cv2.imshow("Hands", frame)

    # Wait for key press
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break
    elif key == ord("c") and results.multi_hand_landmarks:
        # Save landmarks to JSON
        hands_data = []
        for hand_landmarks in results.multi_hand_landmarks:
            hand_list = []
            for lm in hand_landmarks.landmark:
                hand_list.append({
                    "x": lm.x,
                    "y": lm.y,
                    "z": lm.z
                })
            hands_data.append(hand_list)

        filename = f"hands_capture_{capture_count}.json"
        with open(filename, "w") as f:
            json.dump(hands_data, f, indent=2)

        print(f"Hand landmarks saved to {filename}")
        capture_count += 1

cap.release()
cv2.destroyAllWindows()
