import cv2
import mediapipe as mp
import json
import numpy as np

# -----------------------------
# Helpers
# -----------------------------
def lm_xy(lm):
    return np.array([lm.x, lm.y], dtype=np.float32)

def angle_3pts(a, b, c):
    """Angle at point b given points a-b-c (2D)."""
    ba = a - b
    bc = c - b
    denom = (np.linalg.norm(ba) * np.linalg.norm(bc)) + 1e-6
    cosang = float(np.dot(ba, bc) / denom)
    cosang = np.clip(cosang, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))

def point_line_distance_signed(p, a, b):
    """Signed distance from point p to line through a-b (2D).
       Positive = below line, Negative = above line (in image coords)."""
    ab = b - a
    ap = p - a
    cross = ab[0] * ap[1] - ab[1] * ap[0]
    denom = np.linalg.norm(ab) + 1e-6
    return float(cross / denom)

def ema(prev, new, alpha=0.3):
    return new if prev is None else (alpha * new + (1 - alpha) * prev)

# -----------------------------
# MediaPipe init
# -----------------------------
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

cap = cv2.VideoCapture(1)
capture_count = 0

# -----------------------------
# Smoothed values
# -----------------------------
hip_angle_s = None       # shoulder-hip-ankle angle
hip_sag_s = None         # hip deviation from shoulder-ankle line
head_drop_s = None       # head position relative to shoulder line
arm_angle_s = None       # elbow-shoulder-hip angle (arm perpendicular to torso)

# Thresholds
IDEAL_HIP_ANGLE = 175.0
HIP_ANGLE_TOL = 12.0
HIP_SAG_TOL = 0.04
HEAD_DROP_TOL = 0.06
IDEAL_ARM_ANGLE = 90.0   # arm perpendicular to torso
ARM_ANGLE_TOL = 15.0     # degrees tolerance

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    drawing_spec = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=drawing_spec
        )

        lms = results.pose_landmarks.landmark
        R = mp_pose.PoseLandmark

        # Use whichever side is more visible (side view)
        right_vis = (lms[R.RIGHT_SHOULDER].visibility +
                     lms[R.RIGHT_HIP].visibility +
                     lms[R.RIGHT_ANKLE].visibility) / 3.0
        left_vis = (lms[R.LEFT_SHOULDER].visibility +
                    lms[R.LEFT_HIP].visibility +
                    lms[R.LEFT_ANKLE].visibility) / 3.0

        side = "RIGHT" if right_vis >= left_vis else "LEFT"

        if side == "RIGHT":
            shoulder = lm_xy(lms[R.RIGHT_SHOULDER])
            hip = lm_xy(lms[R.RIGHT_HIP])
            ankle = lm_xy(lms[R.RIGHT_ANKLE])
            elbow = lm_xy(lms[R.RIGHT_ELBOW])
            ear = lm_xy(lms[R.RIGHT_EAR])
            key_vis = [lms[R.RIGHT_SHOULDER].visibility, lms[R.RIGHT_HIP].visibility,
                       lms[R.RIGHT_ANKLE].visibility, lms[R.RIGHT_ELBOW].visibility]
        else:
            shoulder = lm_xy(lms[R.LEFT_SHOULDER])
            hip = lm_xy(lms[R.LEFT_HIP])
            ankle = lm_xy(lms[R.LEFT_ANKLE])
            elbow = lm_xy(lms[R.LEFT_ELBOW])
            ear = lm_xy(lms[R.LEFT_EAR])
            key_vis = [lms[R.LEFT_SHOULDER].visibility, lms[R.LEFT_HIP].visibility,
                       lms[R.LEFT_ANKLE].visibility, lms[R.LEFT_ELBOW].visibility]

        ok_conf = min(key_vis) >= 0.4

        if ok_conf:
            # 1. Hip angle: shoulder-hip-ankle (ideal ~175-180 for straight body)
            hip_angle = angle_3pts(shoulder, hip, ankle)

            # 2. Hip sag/pike: signed distance of hip from shoulder-ankle line
            body_len = np.linalg.norm(shoulder - ankle) + 1e-6
            hip_sag = point_line_distance_signed(hip, shoulder, ankle) / body_len

            # 3. Head position: distance of ear from shoulder-hip line extension
            torso_len = np.linalg.norm(shoulder - hip) + 1e-6
            head_drop = point_line_distance_signed(ear, shoulder, hip) / torso_len

            # 4. Arm angle: elbow-shoulder-hip (ideal 90 degrees - arm perpendicular to torso)
            arm_angle = angle_3pts(elbow, shoulder, hip)

            # Smooth
            hip_angle_s = ema(hip_angle_s, hip_angle, alpha=0.35)
            hip_sag_s = ema(hip_sag_s, hip_sag, alpha=0.35)
            head_drop_s = ema(head_drop_s, head_drop, alpha=0.35)
            arm_angle_s = ema(arm_angle_s, arm_angle, alpha=0.35)

            # -----------------------------
            # Scoring & Feedback
            # -----------------------------
            score = 100.0
            cues = []

            # Hip angle scoring
            angle_dev = abs(hip_angle_s - IDEAL_HIP_ANGLE)
            if angle_dev > HIP_ANGLE_TOL:
                penalty = min(30.0, (angle_dev - HIP_ANGLE_TOL) / 20.0 * 30.0)
                score -= penalty
                if hip_angle_s < IDEAL_HIP_ANGLE - HIP_ANGLE_TOL:
                    cues.append("Hips sagging: squeeze glutes")
                else:
                    cues.append("Hips too high: lower them")

            # Hip sag/pike scoring
            if hip_sag_s > HIP_SAG_TOL:
                penalty = min(20.0, (hip_sag_s - HIP_SAG_TOL) / 0.08 * 20.0)
                score -= penalty
                if "sagging" not in str(cues):
                    cues.append("Engage core: hips dropping")
            elif hip_sag_s < -HIP_SAG_TOL:
                penalty = min(20.0, (abs(hip_sag_s) - HIP_SAG_TOL) / 0.08 * 20.0)
                score -= penalty
                if "too high" not in str(cues):
                    cues.append("Hips piking up")

            # Arm angle scoring (should be ~90 degrees)
            arm_dev = abs(arm_angle_s - IDEAL_ARM_ANGLE)
            if arm_dev > ARM_ANGLE_TOL:
                penalty = min(25.0, (arm_dev - ARM_ANGLE_TOL) / 25.0 * 25.0)
                score -= penalty
                if arm_angle_s < IDEAL_ARM_ANGLE - ARM_ANGLE_TOL:
                    cues.append("Arms too far back: align under shoulders")
                else:
                    cues.append("Arms too far forward: align under shoulders")

            # Head position scoring
            if abs(head_drop_s) > HEAD_DROP_TOL:
                penalty = min(15.0, (abs(head_drop_s) - HEAD_DROP_TOL) / 0.1 * 15.0)
                score -= penalty
                if head_drop_s > HEAD_DROP_TOL:
                    cues.append("Head dropping: look slightly forward")
                else:
                    cues.append("Head too high: neutral neck")

            score = max(0.0, min(100.0, score))
            feedback = " | ".join(cues) if cues else "Good form!"

            # Color based on score
            if score >= 80:
                score_color = (0, 255, 0)
                status = "GOOD"
            elif score >= 60:
                score_color = (0, 255, 255)
                status = "OK"
            else:
                score_color = (0, 0, 255)
                status = "FIX FORM"

            # Draw reference lines
            h, w = frame.shape[:2]
            pt_s = (int(shoulder[0] * w), int(shoulder[1] * h))
            pt_a = (int(ankle[0] * w), int(ankle[1] * h))
            pt_h = (int(hip[0] * w), int(hip[1] * h))
            pt_e = (int(elbow[0] * w), int(elbow[1] * h))
            
            # Body line (shoulder to ankle)
            cv2.line(frame, pt_s, pt_a, (255, 255, 0), 2)
            # Arm line (elbow to shoulder)
            arm_color = (0, 255, 0) if arm_dev <= ARM_ANGLE_TOL else (0, 0, 255)
            cv2.line(frame, pt_e, pt_s, arm_color, 2)
            # Hip indicator
            cv2.circle(frame, pt_h, 8, score_color, -1)

            # Overlay
            cv2.putText(frame, f"Side: {side}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Form Score: {score:.0f} - {status}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, score_color, 2)
            cv2.putText(frame, f"Hip Angle: {hip_angle_s:.1f} (ideal: 175)", (10, 95),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"Arm Angle: {arm_angle_s:.1f} (ideal: 90)", (10, 125),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"Hip Sag: {hip_sag_s:+.3f}", (10, 155),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"Head Pos: {head_drop_s:+.3f}", (10, 185),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, feedback[:65], (10, 220),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    cv2.imshow("Plank Form Analyzer", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("c") and results.pose_landmarks:
        pose_data = []
        for lm in results.pose_landmarks.landmark:
            pose_data.append({
                "x": lm.x,
                "y": lm.y,
                "z": lm.z,
                "visibility": lm.visibility
            })
        filename = f"plank_capture_{capture_count}.json"
        with open(filename, "w") as f:
            json.dump(pose_data, f, indent=2)
        print(f"Saved to {filename}")
        capture_count += 1

cap.release()
cv2.destroyAllWindows()