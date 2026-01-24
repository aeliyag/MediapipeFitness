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

def point_line_distance(p, a, b):
    """Distance from point p to line through a-b (2D)."""
    ab = b - a
    ap = p - a
    denom = np.linalg.norm(ab) + 1e-6
    cross = abs(ab[0]*ap[1] - ab[1]*ap[0])  # 2D cross magnitude
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

cap = cv2.VideoCapture(0)

capture_count = 0

# -----------------------------
# Curl logic state
# -----------------------------
state = "down"
reps = 0
last_score = None
last_feedback = ""

# smoothed angles
elbow_s = None
shoulder_s = None
torso_s = None
drift_s = None

# per-rep trackers
rep_elbow_min = None
rep_elbow_max = None
rep_shoulder_min = None
rep_shoulder_max = None
rep_torso_min = None
rep_torso_max = None
rep_drift_min = None
rep_drift_max = None

# thresholds (tune these)
UP_TH = 65      # elbow angle < this => "up"
DOWN_TH = 155   # elbow angle > this => "down"

SWING_OK = 10.0       # degrees torso change allowed
SHOULDER_OK = 15.0    # degrees shoulder angle change allowed
DRIFT_OK = 0.12       # normalized elbow drift change allowed (fraction of torso length)

def reset_rep_trackers():
    global rep_elbow_min, rep_elbow_max, rep_shoulder_min, rep_shoulder_max
    global rep_torso_min, rep_torso_max, rep_drift_min, rep_drift_max
    rep_elbow_min = rep_elbow_max = None
    rep_shoulder_min = rep_shoulder_max = None
    rep_torso_min = rep_torso_max = None
    rep_drift_min = rep_drift_max = None

def track_minmax(val, vmin, vmax):
    if vmin is None or val < vmin: vmin = val
    if vmax is None or val > vmax: vmax = val
    return vmin, vmax

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    drawing_spec = mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=drawing_spec
        )

        lms = results.pose_landmarks.landmark

        # Decide which arm to use based on visibility
        R = mp_pose.PoseLandmark
        right_vis = (lms[R.RIGHT_SHOULDER].visibility +
                     lms[R.RIGHT_ELBOW].visibility +
                     lms[R.RIGHT_WRIST].visibility) / 3.0
        left_vis  = (lms[R.LEFT_SHOULDER].visibility +
                     lms[R.LEFT_ELBOW].visibility +
                     lms[R.LEFT_WRIST].visibility) / 3.0

        side = "RIGHT" if right_vis >= left_vis else "LEFT"

        if side == "RIGHT":
            S = lm_xy(lms[R.RIGHT_SHOULDER])
            E = lm_xy(lms[R.RIGHT_ELBOW])
            W = lm_xy(lms[R.RIGHT_WRIST])
            H = lm_xy(lms[R.RIGHT_HIP])
        else:
            S = lm_xy(lms[R.LEFT_SHOULDER])
            E = lm_xy(lms[R.LEFT_ELBOW])
            W = lm_xy(lms[R.LEFT_WRIST])
            H = lm_xy(lms[R.LEFT_HIP])

        # Require decent confidence on key joints
        ok_conf = True
        key_vis = []
        if side == "RIGHT":
            key_vis = [lms[R.RIGHT_SHOULDER].visibility, lms[R.RIGHT_ELBOW].visibility,
                       lms[R.RIGHT_WRIST].visibility, lms[R.RIGHT_HIP].visibility]
        else:
            key_vis = [lms[R.LEFT_SHOULDER].visibility, lms[R.LEFT_ELBOW].visibility,
                       lms[R.LEFT_WRIST].visibility, lms[R.LEFT_HIP].visibility]
        if min(key_vis) < 0.5:
            ok_conf = False

        if ok_conf:
            # Elbow angle
            elbow_angle = angle_3pts(S, E, W)

            # Shoulder "help" angle: angle at shoulder using hip-shoulder-elbow
            shoulder_angle = angle_3pts(H, S, E)

            # Torso lean angle vs vertical using hip->shoulder vector
            torso_vec = S - H
            vertical = np.array([0.0, -1.0], dtype=np.float32)
            torso_angle = angle_3pts(H + vertical, H, S)  # angle between vertical and torso

            # Elbow drift from torso line (S-H), normalized by torso length
            torso_len = float(np.linalg.norm(S - H)) + 1e-6
            drift = point_line_distance(E, S, H) / torso_len  # normalized

            # Smooth values (cuts jitter)
            elbow_s = ema(elbow_s, elbow_angle, alpha=0.35)
            shoulder_s = ema(shoulder_s, shoulder_angle, alpha=0.35)
            torso_s = ema(torso_s, torso_angle, alpha=0.35)
            drift_s = ema(drift_s, drift, alpha=0.35)

            # Track min/max during rep (for scoring)
            if state == "up" or state == "down":
                rep_elbow_min, rep_elbow_max = track_minmax(elbow_s, rep_elbow_min, rep_elbow_max)
                rep_shoulder_min, rep_shoulder_max = track_minmax(shoulder_s, rep_shoulder_min, rep_shoulder_max)
                rep_torso_min, rep_torso_max = track_minmax(torso_s, rep_torso_min, rep_torso_max)
                rep_drift_min, rep_drift_max = track_minmax(drift_s, rep_drift_min, rep_drift_max)

            # Rep state machine
            if state == "down" and elbow_s < UP_TH:
                state = "up"
                reset_rep_trackers()
                # start tracking for this rep
                rep_elbow_min, rep_elbow_max = track_minmax(elbow_s, rep_elbow_min, rep_elbow_max)
                rep_shoulder_min, rep_shoulder_max = track_minmax(shoulder_s, rep_shoulder_min, rep_shoulder_max)
                rep_torso_min, rep_torso_max = track_minmax(torso_s, rep_torso_min, rep_torso_max)
                rep_drift_min, rep_drift_max = track_minmax(drift_s, rep_drift_min, rep_drift_max)

            elif state == "up" and elbow_s > DOWN_TH:
                # Rep completed
                reps += 1
                state = "down"

                swing_delta = (rep_torso_max - rep_torso_min) if (rep_torso_max is not None) else 0.0
                shoulder_delta = (rep_shoulder_max - rep_shoulder_min) if (rep_shoulder_max is not None) else 0.0
                drift_delta = (rep_drift_max - rep_drift_min) if (rep_drift_max is not None) else 0.0
                elbow_min = rep_elbow_min if rep_elbow_min is not None else elbow_s
                elbow_max = rep_elbow_max if rep_elbow_max is not None else elbow_s

                # Score penalties
                score = 100.0

                # Swing penalty up to 40
                if swing_delta > SWING_OK:
                    score -= min(40.0, (swing_delta - SWING_OK) / 10.0 * 40.0)

                # Shoulder help penalty up to 30
                if shoulder_delta > SHOULDER_OK:
                    score -= min(30.0, (shoulder_delta - SHOULDER_OK) / 15.0 * 30.0)

                # Elbow drift penalty up to 30
                if drift_delta > DRIFT_OK:
                    score -= min(30.0, (drift_delta - DRIFT_OK) / 0.15 * 30.0)

                # ROM penalties (optional)
                if elbow_min > 75:   # not curling high enough
                    score -= 10.0
                if elbow_max < 150:  # not extending enough
                    score -= 10.0

                score = max(0.0, min(100.0, score))
                last_score = score

                # Feedback
                cues = []
                if swing_delta > 12:
                    cues.append("Stop swinging: brace core")
                if shoulder_delta > 20:
                    cues.append("Shoulder helping: keep elbow pinned")
                if drift_delta > 0.15:
                    cues.append("Elbow drifting: keep it by your side")
                if elbow_min > 75:
                    cues.append("Curl higher (full ROM)")
                if elbow_max < 150:
                    cues.append("Extend more at bottom")

                last_feedback = " | ".join(cues) if cues else "Nice rep."

            # Overlay
            cv2.putText(frame, f"Arm: {side}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.putText(frame, f"Reps: {reps}  State: {state}", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.putText(frame, f"Elbow: {elbow_s:.1f}", (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.putText(frame, f"Torso d: {((rep_torso_max-rep_torso_min) if rep_torso_max else 0):.1f}", (10, 115),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            cv2.putText(frame, f"Elbow drift d: {((rep_drift_max-rep_drift_min) if rep_drift_max else 0):.3f}", (10, 145),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

            if last_score is not None:
                cv2.putText(frame, f"Last rep score: {last_score:.0f}", (10, 175),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                cv2.putText(frame, last_feedback[:60], (10, 205),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    # Show
    cv2.imshow("Pose", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break
    elif key == ord("c") and results.pose_landmarks:
        # Save raw landmarks to JSON (your original behavior)
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
