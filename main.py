import mediapipe as mp 
import cv2 

# Using the " Creating Your Own AI Fitness Trainer: Analyzing Squats with MediaPipe"
# Youtube Tutorial: https://www.youtube.com/watch?v=Ae3SPjsXETc
# open virtual environment (source venv/bin/activate)
# install opencv and mediapipe 
# pip install opencv-python
# pip install mediapipe==0.10.14
#  

pose = mp.solutions.pose.Pose(
    static_image_mode           = True, 
    model_complexity          = 1,
    smooth_landmarks         = True,
    min_detection_confidence = 0.5,
    min_tracking_confidence  = 0.5
)

drawing = mp.solutions.drawing_utils 

img = cv2.imread("human-pose.png", 1) # image to read in 
imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # convert to RGB

result = pose.process(imgRGB) # process the image

if result.pose_landmarks: 
    drawing.draw_landmarks(
        image = img, 
        landmark_list = result.pose_landmarks,
        connections = mp.solutions.pose.POSE_CONNECTIONS, 
        landmark_drawing_spec = drawing.DrawingSpec (
            color = (255, 255, 255), 
            thickness = 7, 
            circle_radius = 4
        ), 
        connection_drawing_spec = drawing.DrawingSpec (
            color = (0, 0, 255), 
            thickness = 11, 
            circle_radius = 3
        )
    )

print(result.pose_landmarks) 
cv2.imwrite("Pose.png", img)