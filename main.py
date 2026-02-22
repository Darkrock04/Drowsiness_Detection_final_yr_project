import cv2
import dlib
import imutils
from imutils import face_utils
import time
import pygame
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

from utils.facial_features import eye_aspect_ratio, mouth_aspect_ratio, get_head_pose
from utils.object_detection import detect_objects

# -------- Configuration / Thresholds --------
EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 20

MOUTH_AR_THRESH = 0.79

HEAD_TILT_THRESH = 20.0  # Degrees
HEAD_TILT_CONSEC_FRAMES = 15

# -------- Initialize Pygame for Alarm --------
pygame.mixer.init()
def play_alarm():
    if not pygame.mixer.music.get_busy():
        try:
            alarm_path = os.path.join(BASE_DIR, 'alarm.wav')
            pygame.mixer.music.load(alarm_path)
            pygame.mixer.music.play()
        except Exception as e:
            print("Could not play alarm:", e)

def stop_alarm():
    if pygame.mixer.music.get_busy():
        pygame.mixer.music.stop()

# -------- Load Dlib's face detector and shape predictor --------
print("[INFO] Loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor_path = os.path.join(BASE_DIR, "shape_predictor_68_face_landmarks.dat")
if not os.path.exists(predictor_path):
    print(f"[ERROR] Could not find {predictor_path}. Ensure it is downloaded.")
predictor = dlib.shape_predictor(predictor_path)

# Extract eye and mouth indices from face_utils
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mStart, mEnd) = (49, 68)

# Historical Counters
ear_counter = 0
tilt_counter = 0

# -------- Session Monitoring Statistics --------
session_start_time = time.time()
metrics = {
    "drowsiness_alerts": 0,
    "yawning_alerts": 0,
    "distraction_alerts": 0,
    "phone_usage_alerts": 0
}

# Cooldown timers to prevent logging the exact same event 100 times a second
last_drowsy_log = 0
last_yawn_log = 0
last_distract_log = 0
last_phone_log = 0
COOLDOWN = 5.0  # seconds between logging the same type of alert

print("[INFO] Starting video stream for driving session (External Camera)...")
# Index 1 is typically the first external USB webcam on Windows. 
cap = cv2.VideoCapture(1)
time.sleep(2.0)

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("[WARNING] Empty frame received from webcam. Retrying...")
        time.sleep(0.1)
        continue
        
    # Resize and convert to grayscale for dlib
    frame = imutils.resize(frame, width=800)
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    except Exception as e:
        print(f"[ERROR] Could not convert frame to grayscale: {e}")
        continue
    size = gray.shape

    # --- Feature 1: Object Detection (Phone/Seatbelt/Passenger) ---
    # Pass a precise copy of the frame to YOLO so it doesn't corrupt the numpy array memory layout for dlib
    person_detected, phone_detected, frame_yolo = detect_objects(frame.copy())
    
    # We will use the YOLO annotated frame as our base for drawing facial landmarks later
    display_frame = frame_yolo 

    # --- Detect faces ---
    # Note: detector requires a clean 8-bit unsigned integer array (which our grayscale conversion provides)
    rects = detector(gray, 0)
    
    alert_triggered = False

    if len(rects) > 0:
        cv2.putText(display_frame, "Face Detected", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        for rect in rects:
            # 1. Prediction and facial landmarks
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            # 2. Extract eyes
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0
            
            # Convex hull for eyes visualization
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(display_frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(display_frame, [rightEyeHull], -1, (0, 255, 0), 1)

            # --- Drowsiness Check ---
            if ear < EYE_AR_THRESH:
                ear_counter += 1
                if ear_counter >= EYE_AR_CONSEC_FRAMES:
                    cv2.putText(display_frame, "DROWSINESS ALERT!", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    alert_triggered = True
                    if time.time() - last_drowsy_log > COOLDOWN:
                        metrics["drowsiness_alerts"] += 1
                        last_drowsy_log = time.time()
            else:
                ear_counter = 0

            # 3. Extract Mouth 
            mouth = shape[mStart:mEnd]
            mar = mouth_aspect_ratio(mouth)
            mouthHull = cv2.convexHull(mouth)
            cv2.drawContours(display_frame, [mouthHull], -1, (0, 255, 0), 1)
            
            # --- Yawn Check ---
            if mar > MOUTH_AR_THRESH:
                cv2.putText(display_frame, "YAWNING ALERT!", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                alert_triggered = True
                if time.time() - last_yawn_log > COOLDOWN:
                        metrics["yawning_alerts"] += 1
                        last_yawn_log = time.time()

            # 4. Head Pose Estimation (Distraction)
            image_points = np.array([
                shape[33],  # Nose tip 
                shape[8],   # Chin 
                shape[36],  # Left eye left corner 
                shape[45],  # Right eye right corne 
                shape[48],  # Left Mouth corner 
                shape[54]   # Right mouth corner 
            ], dtype="double")
            
            # Draw projection points
            for p in image_points:
                cv2.circle(display_frame, (int(p[0]), int(p[1])), 3, (0, 0, 255), -1)
                
            head_tilt_degree, sp, ep, ep_alt = get_head_pose(size, image_points, frame.shape[0])
            cv2.line(display_frame, sp, ep, (255, 0, 0), 2)
            cv2.line(display_frame, sp, ep_alt, (0, 0, 255), 2)

            # --- Distraction Check by Head Pose ---
            if head_tilt_degree > HEAD_TILT_THRESH:
                tilt_counter += 1
                if tilt_counter >= HEAD_TILT_CONSEC_FRAMES:
                    cv2.putText(display_frame, "DISTRACTION ALERT! (Head Tilt)", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    alert_triggered = True
                    if time.time() - last_distract_log > COOLDOWN:
                        metrics["distraction_alerts"] += 1
                        last_distract_log = time.time()
            else:
                tilt_counter = 0

            # --- Text Overlays (Stats) ---
            cv2.putText(display_frame, f"EAR: {ear:.2f}", (600, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(display_frame, f"MAR: {mar:.2f}", (600, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            # handle array from degree calc
            tilt_val = head_tilt_degree[0] if isinstance(head_tilt_degree, (list, np.ndarray)) else head_tilt_degree
            cv2.putText(display_frame, f"TILT: {tilt_val:.2f}", (600, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    # --- Secondary Activities Check ---
    if phone_detected:
        cv2.putText(display_frame, "PHONE USAGE ALERT!", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        alert_triggered = True
        if time.time() - last_phone_log > COOLDOWN:
            metrics["phone_usage_alerts"] += 1
            last_phone_log = time.time()

    # --- Seatbelt Status ---
    # Placeholder warning for missing custom seatbelt YOLO
    cv2.putText(display_frame, "Seatbelt: Not Trained (YOLOv8 Default)", (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # --- Trigger Audio Alarm ---
    if alert_triggered:
        play_alarm()
    else:
        stop_alarm()

    cv2.imshow("Continuous Driver Session Monitoring", display_frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# --- End of Session Reporting ---
session_end_time = time.time()
session_duration = session_end_time - session_start_time

print("\n" + "="*40)
print(" DRIVING SESSION COMPLETED ")
print("="*40)
print(f"Total Session Duration: {session_duration:.2f} seconds")
print("--- Infraction Summary ---")
print(f"Drowsiness Instances: {metrics['drowsiness_alerts']}")
print(f"Yawning Instances:    {metrics['yawning_alerts']}")
print(f"Distraction Events:   {metrics['distraction_alerts']}")
print(f"Phone Usage Events:   {metrics['phone_usage_alerts']}")
print("========================================")

cv2.destroyAllWindows()
cap.release()
