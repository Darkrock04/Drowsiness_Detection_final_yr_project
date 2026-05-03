import cv2
import pygame
from ultralytics import YOLO
import time
import os

# 1. Initialize Pygame for Audio
pygame.mixer.init()
pygame.mixer.music.load("alarm.wav")

def play_alarm():
    if not pygame.mixer.music.get_busy():
        pygame.mixer.music.play()

def stop_alarm():
    if pygame.mixer.music.get_busy():
        pygame.mixer.music.stop()

# 2. Load Your Custom Single Lite Model
print("[INFO] Loading Custom TFLite Model...")
model_path = "final_yr_drowsiness_model.tflite"

if not os.path.exists(model_path):
    print(f"[ERROR] Could not find {model_path}!")
    exit()

# Ultralytics seamlessly handles .tflite files!
model = YOLO(model_path)

# 3. Start Video Stream
cap = cv2.VideoCapture(1)
time.sleep(2.0)

# Alert Counters (Prevents jittery/false alarms)
drowsy_counter = 0
yawn_counter = 0
distracted_counter = 0
phone_counter = 0
smoke_counter = 0

# How many consecutive frames an action must be seen to trigger the alarm
# Lowered to 5 because your dataset struggles to hold confident detections for long periods
ALARM_THRESH = 5 

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    frame = cv2.flip(frame, 1)
    
    # 4. RUN THE SINGLE BLACK BOX MODEL
    # We pass the frame in, and YOLO finds ALL behaviors instantly.
    # CONFIDENCE IS SET TO 0.45 to prevent "ghost" boxes and hallucinations!
    results = model(frame, verbose=False, conf=0.45)
    
    # Flags for the current frame
    is_drowsy = False
    is_yawning = False
    is_distracted = False
    is_phone = False
    is_smoking = False
    
    # Parse the YOLO outputs
    if len(results[0].boxes) > 0:
        for box in results[0].boxes:
            class_id = int(box.cls[0])
            class_name = results[0].names[class_id].lower()
            
            # Map the detection to our alert flags based on the 52k dataset!
            if class_name == "eyes_closed":
                is_drowsy = True
            elif class_name == "yawn":
                is_yawning = True
            elif class_name == "distracted":
                is_distracted = True
            elif class_name == "phone":
                is_phone = True
            elif class_name == "smoking":
                is_smoking = True
                
    # --- DROWSINESS LOGIC (STABLE) ---
    if is_drowsy:
        drowsy_counter += 1
        if drowsy_counter >= 5: # 5 frames prevents triggering on a normal eye blink!
            play_alarm()
            cv2.putText(frame, "DROWSINESS ALERT!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        drowsy_counter = max(0, drowsy_counter - 1)
        
    # --- YAWNING LOGIC (STABLE) ---
    if is_yawning:
        yawn_counter += 1
        if yawn_counter >= 8: # 8 frames prevents triggering from just talking
            play_alarm()
            cv2.putText(frame, "YAWNING ALERT!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        yawn_counter = max(0, yawn_counter - 1)
        
    # --- DISTRACTION LOGIC (STABLE) ---
    if is_distracted:
        distracted_counter += 1
        if distracted_counter >= 5: # 5 frames prevents triggering from a quick mirror glance
            play_alarm()
            cv2.putText(frame, "DISTRACTION ALERT!", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        distracted_counter = max(0, distracted_counter - 1)
        
    # --- PHONE LOGIC (STABLE) ---
    if is_phone:
        phone_counter += 1
        if phone_counter >= 3: # 3 frames
            play_alarm()
            cv2.putText(frame, "PHONE USAGE ALERT!", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        phone_counter = max(0, phone_counter - 1)
        
    # --- SMOKING LOGIC (STABLE) ---
    if is_smoking:
        smoke_counter += 1
        if smoke_counter >= 4: # 4 frames
            play_alarm()
            cv2.putText(frame, "SMOKING ALERT!", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        smoke_counter = max(0, smoke_counter - 1)
        
    # --- STOP ALARM IF BEHAVING ---
    if not any([is_drowsy, is_yawning, is_distracted, is_phone, is_smoking]) and max(drowsy_counter, yawn_counter, distracted_counter, phone_counter, smoke_counter) < 3:
        stop_alarm()
        
    # Automatically draw the detection boxes from the TFLite model
    annotated_frame = results[0].plot()
        
    cv2.imshow("Final Year Project - Single Lite Model", annotated_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
