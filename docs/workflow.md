# System Workflow & Architecture

This document details the complete step-by-step pipeline of how the Driver Monitoring System processes real-world data and converts it into actionable safety alerts.

---

## System Pipeline (Visual Overview)

```
USB Webcam
    │
    ▼
┌─────────────────────────────┐
│  1. Frame Acquisition       │  ← cv2.VideoCapture() captures a raw RGB frame
│     + Horizontal Flip       │  ← cv2.flip() mirrors the image
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  2. Model Inference         │  ← final_yr_drowsiness_model.tflite
│     Confidence ≥ 0.45       │  ← Filters out weak/ghost detections
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  3. Classification Mapping  │  ← Maps class labels to boolean flags
│     eyes_closed → is_drowsy │     (is_drowsy, is_yawning, etc.)
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  4. Stability Counters      │  ← Prevents false alarms from blinks
│     Counter must breach     │     or brief glances
│     threshold to trigger    │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  5. Alert Trigger           │  ← Red text on screen
│     Audio + Visual          │  ← alarm.wav plays via PyGame
└─────────────────────────────┘
```

---

## Detailed Execution Pipeline

The core loop of the application runs continuously, performing the following steps multiple times per second:

### Step 1: Camera Initialization & Frame Acquisition
* On startup, `OpenCV` connects to the USB webcam via `cv2.VideoCapture(1)`.
* A `time.sleep(2.0)` delay is applied to allow the camera hardware to auto-adjust its brightness and exposure settings. Without this, the first few frames would be black or overexposed.
* Inside the main loop, `cap.read()` pulls a raw RGB frame from the webcam.
* The frame is horizontally flipped using `cv2.flip(frame, 1)` to act as a natural mirror, ensuring the driver sees a correct left-right orientation on screen.

### Step 2: Model Inference
* The raw frame is passed directly into our custom-trained model (`final_yr_drowsiness_model.tflite`) via the Ultralytics API.
* The model evaluates the frame using a **Confidence Threshold of 45% (`conf=0.45`)**. This means the model will only report a detection if it is at least 45% confident. This mathematically filters out "ghost boxes" or hallucinations (e.g., preventing the model from mistaking a shadow for a phone or a talking mouth for a yawn).
* The model returns a list of **bounding boxes** (rectangular coordinates around detected objects) and **class IDs** (integer identifiers corresponding to the detected behavior).

### Step 3: Classification Mapping
* The system iterates through every detection returned by the model.
* For each detection, it reads the class name string (e.g., `"eyes_closed"`, `"yawn"`, `"phone"`) and sets the corresponding boolean flag to `True`.
* Example: If the model detects `"eyes_closed"` in the frame, the system sets `is_drowsy = True` for that specific frame.
* The `awake` class is intentionally ignored since it represents the safe, default state.

### Step 4: Stability Thresholds (Counter Logic)
A single frame of a closed eye does not mean the driver is asleep — it could just be a natural blink. To prevent erratic false alarms, the system utilizes engineered **Stability Counters**:

| Behavior | Threshold | Why This Value? |
| :--- | :--- | :--- |
| **Drowsiness** (`eyes_closed`) | 5 consecutive frames (~0.5s) | Ignores natural eye blinks which last ~0.3s |
| **Yawning** (`yawn`) | 8 consecutive frames | Ignores the driver opening their mouth to talk or sing |
| **Distraction** (`distracted`) | 5 consecutive frames | Allows quick mirror/window checks without penalty |
| **Phone Usage** (`phone`) | 3 consecutive frames | Phone is a hard physical object, fewer frames needed |
| **Smoking** (`smoking`) | 4 consecutive frames | Cigarette is a hard physical object |

**How the counters work:**
* **Increment:** If a behavior IS detected in the current frame, its counter increases by 1.
* **Decrement:** If a behavior is NOT detected, its counter decreases by 1 (but never goes below 0). This gradual decay prevents "lingering" false positives from a single noisy frame.
* **Trigger:** The alarm only fires when a counter reaches or exceeds its threshold value.

### Step 5: Audio-Visual Alert Trigger
* **Visual Alert:** If any counter breaches its threshold, a bright red warning text (e.g., `"DROWSINESS ALERT!"`) is rendered directly onto the OpenCV video frame using `cv2.putText()`. Each behavior has its own Y-coordinate position so multiple alerts can display simultaneously without overlapping.
* **Audio Alert:** The `play_alarm()` function checks if the alarm is already playing (`pygame.mixer.music.get_busy()`). If not, it triggers `pygame.mixer.music.play()`. The `alarm.wav` file plays asynchronously in a background thread, ensuring the video feed never freezes.

### Step 6: Alarm Stop (Safe Behavior Recovery)
* If **none** of the 5 behaviors are detected in the current frame AND all counters have dropped below the safe threshold (< 3), the `stop_alarm()` function is called.
* This instantly mutes the audio thread via `pygame.mixer.music.stop()`.
* The system immediately resumes normal monitoring without any residual alert state.

### Step 7: Frame Display & Exit
* The fully annotated frame (with bounding boxes drawn by the model and alert text drawn by OpenCV) is displayed in a window titled `"Driver Monitoring System"` via `cv2.imshow()`.
* The system listens for the `q` key press via `cv2.waitKey(1)`. When pressed, the loop breaks, the camera is released, and all windows are destroyed cleanly.
