# System Workflow & Architecture

This document details the step-by-step pipeline of how the Driver Monitoring System processes real-world data and converts it into actionable safety alerts.

## The Execution Pipeline

The core loop of the application runs continuously, performing the following steps multiple times per second:

### 1. Frame Acquisition
* `OpenCV` connects to the USB webcam and pulls a raw, high-definition RGB frame.
* The frame is horizontally flipped to act as a mirror, ensuring natural visual feedback for the user on the screen.

### 2. AI Inference (The Black Box)
* The raw frame is passed directly into the custom `final_yr_drowsiness_model.tflite` model via the Ultralytics API.
* The model evaluates the frame using a **Confidence Threshold of 45% (`conf=0.45`)**. This mathematically filters out "ghost boxes" or hallucinations (e.g., preventing the AI from mistaking a shadow for a phone).
* The AI returns a list of bounding boxes and class IDs for everything it recognizes in the frame.

### 3. Classification Mapping
* The system parses the AI's output and maps the specific class string to boolean flags. 
* For example, if the AI detects the string `"eyes_closed"`, the system triggers the internal `is_drowsy = True` flag for that specific frame.

### 4. Stability Thresholds (Counter Logic)
A single frame of a closed eye does not mean the driver is asleep; it could just be a natural blink. To prevent erratic false alarms, the system utilizes engineered "Stability Counters":
* **Drowsiness (`eyes_closed`)**: Must be detected for **5 consecutive frames** (~0.5 seconds).
* **Yawning (`yawn`)**: Must be detected for **8 consecutive frames**. This prevents the alarm from triggering if the driver opens their mouth to speak or sing.
* **Distraction (`distracted`)**: Must be detected for **5 consecutive frames**. This allows the driver to quickly check their mirrors without penalty.
* **Phone & Smoking**: Hard objects. Set to **3 and 4 consecutive frames** respectively to ensure the object is firmly in hand.

If a behavior stops being detected before reaching its threshold, the counter degrades (subtracts by 1), preventing lingering false positives.

### 5. Audio-Visual Alert Trigger
* If any counter breaches its specific threshold, two actions occur simultaneously:
  1. A bright red warning text (e.g., `"DROWSINESS ALERT!"`) is rendered onto the OpenCV video frame.
  2. The `pygame.mixer` executes `play_alarm()`. The `alarm.wav` file is played in an asynchronous background thread, ensuring the video feed never freezes while the sound is active.
* If all counters drop below a safe threshold, the `stop_alarm()` function is called, instantly muting the audio thread.
