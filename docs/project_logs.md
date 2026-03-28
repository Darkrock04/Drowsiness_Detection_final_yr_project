# Project Development Logs

This document serves as a comprehensive log detailing the step-by-step development process, from conceptualization to deployment, for the Final Year Project: Driver Monitoring System. It highlights technical challenges faced and how they were resolved.

## Phase 1: Environment Setup and Foundation

**Objective:** Establish the foundational Python environment capable of real-time computer vision processing.

**Challenges Tackled:**
- **Problem:** Installing `dlib` on Windows natively is notoriously tricky because it requires CMake and dedicated C++ compilation tools, often failing on Python versions > 3.10.
- **Resolution:** Dedicated the environment strictly to **Python 3.10** and ensured Visual Studio Build Tools were installed. Used the pre-compiled `shape_predictor_68_face_landmarks.dat` from the official repository to avoid training an extensive model locally.

## Phase 2: Core Video Capture and Face Detection

**Objective:** Access the physical webcam and pass valid frames into the face detector in real-time.

**Challenges Tackled:**
- **Problem:** `cv2.VideoCapture(0)` was grabbing the laptop's internal, tightly integrated webcam, which offered a poor angle for a driver monitoring system simulation.
- **Resolution:** Swapped to `cv2.VideoCapture(1)` and physically set up an external USB webcam on the side to simulate a realistic dashboard view.
- **Problem:** Dlib's facial landmark detector was dropping frame rates drastically when fed 1080p RGB frames.
- **Resolution:** Implemented `imutils.resize` to cap width at 800px and converted frames to `cv2.COLOR_BGR2GRAY`. Dlib's standard HOG face detector runs highly optimized on single-channel 8-bit unsigned integer arrays.

## Phase 3: Mathematical Ratios for Driver State (EAR / MAR)

**Objective:** Accurately classify "Drowsiness" and "Yawning" states.

**Challenges Tackled:**
- **Problem:** Initially, simply checking if `EAR < 0.25` caused the alarm to trigger every time the driver naturally blinked.
- **Resolution:** Introduced the `EYE_AR_CONSEC_FRAMES = 20` counter. The system now requires the eyes to remain below the threshold for 20 continuous frames to certify it's genuine drowsiness, not a blink. 
- **Problem:** Standard 68-point trackers count the whole mouth structure, making talking look like yawning.
- **Resolution:** Extracted specifically the **inner lip indices** (points 60 to 68) for the Mouth Aspect Ratio (MAR) to ensure only genuine mouth opening (yawning) breaks the threshold (> 0.65).

## Phase 4: Head Pose Estimation (Distraction)

**Objective:** Determine if the driver is looking away from the road using a 2D camera.

**Challenges Tackled:**
- **Problem:** Standard webcams have no depth perception. Mapping 3D angles required complex intrinsic calibration.
- **Resolution:** Leveraged OpenCV's `cv2.solvePnP` function. By passing 6 reference 2D pixels (nose tip, chin, eye corners, mouth corners) against a generic 3D human head model, the projection yielded Euler angles. 
- **Problem:** False positives occurred because the external webcam was mounted off-center.
- **Resolution:** Tweaked the `HEAD_TILT_THRESH` from a default of 20° to 30.0° to accommodate the camera angle discrepancy and prevent false triggers.

## Phase 5: Secondary Object Detection (YOLOv8)

**Objective:** Detect external dangerous behaviors like cell phone usage while driving.

**Challenges Tackled:**
- **Problem:** The system crashed frequently when integrating YOLOv8 with dlib. YOLO inference internally modified the numpy array's memory layout, making the array "non-continuous." Dlib absolutely refuses to process non-continuous arrays.
- **Resolution:** Implemented a hard memory copy before inference: `detect_objects(frame.copy())`. This allows YOLO to annotate the visualization frame while Dlib safely processes the unmodified standard grayscale frame parallelly.

## Phase 6: Threading User Experience and Alarms

**Objective:** Provide audible feedback and session statistics without halting the video processing thread.

**Challenges Tackled:**
- **Problem:** Firing an alarm using standard `os.system` or basic libraries paused the execution thread until the audio finished. During this pause, the driver could do anything unmonitored.
- **Resolution:** Substituted with the `pygame.mixer` module. This allowed the `alarm.wav` to play asynchronously in a background thread.
- **Problem:** The console logger was writing "DROWSY ALERT" 30 times a second, filling up memory and terminal buffers.
- **Resolution:** Implemented a global `COOLDOWN = 5.0` timer logic. If an infraction is logged and metrics are updated, the system will not register the exact same infraction type for another 5 seconds, giving the driver time to react. 

## Final Deployment
The complete sequence reliably runs the YOLO object detection, PyGame threading, Dlib HOG face boundaries, and math-driven aspect ratio thresholds simultaneously on a CPU, ending with a detailed session metric breakdown upon `q` exit.
