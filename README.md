# Driver Monitoring System (Final Year Project)

This project implements a comprehensive in-cabin monitoring system designed to promote driver safety. It aggregates various computer vision and deep learning techniques to track the driver's state and detect secondary activities that could lead to accidents.

## Documentation Navigation

The technical methodologies and dataset details have been split into specific documentation files for easy reading:

1. **[Facial Landmarks, Drowsiness, Yawning & Head Pose](docs/facial_features.md)**: Details the `dlib` 300-W implementation and the mathematical ratios (EAR/MAR) used for fatigue detection.
2. **[Object Detection & Occupancy](docs/object_detection.md)**: Details the transition from classical Image Classification (Fisher Vectors) to modern Deep Learning (`YOLOv8`) for detecting cell phones and passengers.

## Features & Methodologies

1. **Continuous Driving Session Monitoring**
   - **Methodology**: The system actively tracks the duration of the entire driving session from start to finish. When the trip concludes, it prints a post-session summary report detailing total driving time and all logged safety infractions.

2. **Drowsiness Detection**
   - **Methodology**: Calculates the Eye Aspect Ratio (EAR) using 68-point facial landmarks. The EAR algorithm calculates the distances between the vertical and horizontal eye landmarks. If the EAR falls below a predefined threshold for a continuous number of frames, an alert is triggered and logged to the session metrics.
   - **Implementation**: Utilizes `dlib`'s pre-trained HOG face detector and the `shape_predictor_68_face_landmarks.dat` model to locate the eyes.

3. **Yawning Detection**
   - **Methodology**: Calculates the Mouth Aspect Ratio (MAR). Similar to EAR, it computes the distances between specific upper/lower lip landmarks. When the mouth opens beyond a threshold indicating a yawn, an alert is triggered and logged.
   - **Implementation**: Also relies on the `dlib` 68-point landmark predictor, specifically targeting the jaw and lip contour points.

4. **Distraction Detection (Head Pose Estimation)**
   - **Methodology**: 3D Head pose estimation is performed by solving the Perspective-n-Point (PnP) problem (`cv2.solvePnP` function). This projects 2D facial landmarks (nose, chin, eyes, mouth corners) into a 3D space to calculate the Euler angles of the head. Excessive head tilt away from the forward-facing position triggers a distraction alert and is logged to the session.
   - **Implementation**: Uses standard 3D anthropometric face models mapped to the 2D `dlib` landmarks extracted from the frame.

5. **Secondary Activity Recognition (Phone & Objects)**
   - **Methodology**: Deep learning object detection runs concurrently with facial tracking to detect if the driver is holding a cell phone or other distracting devices. 
   - **Implementation**: Utilizes the modern `Ultralytics YOLOv8` architecture. The pre-trained Nano model (`yolov8n.pt`) is used to quickly identify "person" and "cell phone" classes from the COCO dataset in real-time. Usage is logged during the session.

6. **Passenger Occupancy & Seatbelt Classification**
   - **Methodology**: The `YOLOv8` model also identifies "person" objects within the vehicle cabin. By analyzing their position and the presence of a seatbelt (if detectable by the model), the system can infer passenger occupancy and seatbelt usage.
   - **Implementation**: Leverages the same `YOLOv8` model for object detection.

## System Architecture

The pipeline is driven directly from the webcam feed via OpenCV. Below is the continuous data flow for each frame:
1. Frame is captured via `cv2.VideoCapture`.
2. Frame is passed to `YOLOv8` for rapid object detection (Phone/Person).
3. Frame is converted to Grayscale for classical feature extraction.
4. `dlib` HOG Face Detector locates the face bounding box.
5. `dlib` 68-point shape predictor extracts the 68 specific facial landmarks.
6. The landmarks are geometrically split into Left Eye, Right Eye, and Mouth coordinates.
7. EAR, MAR, and Head Pose are computed simultaneously using Euclidean distances and PnP projection.
8. If any threshold is breached, a visual alert is rendered on the frame, and `pygame` plays an asynchronous audio alarm (`alarm.wav`).

## Installation & Usage

1. **Navigate to the absolute project folder:**
   ```bash
   cd d:\CODE\Drowsiness_Detection\Drowsiness_Detection\final_year_project
   ```
2. **Setup virtual environment (Optional but Recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On macOS/Linux
   venv\Scripts\activate     # On Windows
   ```
3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Ensure the dlib model is present:**
   Make sure `shape_predictor_68_face_landmarks.dat` is downloaded and placed in the root of the project folder.
   
5. **Run the Application:**
   ```bash
   python main.py
   ```
   **Press `q` to terminate the video stream and exit the application.**
