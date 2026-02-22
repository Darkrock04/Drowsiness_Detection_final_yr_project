# Driver Monitoring System (Final Year Project)

This project implements a comprehensive in-cabin monitoring system designed to promote driver safety. It aggregates computer vision and deep learning techniques to continuously track the driver's state throughout their entire driving session.

## Documentation Navigation

Technical methodologies and dataset references are in the `docs/` folder:

1. **[Facial Landmarks, Drowsiness, Yawning & Head Pose](docs/facial_features.md)**: Details the `dlib` 300-W implementation and the mathematical ratios (EAR/MAR) used for fatigue detection.
2. **[Object Detection & Occupancy](docs/object_detection.md)**: Covers YOLOv8 usage for phone detection and passenger occupancy.

---

## Features

| Feature | Method | Alert Trigger |
|---|---|---|
| **Drowsiness Detection** | Eye Aspect Ratio (EAR) via dlib | EAR < 0.25 for 20 consecutive frames |
| **Yawning Detection** | Inner Lip Aspect Ratio (MAR) via dlib | MAR > 0.65 for 15 consecutive frames |
| **Distraction Detection** | 3D Head Pose Estimation via `cv2.solvePnP` | Head tilt > 30° for 15 consecutive frames |
| **Phone Usage** | YOLOv8 Nano object detection (COCO class 67) | Phone detected > 40% confidence |
| **Session Monitoring** | Session timer + infraction counters | End-of-session summary printed on exit |

---

## System Pipeline (per frame)

```
Camera Frame
    │
    ├── YOLOv8 ──────────────────► Detect Phone / Person
    │
    ├── Grayscale Conversion
    │
    ├── dlib HOG Face Detector ──► Face Bounding Box
    │
    ├── dlib 68-pt Landmarks ────► Eye / Mouth / Nose Points
    │
    ├── EAR Calculation ─────────► Drowsiness Alert
    ├── MAR Calculation ─────────► Yawning Alert
    └── cv2.solvePnP ────────────► Distraction Alert
```

---

## Installation (Windows)

### Prerequisites
- **Python 3.10** (recommended — dlib-bin does not yet support 3.13 on Windows)
- An **external USB webcam** (connected as Camera Index `1`)
- `shape_predictor_68_face_landmarks.dat` placed in the project root  
  ↳ Download from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
- `alarm.wav` audio file placed in the project root

### Setup

```bash
# 1. Navigate to the project folder
cd d:\CODE\Drowsiness_Detection\Drowsiness_Detection\final_year_project

# 2. Create and activate the virtual environment
python -m venv venv
.\venv\Scripts\activate

# 3. Install all dependencies into the venv
pip install -r requirements.txt
```

### Run
```bash
# Make sure the venv is activated first!
.\venv\Scripts\activate

python main.py
```
Press **`q`** to end the driving session. A summary of all infractions will be printed to the terminal.

---

## End-of-Session Report

When you press `q` to exit, the console will display a structured summary like:

```
========================================
 DRIVING SESSION COMPLETED
========================================
Total Session Duration: 432.67 seconds
--- Infraction Summary ---
Drowsiness Instances: 3
Yawning Instances:    1
Distraction Events:   5
Phone Usage Events:   2
========================================
```

---

## Project Structure

```
final_year_project/
├── main.py                          # Main application entry point
├── requirements.txt                 # Python dependencies
├── .gitignore
├── alarm.wav                        # Audio alarm (not tracked in git)
├── shape_predictor_68_face_landmarks.dat  # dlib model (not tracked in git)
├── utils/
│   ├── facial_features.py           # EAR, MAR, Head Pose functions
│   └── object_detection.py          # YOLOv8 phone/person detection
└── docs/
    ├── facial_features.md           # Facial detection methodology
    └── object_detection.md          # Object detection methodology
```
