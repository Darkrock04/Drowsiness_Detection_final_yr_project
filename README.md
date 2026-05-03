# Driver Monitoring System (Final Year Project)

This project implements a state-of-the-art, **Unified Edge-AI In-Cabin Monitoring System** designed to promote driver safety. By utilizing a single, highly optimized deep neural network, the system continuously analyzes the driver's face and actions in real-time to detect fatigue, distraction, and unsafe behaviors.

## Project Overview

The core of this system is a custom-trained **YOLOv8s** (Small) object detection model that has been exported to **TensorFlow Lite (`Float16`)** format. This deployment strategy ensures the system runs with extremely high accuracy while maintaining lightweight computational requirements, making it ideal for edge devices (such as a Raspberry Pi or dashboard computer).

The AI acts as a single "Black Box" engine. It consumes raw video frames from an external webcam and outputs precise bounding boxes and class identifiers for 5 distinct driver behaviors simultaneously.

## Core Features & Capabilities

The system tracks 5 critical infractions. To ensure professional stability and prevent false alarms (e.g., from natural blinking), the system enforces "Stability Thresholds" requiring sustained detection over a specific number of consecutive frames.

| Driver Behavior | System Detection | Stability Threshold (Alert Trigger) |
| :--- | :--- | :--- |
| **Drowsiness** | Detects closed eyes / micro-sleeps | 5 consecutive frames |
| **Yawning** | Detects sustained yawning | 8 consecutive frames |
| **Distraction** | Detects the driver looking away | 5 consecutive frames |
| **Phone Usage** | Detects holding/using a mobile device | 3 consecutive frames |
| **Smoking** | Detects a cigarette/smoking gesture | 4 consecutive frames |

---

## Detailed Documentation

The comprehensive technical breakdown of this project is organized within the `docs/` folder:

1. **[System Workflow & Architecture](docs/workflow.md)**: Explains the step-by-step pipeline of how the video frame is captured, processed by the AI, and converted into audio alerts.
2. **[Modules & Technologies Used](docs/modules_used.md)**: Details the specific Python libraries and frameworks that power the system.
3. **[References & Training Data](docs/references.md)**: Contains links to the massive 52,000-image dataset used to train the model, along with details on the Kaggle training environment.

---

## Installation & Setup (Windows)

### Prerequisites
- **Python 3.10**
- An **external USB webcam**
- `alarm.wav` audio file placed in the project root
- `final_yr_drowsiness_model.tflite` placed in the project root

### Environment Setup

```bash
# 1. Navigate to the project folder
cd d:\CODE\Drowsiness_Detection\Drowsiness_Detection\final_year_project

# 2. Create and activate the virtual environment
python -m venv venv
.\venv\Scripts\activate

# 3. Install required dependencies
pip install -r requirements.txt
```

### Running the System
```bash
# Ensure the venv is activated
.\venv\Scripts\activate

# Launch the monitoring system
python main_final.py
```
Press **`q`** on the video window to safely end the monitoring session.
