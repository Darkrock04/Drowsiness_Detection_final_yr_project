# Driver Monitoring System

This project implements a **Driver Monitoring System** designed to promote driver safety. By utilizing a single, highly optimized custom-trained deep learning model, the system continuously analyzes the driver's face and actions in real-time to detect fatigue, distraction, and unsafe behaviors.

## Project Overview

The core of this system is a **custom-trained deep learning model** (`final_yr_drowsiness_model.tflite`) that we built and trained ourselves on a massive dataset of over 52,000 images. The model has been exported to **TensorFlow Lite (`Float16`)** format, ensuring it runs with extremely high accuracy while maintaining lightweight computational requirements, making it ideal for edge devices (such as a Raspberry Pi or dashboard computer).

The model acts as a single inference engine. It consumes raw video frames from an external webcam and outputs precise bounding boxes and class identifiers for 5 distinct driver behaviors simultaneously.

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

## Model Training Details

We trained our own custom deep learning model from scratch using Transfer Learning on a curated dataset of 52,000+ images.

*   **Training Platform:** Kaggle (T4x2 GPU, 50 Epochs, ~9 hours)
*   **Dataset:** [Driver Drowsiness Dataset (v10) — Roboflow Universe](https://universe.roboflow.com/drowsiness-detection-system/driver-drowsiness-3ldhq/dataset/10)
*   **Model Classes:** `awake`, `distracted`, `eyes_closed`, `phone`, `smoking`, `yawn`
*   **Export Format:** TensorFlow Lite with Float16 Quantization
*   **Model File:** `final_yr_drowsiness_model.tflite`

---

## Detailed Documentation

The comprehensive technical breakdown of this project is organized within the `docs/` folder:

1. **[System Workflow & Architecture](docs/workflow.md)**: Explains the step-by-step pipeline of how the video frame is captured, processed by the model, and converted into audio alerts.
2. **[Modules & Technologies Used](docs/modules_used.md)**: Details the specific Python libraries and frameworks that power the system.
3. **[Literature Survey & Research Gap](docs/literature_survey.md)**: Explains why our approach is different from existing research and what gap it fills.
4. **[References & Training Data](docs/references.md)**: Contains links to the dataset, training platform, and tools used.

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
