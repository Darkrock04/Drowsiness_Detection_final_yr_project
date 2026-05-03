# Project References

This document details the external datasets, frameworks, and technologies used to build the Unified Edge-AI Driver Monitoring System.

### 1. Driver Drowsiness Dataset (DMS)
* **Source:** Roboflow Universe
* **Link:** [https://universe.roboflow.com/drowsiness-detection-system/driver-drowsiness-3ldhq](https://universe.roboflow.com/drowsiness-detection-system/driver-drowsiness-3ldhq)
* **Description:** 
  * This is the massive 52,000+ image dataset used to train the final custom YOLOv8 model.
  * It provides explicitly balanced classes for the core behaviors required for this project: `awake`, `distracted`, `eyes_closed`, `phone`, `smoking`, and `yawn`.
  * The dataset utilizes advanced Mosaic Augmentation to ensure the AI learns to detect features across diverse lighting conditions, camera angles, and ethnicities.

### 2. Ultralytics YOLOv8 Framework
* **Source:** [https://github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)
* **Description:** 
  * YOLO (You Only Look Once) version 8 was used as the foundational architecture (`yolov8s.pt`) due to its industry-leading speed and accuracy for real-time object detection.
  * The framework was utilized to execute Transfer Learning, adapting the general-purpose YOLO weights strictly to the custom Driver Monitoring dataset.

### 3. TensorFlow Lite & XNNPACK
* **Source:** [https://www.tensorflow.org/lite](https://www.tensorflow.org/lite)
* **Description:**
  * TensorFlow Lite was selected as the deployment framework to ensure the model could run efficiently on edge-computing devices (such as a Raspberry Pi).
  * The model was exported using **Float16 Quantization** to reduce memory footprint by 50% while preserving mathematical precision.
  * The XNNPACK CPU delegate is leveraged at runtime to provide massive hardware acceleration without requiring a discrete GPU.

### 4. OpenCV & PyGame
* **Description:**
  * **OpenCV (`cv2`)**: Used for interfacing with the external USB webcam, capturing frames, and rendering bounding boxes/text overlays in real-time.
  * **PyGame**: Utilized to trigger asynchronous, non-blocking audio alarms (`alarm.wav`) when infraction thresholds are breached.
