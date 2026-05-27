# Project References

This document details the external datasets, training platform, frameworks, and technologies used to build the Driver Monitoring System.

### 1. Training Dataset — Driver Drowsiness Dataset (DMS)
* **Source:** Roboflow Universe
* **Link:** [https://universe.roboflow.com/drowsiness-detection-system/driver-drowsiness-3ldhq](https://universe.roboflow.com/drowsiness-detection-system/driver-drowsiness-3ldhq)
* **Description:** 
  * This is the massive 52,000+ image dataset used to train our custom deep learning model (`final_yr_drowsiness_model.tflite`, ~22MB).
  * It provides explicitly balanced classes for the core behaviors required for this project: `awake`, `distracted`, `eyes_closed`, `phone`, `smoking`, and `yawn`.
  * The dataset utilizes advanced Mosaic Augmentation to ensure the model learns to detect features across diverse lighting conditions, camera angles, and ethnicities.

### 2. Training Platform — Kaggle
* **Link:** [https://www.kaggle.com](https://www.kaggle.com)
* **Description:**
  * Our custom model was trained entirely on **Kaggle**, using dual NVIDIA T4 GPUs.
  * The training ran for **50 epochs over approximately 9 hours**.
  * Kaggle was selected over Google Colab because it provides longer session limits (up to 12 hours) and free access to high-performance GPU accelerators, which were essential for processing a dataset of this scale.

### 3. Ultralytics Framework
* **Source:** [https://github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)
* **Description:** 
  * The Ultralytics library was used to execute Transfer Learning, adapting general-purpose pre-trained weights to our custom Driver Monitoring dataset.
  * The same library provides a high-level API for loading and running inference on `.tflite` models at runtime.

### 4. TensorFlow Lite & XNNPACK
* **Source:** [https://www.tensorflow.org/lite](https://www.tensorflow.org/lite)
* **Description:**
  * TensorFlow Lite was selected as the deployment framework to ensure the model could run efficiently on edge-computing devices (such as a Raspberry Pi).
  * The model was exported using **Float16 Quantization** to reduce memory footprint by 50% while preserving mathematical precision.
  * The XNNPACK CPU delegate is leveraged at runtime to provide massive hardware acceleration without requiring a discrete GPU.

### 5. OpenCV & PyGame
* **Description:**
  * **OpenCV (`cv2`)**: Used for interfacing with the external USB webcam, capturing frames, and rendering bounding boxes/text overlays in real-time.
  * **PyGame**: Utilized to trigger asynchronous, non-blocking audio alarms (`alarm.wav`) when infraction thresholds are breached.
