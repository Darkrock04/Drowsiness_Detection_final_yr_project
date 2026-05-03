# Modules & Technologies Used

This document outlines the specific Python libraries, frameworks, and tools utilized to construct the Driver Monitoring System.

### 1. Ultralytics (YOLOv8)
* **Purpose:** The core deep learning framework.
* **Usage in Project:** Ultralytics provides the architecture for the YOLOv8 (You Only Look Once) object detection model. It was used during the training phase to adapt the generic pre-trained weights to our specific custom dataset via Transfer Learning. In the execution phase, the library provides a high-level API to seamlessly load and run inference on `.tflite` models.

### 2. TensorFlow (LiteRT)
* **Purpose:** Edge-optimized inference engine.
* **Usage in Project:** TensorFlow is required on the backend to execute the `.tflite` model format. Specifically, it leverages the **XNNPACK delegate**, which provides massive hardware acceleration for neural networks running on standard CPUs. This is what allows the complex AI to run at high frames-per-second on a standard laptop or microcomputer without requiring an expensive NVIDIA GPU.

### 3. OpenCV (`opencv-python`)
* **Purpose:** Real-time computer vision and image processing.
* **Usage in Project:** 
  * `cv2.VideoCapture()` is used to interface with the external USB webcam and fetch live video frames.
  * Used to flip the image horizontally for a natural "mirror" view.
  * `cv2.putText()` and `cv2.rectangle()` are utilized to draw visual alerts and bounding boxes directly onto the live video feed.

### 4. PyGame (`pygame`)
* **Purpose:** Asynchronous audio management.
* **Usage in Project:** Standard Python audio libraries pause the entire program's execution while a sound is playing, which would freeze the video feed. `pygame.mixer` is used to load and play the `alarm.wav` file in a completely separate, non-blocking background thread. This ensures the AI continues to monitor the driver uninterrupted even while the alarm is sounding.

### 5. NumPy (`numpy`)
* **Purpose:** Matrix mathematics and array manipulation.
* **Usage in Project:** Video frames captured by OpenCV are fundamentally 3D NumPy arrays (Height x Width x RGB Channels). NumPy handles the rapid, low-level memory allocation required to pass these frames continuously into the AI model.
