# Modules & Technologies Used

This document outlines every single Python library, framework, and tool utilized to construct the Driver Monitoring System, along with a detailed explanation of their purpose and how they are used in the project.

---

### 1. Ultralytics
* **What is it?** Ultralytics is a Python deep learning framework that provides a high-level API for training, exporting, and running object detection models.
* **Why we use it:** It is the backbone of our entire inference engine. Without it, we would have to manually write hundreds of lines of low-level TensorFlow code to preprocess images, run tensor computations, and decode bounding boxes.
* **How it is used in the project:**
  * **Training Phase (Kaggle):** Used to apply Transfer Learning on our 52,000-image dataset, fine-tuning a pre-trained architecture to recognize our 6 custom classes.
  * **Runtime (main_final.py):** The `YOLO()` class loads our custom model (`final_yr_drowsiness_model.tflite`) and the `model()` call handles frame preprocessing, inference, and bounding box extraction in a single function call.
  * **Key function:** `model(frame, verbose=False, conf=0.45)` — runs inference on a single video frame and returns detection results.

---

### 2. TensorFlow / TensorFlow Lite (LiteRT)
* **What is it?** TensorFlow is Google's open-source machine learning platform. TensorFlow Lite (TFLite) is a lightweight version designed specifically for mobile and embedded devices.
* **Why we use it:** Our model is saved in `.tflite` format. TensorFlow is required as the backend engine to interpret and execute this optimized model format.
* **How it is used in the project:**
  * Runs silently in the background — when Ultralytics loads a `.tflite` file, it automatically delegates the computation to TensorFlow's **XNNPACK CPU delegate**.
  * XNNPACK provides massive hardware acceleration for neural networks running on standard CPUs, allowing our model to run at high frames-per-second without an expensive NVIDIA GPU.
  * On the Raspberry Pi, this can be replaced with the much lighter `tflite-runtime` package (only 5MB) instead of the full TensorFlow installation.

---

### 3. OpenCV (`opencv-python`)
* **What is it?** OpenCV (Open Source Computer Vision Library) is the industry-standard library for real-time image and video processing.
* **Why we use it:** It provides all the tools needed to capture live video, manipulate frames, and display visual output.
* **How it is used in the project:**
  * `cv2.VideoCapture(1)` — Connects to the external USB webcam and captures live video frames.
  * `cv2.flip(frame, 1)` — Flips the frame horizontally to create a natural "mirror" view for the driver.
  * `cv2.putText()` — Draws red alert text (e.g., `"DROWSINESS ALERT!"`) directly onto the video frame when an infraction is detected.
  * `cv2.imshow()` — Displays the live annotated video feed in a window titled "Driver Monitoring System".
  * `cv2.waitKey(1)` — Listens for the `q` key press to safely exit the application.

---

### 4. PyGame (`pygame`)
* **What is it?** PyGame is a Python library originally built for game development, but widely used for audio playback due to its non-blocking audio engine.
* **Why we use it:** Standard Python audio libraries (like `playsound` or `os.system`) pause the entire program while a sound is playing. This would freeze the video feed and leave the driver unmonitored during the alarm. PyGame solves this by playing audio in a completely separate background thread.
* **How it is used in the project:**
  * `pygame.mixer.init()` — Initializes the audio engine at program start.
  * `pygame.mixer.music.load("alarm.wav")` — Pre-loads the alarm sound file into memory.
  * `pygame.mixer.music.play()` — Plays the alarm asynchronously (non-blocking).
  * `pygame.mixer.music.stop()` — Instantly mutes the alarm when the driver returns to safe behavior.
  * `pygame.mixer.music.get_busy()` — Checks if the alarm is currently playing to prevent overlapping audio loops.

---

### 5. NumPy (`numpy`)
* **What is it?** NumPy is the fundamental package for numerical computing in Python, providing support for large multi-dimensional arrays and matrices.
* **Why we use it:** Video frames captured by OpenCV are fundamentally 3D NumPy arrays with dimensions (Height × Width × RGB Channels). NumPy is a mandatory dependency for both OpenCV and Ultralytics.
* **How it is used in the project:**
  * Handles the rapid, low-level memory allocation required to pass video frames continuously into the model for inference.
  * Used internally by Ultralytics to process bounding box coordinates and confidence scores.

---

### 6. Python Standard Library Modules
These are built-in Python modules that do not require installation:

* **`time`**
  * `time.sleep(2.0)` — Adds a 2-second delay after opening the webcam to allow the camera hardware to properly initialize and auto-adjust its brightness/exposure settings. Without this, the first few frames would be completely black or overexposed.

* **`os`**
  * `os.path.exists(model_path)` — Checks if the model file (`final_yr_drowsiness_model.tflite`) actually exists in the project directory before attempting to load it. This prevents a cryptic crash and instead displays a clean error message: `"[ERROR] Could not find final_yr_drowsiness_model.tflite!"`.

---

## Detection Classes

Our custom model (`final_yr_drowsiness_model.tflite`) was trained to recognize 6 distinct classes. Here is what each class means and how the system uses it:

| Class Label | What the Model Detects | System Response |
| :--- | :--- | :--- |
| `awake` | Driver's eyes are open and alert | No action (safe state) |
| `eyes_closed` | Driver's eyes are shut (drowsiness/micro-sleep) | Triggers **DROWSINESS ALERT** after 5 frames |
| `yawn` | Driver's mouth is wide open in a sustained yawn | Triggers **YAWNING ALERT** after 8 frames |
| `distracted` | Driver is looking away from the road | Triggers **DISTRACTION ALERT** after 5 frames |
| `phone` | Driver is holding or using a mobile device | Triggers **PHONE USAGE ALERT** after 3 frames |
| `smoking` | Driver is holding a cigarette or smoking | Triggers **SMOKING ALERT** after 4 frames |

---

## Audio Alert File

* **File:** `alarm.wav`
* **Purpose:** This is a short, loud warning tone that plays through the device's speakers when any infraction threshold is breached.
* **Format:** Standard WAV audio format, compatible with PyGame's mixer engine.
* **Behavior:** The alarm plays in a non-blocking loop and automatically stops the moment the driver corrects their behavior and all counters drop below the safe threshold.
