# Project References

This internal documentation details the open-source repositories and educational references used as inspiration and foundational code for building the Driver Monitoring System from scratch.

### 1. Drowsiness Detection by Akshay Bahadur
* **Original Repo Path / Source:** `referance_1` - [https://github.com/akshaybahadur21/Drowsiness_Detection](https://github.com/akshaybahadur21/Drowsiness_Detection)
* **Contributions Used:** 
  * Provided the foundational logic for the **Eye Aspect Ratio (EAR)** calculation.
  * Demonstrated the mathematical relationship of the 6 coordinates surrounding the eye to confidently classify eye closure.

### 2. Driver Drowsiness Detector by Mohit
* **Original Repo Path / Source:** `referance_2` - [https://github.com/mohitwildbeast/Driver-Drowsiness-Detector](https://github.com/mohitwildbeast/Driver-Drowsiness-Detector)
* **Contributions Used:**
  * Inspired the overall application flow for real-time webcam video feed parsing using OpenCV.
  * Demonstrated the use of the `pygame` library to play non-blocking audio alarms when fatigue thresholds are met.

### 3. Driver Drowsiness Detection by Neelanjan Manna
* **Original Repo Path / Source:** `referance_3` - [https://github.com/neelanjan00/Driver-Drowsiness-Detection](https://github.com/neelanjan00/Driver-Drowsiness-Detection)
* **Contributions Used:**
  * Served as the primary inspiration for the **trifold behavioral approach**: monitoring Eye Aspect Ratio (EAR), Mouth Aspect Ratio (MAR) for yawning, and forward head tilt angle for distraction.
  * This reference shaped the core requirement logic evaluated in our main pipeline.

### 4. Deep Learning Based Driver Monitoring System (AI-Driver-Safety)
* **Original Repo Path / Source:** `referance_4` - [https://github.com/prasad-kumkar/ai-driver-safety](https://github.com/prasad-kumkar/ai-driver-safety)
* **Contributions Used:**
  * Provided a broader scope of what a real production-grade driver monitor should include (activity recognition, object detection).
  * Inspired the integration of a secondary neural network (in our case, YOLOv8) to concurrently monitor objects like phones while the primarily dlib-based network monitors the face.
