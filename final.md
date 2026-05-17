# Smart Driver Monitoring System Using Edge AI

## Abstract

Road accidents caused by driver fatigue, distraction, mobile phone usage, and unsafe in-vehicle behavior remain a serious safety problem. This project presents a Smart Driver Monitoring System in which we have trained our own deep learning model to observe the driver in real time and detect unsafe behaviors before they lead to risk. The system is built around our trained object detection model, developed using a large driver-monitoring dataset and exported to TensorFlow Lite with Float16 quantization for lightweight edge deployment.

The model detects multiple driver states and actions from a live camera feed, including drowsiness, yawning, distraction, phone usage, and smoking. Instead of depending on fragile facial-landmark calculations such as Eye Aspect Ratio or Mouth Aspect Ratio, the proposed system uses object detection to identify driver behavior directly from video frames. OpenCV captures camera frames, the trained TensorFlow Lite model performs inference, and the Raspberry Pi sends annotated images with alert tags to the Android application through a TCP socket connection.

The system is designed for practical deployment in a Raspberry Pi environment. In the integrated setup, the Raspberry Pi works as a server and waits for the Android application to connect. Once connected, it streams compressed annotated frames and sends an alert tag such as `DROWSY`, `YAWN`, `PHONE`, `SMOKING`, `DISTRACTED`, or `STREAM`. The Android application can use these tags to display alerts and store detected event images for later review.

## Keywords

Smart Driver Monitoring System, Drowsiness Detection, TensorFlow Lite, Raspberry Pi, Edge AI, OpenCV, Driver Distraction, Real-Time Object Detection, Road Safety

## Introduction

Driver monitoring is an important part of modern road safety systems. Many accidents happen because the driver becomes sleepy, looks away from the road, uses a mobile phone, smokes while driving, or shows signs of fatigue such as yawning. Traditional safety systems mainly respond after danger occurs, but a driver monitoring system can warn the driver during the early stage of unsafe behavior.

This project develops a real-time Smart Driver Monitoring System using computer vision and deep learning. A webcam continuously captures the driver's face and upper-body region. Each frame is passed into our trained model that identifies unsafe driver conditions through bounding boxes and class labels. When a risky behavior is detected, the Raspberry Pi sends the detected event tag and annotated image frame to the Android application.

The project focuses on edge deployability. The trained model is exported into TensorFlow Lite format using Float16 quantization, which reduces the model size and makes it suitable for Raspberry Pi or other embedded hardware. The system also includes an Android app-connected Raspberry Pi environment where event images such as drowsy, yawning, distracted, phone usage, and smoking frames can be stored for monitoring and analysis.

## Project Overview

The Smart Driver Monitoring System is a real-time AI-based safety application. Its core component is our trained object detection model saved as `final_yr_drowsiness_model.tflite`. The model receives live camera frames and detects driver behavior classes in one unified inference pipeline.

The project detects the following major driver events:

| Driver Behavior | Model Class / Signal | Android Alert Tag |
| --- | --- | --- |
| Drowsiness | `drowsy` | `DROWSY` |
| Yawning | `yawn` | `YAWN` |
| Distraction | `distracted` | `DISTRACTED` |
| Phone Usage | `phone` | `PHONE` |
| Smoking | `smoking` | `SMOKING` |

If no unsafe behavior is detected in the current frame, the system sends the default `STREAM` tag. This allows the Android app to distinguish between normal live streaming and risky driver events.

During development, the project first used `main_final.py` as the standalone driver monitoring code for local model testing. After Android app integration, the implementation was changed to the Raspberry Pi socket-streaming code shown in `note.txt`. In the final workflow, the Raspberry Pi runs the model, sends the live annotated stream to the Android APK, and the app saves a snapshot whenever it receives an unsafe activity or distraction tag.

## Motivation

Road safety depends heavily on driver alertness. Even a short period of drowsiness or distraction can lead to delayed reaction time and serious accidents. Manual observation of drivers is not possible in normal vehicles, and many existing systems are either too expensive, too computationally heavy, or unreliable under real-world conditions.

The motivation of this project is to create a practical, low-cost, and deployable monitoring system that can run on edge hardware. Raspberry Pi-based deployment makes the system suitable for vehicles where power, cost, and space are limited. The app integration further improves usefulness by storing captured images of unsafe events, allowing later review by the driver, fleet manager, or safety administrator.

## Objectives

The main objective is to design and implement an AI-powered driver monitoring system that can detect unsafe driver behavior in real time and provide immediate alerts.

The specific objectives are:

- Train our own object detection model for driver state and behavior detection.
- Detect drowsiness, yawning, distraction, phone usage, and smoking using a single model.
- Convert the trained model into TensorFlow Lite Float16 format for efficient edge deployment.
- Capture real-time frames from an external webcam using OpenCV.
- Optimize frames for Raspberry Pi processing and Wi-Fi transmission.
- Send alert tags and annotated frames to the Android application in real time.
- Support Raspberry Pi-based deployment for vehicle-level use.
- Integrate an application workflow that stores detected event images for later monitoring.

## Uniqueness of the Work

The project is unique because it uses a single object detection model trained for this driver monitoring task instead of combining multiple separate systems. Many driver monitoring approaches use facial landmarks for drowsiness and a separate object detector for phone or smoking detection. This increases complexity and reduces real-time performance on low-power devices. In this project, our single trained model handles all major driver states and unsafe actions together.

The model is also optimized for edge use. After training, the final model weights are exported to TensorFlow Lite with Float16 quantization. This reduces the memory footprint while preserving useful detection accuracy. The system can therefore be deployed in a Raspberry Pi environment rather than depending only on a high-end GPU computer.

Another important contribution is event logging through the app-connected Raspberry Pi setup. When risky behavior is detected, the Raspberry Pi sends the annotated frame with the event tag, allowing the Android application to store a practical monitoring record of drowsy, yawning, distracted, phone, and smoking events.

## Report Layout

This document follows the structure of a final-year project report and manuscript:

- The introduction explains the problem, motivation, objectives, and uniqueness.
- The literature survey discusses existing driver monitoring methods and their limitations.
- The materials and methods section describes the dataset, tools, model training, and system workflow.
- The implementation section explains the real-time monitoring pipeline and Raspberry Pi/app integration.
- The results section summarizes expected system behavior and event detection output.
- The conclusion and future scope section presents the overall outcome and possible improvements.

## Literature Survey

Existing driver monitoring systems generally follow three approaches: facial landmark analysis, traditional machine learning, and deep learning-based vision models.

Facial landmark methods use points around the eyes and mouth to calculate values such as Eye Aspect Ratio and Mouth Aspect Ratio. These systems can detect closed eyes or yawning in controlled conditions, but they become unreliable when the driver turns their head, lighting changes, the camera angle shifts, or the face is partly covered.

Traditional machine learning systems usually depend on handcrafted features. These systems are simpler, but they often struggle in real-world driving environments because driver appearance, lighting, posture, and camera placement can vary widely.

Deep learning systems such as CNNs and object detectors improve accuracy by learning visual patterns directly from data. However, many deep learning models are computationally heavy and require GPU support for smooth real-time operation. Some systems also use multiple models at the same time, one for face state detection and another for object detection. This increases processing load and makes deployment difficult on embedded hardware.

The research gap is realistic deployability. A driver monitoring system must work inside a vehicle, where hardware resources are limited. It must detect several behaviors at once, run in real time, and avoid unnecessary false alarms. This project addresses the gap through a single-model Edge AI approach using our trained model and TensorFlow Lite.

## Problem Identification

Current driver monitoring systems face the following issues:

- Landmark-based systems can fail when the face angle changes or lighting is poor.
- Heavy CNN or PyTorch-based models may not run efficiently on low-power devices.
- Multi-model systems increase processing time and hardware requirements.
- Many systems detect only drowsiness and ignore other unsafe behaviors such as phone usage, smoking, and distraction.
- Real-time mobile monitoring is difficult if the vehicle-side device cannot stream frames efficiently.
- Event records are often not stored, making later safety review difficult.

The proposed project solves these issues by using one lightweight object detection model, optimizing frames for Raspberry Pi inference, transmitting compressed annotated images to an Android app, and supporting image storage through the app environment.

## Materials and Methods

### Dataset Description

The model was trained using the Driver Drowsiness dataset from Roboflow Universe. The dataset contains more than 52,000 images for driver monitoring and includes the classes required for this project:

- `awake`
- `distracted`
- `drowsy`
- `phone`
- `smoking`
- `yawn`

The dataset was downloaded in object-detection format using the Roboflow API. It supports training a model that can recognize driver states and unsafe actions under different visual conditions.

### Model Training

The model training was performed using an object detection training framework. In this project, we trained our driver monitoring model using the prepared dataset. The training process allowed the model to learn the required driver behavior classes directly from the collected dataset.

The training configuration used in the notebook includes:

| Parameter | Value |
| --- | --- |
| Model type | Object detection model |
| Dataset format | Object-detection dataset |
| Epochs | 50 |
| Image size | 416 |
| Batch size | 16 |
| Training run name | `driver_monitoring_final` |
| Export format | TensorFlow Lite |
| Quantization | Float16 |
| Final model file | `final_yr_drowsiness_model.tflite` |

After training, the best model weights were exported to TensorFlow Lite using `model.export(format='tflite', half=True)`. The generated model was renamed as `final_yr_drowsiness_model.tflite` for deployment.

### Tools Used

| Tool / Library | Purpose in the Project |
| --- | --- |
| Python | Main programming language |
| Object detection framework | Model training and inference |
| TensorFlow Lite | Lightweight edge inference format |
| OpenCV | Webcam access, frame resizing, JPEG compression, and annotated frame handling |
| Python socket | TCP server connection between Raspberry Pi and Android app |
| Struct | Packing tag and image sizes before network transmission |
| NumPy | Array and frame data handling |
| Roboflow | Dataset download and preparation |
| Raspberry Pi | Edge deployment environment |

### System Requirements

Recommended hardware:

- Raspberry Pi 
- External USB webcam
- Android phone connected to the same network
- Storage for captured event images in the app workflow
- Power supply suitable for continuous monitoring

Recommended software:

- Python 3.10 or compatible environment
- `opencv-python`
- `numpy`
- `ultralytics`
- `tensorflow`
- `final_yr_drowsiness_model.tflite`

## Proposed System

The proposed system continuously monitors the driver through a camera. Each captured frame is processed by our TensorFlow Lite model. The model predicts bounding boxes and class labels for driver behavior. The Raspberry Pi maps these detected classes into alert tags such as `DROWSY`, `YAWN`, `DISTRACTED`, `PHONE`, `SMOKING`, or the default safe tag `STREAM`.

The alert system is tag-based. For every frame, the Raspberry Pi sets a default safe tag as `STREAM`. If the model detects `drowsy`, `yawn`, `smoking`, `phone`, or `distracted`, the tag is changed to the detected behavior in uppercase form and sent to the Android application with the annotated frame.

In the integrated Raspberry Pi setup, the Android application can store images whenever a risky behavior tag is received. These stored snapshots provide a record of unsafe driving events such as drowsiness, yawning, distraction, phone usage, and smoking.

The earlier `main_final.py` file was used before Android integration to verify the model and real-time detection logic. The final integrated code is the Raspberry Pi server code in `note.txt`, where the Pi connects with the Android APK over the network and sends detected user activity frames for snapshot storage.

## System Architecture

The system architecture contains the following layers:

1. Input layer: USB webcam captures the live driver video.
2. Processing layer: OpenCV reads frames and prepares them for inference.
3. AI inference layer: Our TensorFlow Lite model detects driver behavior.
4. Decision layer: Class labels are mapped to Android alert tags.
5. Network layer: Raspberry Pi sends the tag and compressed annotated image through TCP.
6. App layer: Android displays the stream and stores detected event images.
7. Output layer: Driver receives app-based monitoring feedback, and event logs remain available for review.

## Execution Workflow

### Frame Acquisition

OpenCV connects to the external USB webcam through `cv2.VideoCapture(0)`. The system waits briefly for the camera to initialize, checks whether the camera opened successfully, and then continuously reads frames. Each frame is resized to `480 x 360` so that the Raspberry Pi can process it faster and transmit it smoothly over Wi-Fi.

### AI Inference

The model is loaded from the TensorFlow Lite file using the object detection API:

```python
model = YOLO("final_yr_drowsiness_model.tflite")
```

Each optimized frame is passed to the model using streaming inference:

```python
results = model(frame, stream=True, verbose=False)
```

The `stream=True` option helps reduce processing overhead on Raspberry Pi, while `verbose=False` avoids unnecessary console output during continuous monitoring.

### Classification Mapping

The system checks each detected bounding box and reads its class name. The class name is converted into an Android alert tag:

- `drowsy` sends `DROWSY`
- `yawn` sends `YAWN`
- `distracted` sends `DISTRACTED`
- `phone` sends `PHONE`
- `smoking` sends `SMOKING`

If none of these classes is detected, the system keeps the default tag as `STREAM`.

### Raspberry Pi Server Setup

The Raspberry Pi acts as a TCP server. It listens on `0.0.0.0:5000`, allowing the Android application to connect over the local network. The socket option `SO_REUSEADDR` is enabled so that the script can restart without failing due to the port being temporarily busy.

### Android Data Transmission

After inference, the detected bounding boxes are drawn on the frame. The annotated frame is compressed into JPEG format at 50% quality to reduce Wi-Fi delay. The Raspberry Pi sends two parts of data to the Android app:

- The alert tag, such as `STREAM`, `DROWSY`, `YAWN`, `PHONE`, `SMOKING`, or `DISTRACTED`.
- The compressed annotated image frame.

Before sending the tag and image, the system packs their byte sizes using `struct.pack(">II", len(tag_bytes), len(img_bytes))`. This helps the Android app correctly separate the tag data from the image data while receiving the stream.

### Event Image Storage

In the Raspberry Pi-connected app workflow, detected unsafe events can be saved as image records by the Android app. For example, if the received tag is `DROWSY`, `YAWN`, `DISTRACTED`, `PHONE`, or `SMOKING`, the corresponding annotated frame can be stored with the event category. This supports later analysis and makes the system useful for fleet monitoring, driver feedback, and safety reporting.

## Implementation Details

The final Raspberry Pi implementation in `note.txt` performs the following tasks:

- Sets `QT_QPA_PLATFORM=offscreen` so the Raspberry Pi script does not crash when no monitor is connected.
- Imports OpenCV, socket, struct, time, and the object detection API.
- Loads our trained TensorFlow Lite model using `YOLO("final_yr_drowsiness_model.tflite")`.
- Creates a TCP server on port `5000`.
- Waits for the Android app to connect.
- Opens the USB webcam using `cv2.VideoCapture(0)`.
- Resizes each frame to `480 x 360` for faster Raspberry Pi processing.
- Runs inference on every frame using `stream=True`.
- Draws bounding boxes on the frame.
- Converts detections into Android alert tags.
- Compresses the annotated frame into JPEG format.
- Sends tag size, image size, tag bytes, and image bytes to the Android app.
- Handles Android app disconnection and returns to listening mode.

## Proposed Algorithm

The proposed algorithm can be summarized as:

1. Set the Raspberry Pi display environment to offscreen mode.
2. Load our trained TensorFlow Lite model.
3. Start a TCP server on `0.0.0.0:5000`.
4. Open the USB webcam.
5. Wait for the Android application to connect.
6. Capture a frame from the camera.
7. Resize the frame to `480 x 360`.
8. Run model inference using streaming mode.
9. Draw detection bounding boxes on the frame.
10. Set the default alert tag to `STREAM`.
11. If a risky class is detected, change the tag to the detected event name.
12. Compress the annotated frame into JPEG format.
13. Send the tag size, image size, tag, and image to the Android app.
14. If the Android app disconnects, close the client socket and return to listening mode.

## Experimentation and Model Evaluation

The project evaluation focuses on whether the system can identify unsafe driver behavior in real time and respond with stable alerts. Our model was trained for 50 epochs. The image size of 416 was selected as a balance between speed and detection quality, while batch size 16 was used during training.

The final model was exported to TensorFlow Lite Float16 format. This deployment format is important because the project targets edge devices such as Raspberry Pi, where memory and processing power are limited compared to high-end GPU machines.

The real-time application evaluates the model output frame by frame. The Raspberry Pi script is optimized for continuous mobile streaming by resizing frames, using streaming inference, compressing the annotated output as JPEG, and sending only the required tag and image bytes to the Android app.

## Results

The implemented system successfully provides a real-time driver monitoring workflow. It can:

- Detect closed eyes as drowsiness.
- Detect yawning behavior.
- Detect driver distraction.
- Detect phone usage.
- Detect smoking behavior.
- Show bounding boxes around detected behavior.
- Draw bounding boxes on the live camera frame.
- Send event tags to the Android app.
- Stream compressed annotated images over Wi-Fi.
- Return to listening mode when the Android app disconnects.
- Support Android app-based storage of detected event images.

The result is a practical driver safety prototype that combines AI detection, Raspberry Pi edge processing, Android live streaming, alert tagging, and event logging.

## System Performance Discussion

The system is designed for real-time use. Our trained model provides a strong balance between accuracy and speed, and TensorFlow Lite Float16 export makes it more suitable for edge hardware. The use of XNNPACK acceleration through TensorFlow Lite helps CPU-based inference, reducing the need for a dedicated GPU.

The network-based implementation improves practical usability because the Raspberry Pi can run without a monitor and stream the processed output directly to the Android app. The `offscreen` display setting prevents GUI-related crashes, while the 480 x 360 frame size and 50% JPEG quality reduce Wi-Fi lag.

The app-connected storage feature improves the monitoring value of the system. Instead of only processing frames locally, the system can send tagged annotated images to the Android app, where detected events can be preserved for later review.

## Advantages

- Uses one model for multiple driver behaviors.
- Suitable for real-time driver monitoring.
- Optimized for edge deployment through TensorFlow Lite.
- Runs on Raspberry Pi without requiring a connected monitor.
- Streams annotated frames to an Android app through TCP.
- Sends event tags along with each image frame.
- Supports event image storage through Android app integration.
- Can be extended for fleet safety and driver behavior analytics.

## Limitations

- Detection quality depends on camera position and lighting.
- The current implementation uses a USB webcam index that may need adjustment on different systems.
- Extreme occlusion, sunglasses, or poor night lighting may reduce detection reliability.
- The app storage workflow depends on proper Raspberry Pi and Android network connectivity.
- If the Wi-Fi connection is weak, streamed frames may lag or disconnect.
- More real-world vehicle testing is needed for final production-level validation.

## Individual Contributions

The project can be divided into the following contribution areas:

| Contribution Area | Work Done |
| --- | --- |
| Literature survey | Studied existing driver monitoring systems, drowsiness detection methods, and edge AI limitations |
| Dataset preparation | Used Roboflow driver drowsiness dataset in object-detection format |
| Model training | Trained our driver monitoring model for driver behavior detection |
| Model optimization | Exported trained model to TensorFlow Lite Float16 format |
| Application development | Built real-time OpenCV monitoring, frame annotation, and JPEG transmission logic |
| Raspberry Pi integration | Built TCP server workflow for Android app streaming and event storage |
| Testing and documentation | Verified model call, camera capture, app connection, alert tags, and documented system workflow |

## Conclusion

This project successfully develops a Smart Driver Monitoring System using our trained model deployed through TensorFlow Lite. The system detects drowsiness, yawning, distraction, phone usage, and smoking from a live camera feed. The Raspberry Pi processes the camera frames, draws the detection output, assigns an event tag, and streams the result to the Android application.

The project demonstrates how a deep learning-based driver safety system can be made practical for edge hardware. By exporting the model to TensorFlow Lite Float16 format and using a socket-based Android streaming workflow, the system becomes more suitable for Raspberry Pi deployment. The app-connected image storage workflow further extends the project from a simple detection system into a monitoring and evidence-recording platform.

## Future Scope

Future improvements can include:

- Add night-vision or infrared camera support for low-light driving.
- Improve Raspberry Pi performance using hardware acceleration.
- Add cloud synchronization for stored event images.
- Build a full mobile dashboard for reviewing alerts and driver history.
- Add GPS and timestamp metadata to each stored event.
- Include driver identity recognition for fleet-level monitoring.
- Train with more real-world vehicle data under different lighting, road, and camera conditions.
- Add seatbelt detection and lane-attention monitoring.
- Use notification alerts through SMS, email, or mobile push messages.

## References

1. Roboflow Universe, Driver Drowsiness Dataset: https://universe.roboflow.com/drowsiness-detection-system/driver-drowsiness-3ldhq
2. Ultralytics object detection framework: https://github.com/ultralytics/ultralytics
3. TensorFlow Lite Documentation: https://www.tensorflow.org/lite
4. OpenCV Documentation: https://opencv.org/
5. Python Socket Programming Documentation: https://docs.python.org/3/library/socket.html
6. Project repository: https://github.com/Darkrock04/Drowsiness_Detection_final_yr_project.git

## Appendices

### Appendix 1: Project Files

| File | Description |
| --- | --- |
| `main_final.py` | Earlier standalone driver monitoring code before Android integration |
| `train_combined_model.ipynb` | Training and export notebook |
| `final_yr_drowsiness_model.tflite` | Trained TensorFlow Lite model |
| `note.txt` | Final Android-integrated Raspberry Pi socket-streaming code with model call |
| `requirements.txt` | Required Python dependencies |
| `README.md` | Project overview and setup instructions |

### Appendix 2: Required Dependencies

```text
opencv-python
numpy
ultralytics
tensorflow
```

### Appendix 3: Android Alert Tags

| Behavior | Tag Sent to App |
| --- | --- |
| Normal stream | `STREAM` |
| Drowsiness | `DROWSY` |
| Yawning | `YAWN` |
| Distraction | `DISTRACTED` |
| Phone usage | `PHONE` |
| Smoking | `SMOKING` |

### Appendix 4: Key Deployment Command

```bash
python note.txt
```

The final Raspberry Pi script listens for the Android app on port `5000`. When the Android APK connects, it receives the annotated stream and saves snapshots whenever an unsafe activity or distraction tag is detected. If the Android app disconnects, the script closes the client socket and returns to listening mode.

## Reflection of the Team Members on the Project

This project helped us understand the complete workflow of an AI-based IoT safety system, from dataset selection and model training to real-time deployment and edge integration. We gained practical experience in object detection, model training, TensorFlow Lite conversion, OpenCV-based video processing, and Raspberry Pi deployment planning.

The work also highlighted the importance of practical engineering decisions. A model that performs well on a computer is not enough for a vehicle environment; it must be lightweight, stable, and able to respond in real time. By combining our trained model, TensorFlow Lite, Raspberry Pi socket streaming, alert tags, and app-based image storage, the project became a complete smart monitoring prototype rather than only a model-training exercise.
