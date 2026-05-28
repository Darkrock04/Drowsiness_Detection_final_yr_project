# Driver Monitoring System - Raspberry Pi Deployment Server

This repository contains the edge-deployment server code for the **Driver Monitoring System**. The system runs real-time inference on a Raspberry Pi 4B using a custom-trained TensorFlow Lite (TFLite) model to detect driver drowsiness and distraction, streaming the live video feed along with safety status tags to a connected Android application over a local TCP socket.

---

## 🌟 Key Features

* **Lightweight TFLite Inference**: Optimized to run on the Raspberry Pi CPU using `tflite-runtime` instead of the heavy TensorFlow package.
* **TCP Live Streaming**: Streams JPEG-compressed video frames (at 45% quality) and active detection labels to an Android device over port `5000`.
* **Hardware Camera Optimization**: Focus-locked camera stream with hardware auto-focus loops disabled to prevent camera hunting and blurry frames.
* **Compatibility Layer**: Integrates a Python 3.13 mock module compatibility layer for running modern deep learning models without deprecated module issues.
* **Stability Counters**: Utilizes frame-by-frame stability thresholding to prevent false alarms from brief blinking or facial changes.

---

## 📁 File Structure

* **`main_final.py`**: The main production script. It initializes the TFLite model, captures video frames, draws bounding boxes, formats data packets, and manages the TCP server connection.
* **`main.py`**: A lightweight test server script used to send a mock static image to verify client-side TCP connection and socket decoding.
* **`requirements.txt`**: List of Python dependencies optimized for the Raspberry Pi environment (utilizing `tflite-runtime` for resource efficiency).
* **`final_yr_drowsiness_model.tflite`**: The custom-trained driver status classification and detection model.
* **`alarm.wav`**: Audio alert file stored for edge-side audible feedback.

---

## 🧠 Custom-Trained Model Details

The system uses a custom-trained model designed specifically for low-latency edge deployment on the Raspberry Pi:

* **Model File**: `final_yr_drowsiness_model.tflite`
* **Training Platform**: Kaggle (Utilized high-performance GPUs and TPU accelerators for training)
* **Dataset Size**: Trained on a dataset of **52,000+ driver images** sourced from Roboflow Universe.
* **Detection Resolution**: Runs inference at a resolution of $320 \times 240$ pixels, achieving high frame rates on edge hardware.

### Class Mappings & Action Thresholds

The model detects 6 specific driver states. To prevent false positives, a classification must persist for a minimum number of consecutive frames before triggering an active alert state.

| Model Class | Detection Label | Counter Threshold | Action Triggered |
| :--- | :--- | :---: | :--- |
| `awake` | **Safe** | *Immediate* | Normal operation, Green status |
| `eyes_closed` | **Drowsy** | 5 Frames | Red alert, play warning, save event |
| `yawn` | **Yawn** | 6 Frames | Red alert, play warning, save event |
| `distracted` | **Distraction** | 4 Frames | Red alert, play warning, save event |
| `phone` | **Phone Usage** | 3 Frames | Red alert, play warning, save event |
| `smoking` | **Smoking** | 3 Frames | Red alert, play warning, save event |

---

## 🔌 Socket Communication Protocol

The Raspberry Pi communicates with the Android client over a raw TCP socket (`socket.AF_INET`, `socket.SOCK_STREAM`) on **Port 5000**. The socket server binds to `0.0.0.0` (all interfaces) and waits for a single connection.

### Data Packet Structure

To handle dynamic image sizes and variable alert lengths, every frame payload is sent using a custom binary packet protocol. The bytes are ordered in **Big-Endian** format.

```
+-------------------+--------------------+------------------------+--------------------------+
|  Tag Length (4B)  |  Image Length (4B) |   Alert Tag (UTF-8)    |   JPEG Raw Frame Data    |
|   [Big-Endian]    |    [Big-Endian]    |  [Size = Tag Length]   |  [Size = Image Length]   |
+-------------------+--------------------+------------------------+--------------------------+
```

1. **Header (8 Bytes total)**:
   * **Bytes 0–3**: Size of the alert tag text in bytes (`Int`, 4 bytes).
   * **Bytes 4–7**: Size of the JPEG image payload in bytes (`Int`, 4 bytes).
2. **Payload**:
   * **Alert Tag**: The UTF-8 encoded string of the current driver status (e.g., `"Drowsy"`, `"Smoking"`, `"Safe"`).
   * **Image Data**: The raw compressed JPEG byte stream. The image features thin, green/red bounding boxes drawn around the detected region, compressed at a JPEG quality value of `45` to minimize network transit latency.

---

## ⚙️ How it Works (Main Loop)

1. **Initialization**: The mock `imp` compatibility module is loaded, the headless offscreen environment variable (`QT_QPA_PLATFORM=offscreen`) is set, and the TFLite model is compiled.
2. **Socket Setup**: The TCP server starts listening on port `5000` for the Android phone connection.
3. **Camera Initialization**: The script probes indexes (`0`, `1`, `2`, `-1`) to find the active camera. It disables hardware auto-focus and sets a static focal length suited for driver distance.
4. **Connection Handshake**: Once the Android app connects, the server enters the streaming loop.
5. **Inference & Packing**:
   * Reads a video frame.
   * Runs the custom model at a confidence threshold of `0.45` with memory streaming (`stream=True`).
   * Iterates through bounding boxes, updates counters, and overlays labels on the frame.
   * Compresses the frame to JPEG.
   * Packs the active alert status tag and JPEG bytes into the 8-byte header structure and sends the packet.
6. **Disconnection Handling**: If the socket throws a `BrokenPipeError` or `ConnectionResetError` (e.g., app closed), the server resets the client connection and waits for the app to reconnect without crashing the Pi script.
