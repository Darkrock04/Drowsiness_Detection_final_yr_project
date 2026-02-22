# Object & Occupancy Detection (YOLOv8)

To track secondary safety features like phone usage and seatbelt/passenger occupancy, the system leverages modern Deep Learning via the YOLO (You Only Look Once) architecture.

## Model Choice: Ultralytics YOLOv8 Nano

The system currently employs the `yolov8n.pt` pre-trained model. 
- **Nano Size**: Chosen specifically for real-time inference on CPU/Webcam streams inside a vehicle cabin without requiring heavy GPU clusters.
- **Dataset**: Pre-trained on the **COCO** (Common Objects in Context) dataset.

## Features Implemented

### 1. Phone Usage Detection
Using the COCO `cell phone` class (Class ID: 67), the YOLO model scans the cabin frame concurrently with the facial landmark detector.
- **Logic**: If a cell phone bounding box is detected with confidence > 40%, the system flags the driver for a Secondary Activity / Phone Alert.

### 2. Passenger Occupancy & Seatbelts
- **Academic Context**: Earlier academic systems for passenger occupancy (such as the method described in *Comparison of Face Detection and Image Classification for Detecting Front Seat Passengers in Vehicles*) utilized classical Image Classification using Fisher Vectors (FV) and Bag-of-Visual-Words (BoW) built on SIFT descriptors.
- **Modern Upgrade**: Deep Neural Networks implicitly handle feature extraction much better than BoW/SIFT. Therefore, the architecture swaps classical image classification for YOLO object detection.
- **Current Limitation**: The base COCO dataset used by `yolov8n` has a `person` class but **does not** include a `seatbelt` class. To enforce real seatbelt detection, developers must fine-tune the YOLOv8 model using a specialized interior cabin seatbelt dataset.
