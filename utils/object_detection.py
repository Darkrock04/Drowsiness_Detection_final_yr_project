from ultralytics import YOLO

# Initialize YOLO model (nano version for speed)
model = YOLO('yolov8n.pt')

# COCO Class IDs
PERSON_CLASS_ID = 0
CELL_PHONE_CLASS_ID = 67

def detect_objects(frame):
    """
    Runs YOLOv8 object detection on the frame.
    Returns boolean flags for phone detection and person detection.
    Note: Standard COCO YOLOv8 does not include a 'seatbelt' class. 
    A custom trained YOLOv8 model would be needed for precise seatbelt detection.
    """
    results = model(frame, verbose=False)
    
    phone_detected = False
    person_detected = False
    
    boxes = results[0].boxes
    for box in boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        
        # We only consider reasonably confident detections
        if confidence > 0.4:
            if class_id == CELL_PHONE_CLASS_ID:
                phone_detected = True
            elif class_id == PERSON_CLASS_ID:
                person_detected = True
                
    return person_detected, phone_detected, results[0].plot()
