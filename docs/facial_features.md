# Facial Landmarks and Head Pose (dlib)

The core driver monitoring features (Drowsiness, Yawning, and Distraction) rely fundamentally on accurate facial landmark mapping.

## Dataset & Pre-trained Model

We utilize the classical machine learning library **dlib**, specifically its pre-trained HOG (Histogram of Oriented Gradients) face detector and the `shape_predictor_68_face_landmarks.dat` model. This predictor was trained on the **iBUG 300-W** dataset, which consists of annotated faces in unconstrained environments.

## Methodology

### 1. Drowsiness (Eye Aspect Ratio - EAR)
The algorithm calculates the Euclidean distances between the vertical eye landmarks and horizontal eye landmarks. 
- **Landmarks Used**: 
  - Left Eye: `[36, 41]`
  - Right Eye: `[42, 47]`
- **Alert Logic**: If the resulting ratio falls below `0.25` for 20 consecutive frames, a Drowsiness Alert is triggered.

### 2. Yawning (Mouth Aspect Ratio - MAR)
Similar to the eyes, we calculate the ratio of the mouth's vertical opening to its horizontal width.
- **Landmarks Used**: Mouth inner/outer contour `[49, 68]`
- **Alert Logic**: If the ratio exceeds `0.79`, meaning the mouth is wide open, a Yawning Alert is triggered.

### 3. Distraction (3D Head Pose Estimation)
To measure if the driver is looking away from the road:
- We map 6 specific 2D facial landmarks (Nose tip, Chin, Eye corners, Mouth corners) to a generic 3D human face model.
- Using OpenCV's `solvePnP` (Perspective-n-Point) algorithm, we calculate the rotation and translation vectors of the head.
- These vectors are converted into Euler angles to determine the exact degree of the driver's head tilt.
- **Alert Logic**: A head tilt exceeding `20 degrees` for 15 frames triggers a Distraction Alert.
