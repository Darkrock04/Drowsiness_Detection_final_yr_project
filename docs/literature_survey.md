# Literature Survey & Research Gap

This document outlines the academic context of the Driver Monitoring System, detailing the flaws in existing research implementations and explaining how our unified Edge-AI approach resolves them.

## 1. Literature Survey (The Status Quo)

Historically, research papers surrounding Driver Monitoring Systems (DMS) have relied heavily on two primary methodologies:

1.  **Mathematical Aspect Ratios (Dlib/MediaPipe):** Many papers utilize 68-point or 468-point 3D facial meshes to calculate Eye Aspect Ratios (EAR) and Mouth Aspect Ratios (MAR). While effective in controlled laboratory environments, these systems are mathematically fragile. If a driver turns their head more than 30 degrees (e.g., looking at a side mirror), the mathematical coordinates break, resulting in systemic false negatives.
2.  **Heavy Convolutional Neural Networks (CNNs):** Recent papers have shifted to using deep CNNs (like ResNet50, VGG16, or heavy PyTorch YOLO architectures) to classify driver states. While highly accurate, these models require massive computational overhead, usually demanding discrete NVIDIA GPUs (CUDA) to process video frames in real-time.
3.  **Fragmented Multi-Model Systems:** To detect both physiological states (drowsiness) and physical objects (phones/cigarettes), researchers often run a facial-landmark detector concurrently with an object detector. 

## 2. The Research Gap

The critical flaw in current academic literature is the **lack of realistic deployability**. 

While existing research papers boast 99% accuracy on high-end desktop computers, they fail entirely when deployed in a real-world vehicle. Modern vehicles rely on low-power embedded microcomputers (such as a Raspberry Pi or an automotive ECU). These embedded systems lack the massive RAM, cooling infrastructure, and dedicated GPUs required to run heavy PyTorch (`.pt`) models or run two separate neural networks simultaneously. 

The gap in the research lies in creating a system that maintains deep-learning accuracy while being fully optimized for a resource-constrained **Edge Device**.

## 3. Our Proposed Solution

This project bridges the deployability gap by introducing a **Single-Model Edge-AI Architecture**.

### The Role of Transfer Learning
Training a neural network entirely from scratch to recognize complex micro-expressions requires millions of images and months of GPU computing. Instead, we utilized **Transfer Learning**. 
By taking the foundational YOLOv8s (Small) architecture—which already possessed a deep understanding of generic shapes, edges, and lighting—we fine-tuned it on a highly specific 52,000-image dataset. This allowed the model to rapidly learn how to natively classify 5 simultaneous driver infractions (Drowsiness, Yawning, Distraction, Phone Usage, Smoking) natively as distinct classes, eliminating the need for fragile mathematical tracking.

### Edge Optimization (The TFLite Advantage)
To directly address the research gap regarding embedded systems, we specifically engineered our deployment pipeline away from standard desktop formats.

The custom-trained model was exported to a **TensorFlow Lite (`.tflite`)** format. During this export, we applied **Float16 (`FP16`) Quantization**. 
This critical engineering decision:
1.  **Reduced Memory Footprint:** It halved the physical size and RAM requirement of the model without suffering the severe accuracy loss associated with `INT8` quantization.
2.  **CPU Hardware Acceleration:** By executing the `.tflite` model using the `tensorflow` package, the system leverages the **XNNPACK CPU Delegate**. This delegate optimizes complex tensor operations to run natively and swiftly on standard ARM-based CPUs.

By compiling a massive 52,000-image intelligence down into a lightweight, quantified TFLite model, our project proves that comprehensive, 5-factor driver monitoring can be reliably deployed on an embedded system like a **Raspberry Pi**, effectively solving the primary limitation found in current DMS literature.
