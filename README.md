# WildGuard Sentinel - Elephant Detection System

**Transforming wildlife monitoring into intelligent, automated surveillance**

![Python](https://img.shields.io/badge/Python-3.10%2B-blue) ![YOLOv8](https://img.shields.io/badge/YOLOv8-Latest-green) ![Firebase](https://img.shields.io/badge/Firebase-Real--time-orange) ![Raspberry Pi](https://img.shields.io/badge/Raspberry%20Pi-Compatible-red)

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Tech Stack](#tech-stack)
- [System Architecture](#system-architecture)
- [Project Structure](#project-structure)
- [Installation & Setup](#installation--setup)
- [Running the Application](#running-the-application)
- [Dataset Capture & Preparation](#dataset-capture--preparation)
- [Model Training](#model-training)
- [Testing & Inference](#testing--inference)
- [API & Integration](#api--integration)
- [Hardware Requirements](#hardware-requirements)

---

## Overview

**WildGuard Sentinel** is an intelligent elephant detection and monitoring system designed for wildlife conservation. It leverages computer vision and real-time processing to detect elephants in both daytime and nighttime environments, logging detection events to Firebase for real-time tracking and analytics.

The system uses **dual YOLO models**:
- **YOLOv8n** for daytime detection (faster, optimized for daylight conditions)
- **YOLOv5** for nighttime detection (handles low-light scenarios)

This dual-model approach ensures reliable elephant detection across varying lighting conditions, making it suitable for 24/7 wildlife monitoring in conservation areas.

---

## Key Features

✨ **Dual-Model Detection System**
- YOLOv8n for daytime elephant detection with high speed
- YOLOv5 for nighttime/low-light elephant detection
- Automatic model selection based on ambient light sensors

🎥 **Multi-Source Input Support**
- USB webcam / Raspberry Pi camera support
- Real-time video processing with adjustable resolution (480x360 default)

📍 **Real-Time Firebase Integration**
- Automatic logging of elephant detections to Firestore
- Timestamp recording for each detection event
- Detection count aggregation
- Cloud-based data persistence for analysis

🔌 **Hardware Sensor Integration**
- PIR (Passive Infrared) motion sensor for energy-efficient triggering
- Light sensor for automatic day/night model switching
- GPIO support for Raspberry Pi deployment

⚙️ **Intelligent Detection Logic**
- 30-second no-detection cooldown before entering standby mode
- Confidence-based filtering (>0.25 threshold for night detections)
- Real-time annotated video output with detection counts
- Model type display on video feed

---

## Tech Stack

| Component | Technology | Version |
|-----------|-----------|---------|
| **Object Detection** | YOLOv8, YOLOv5 | Latest |
| **Deep Learning Framework** | PyTorch | 2.x |
| **Computer Vision** | OpenCV | 4.8+ |
| **Backend Database** | Firebase/Firestore | Cloud |
| **Hardware Interface** | GPIOZero | 1.x |
| **Development** | Python | 3.10+ |
| **Model Training** | Ultralytics | Latest |
| **Notebook Environment** | Google Colab | T4 GPU |

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    WildGuard Sentinel System                 │
└─────────────────────────────────────────────────────────────┘
                              │
                ┌─────────────┼─────────────┐
                │             │             │
          ┌─────▼────┐  ┌────▼─────┐  ┌───▼──────┐
          │  Camera  │  │ PIR      │  │ Light    │
          │ (USB/RPi)│  │ Sensor   │  │ Sensor   │
          └─────┬────┘  └────┬─────┘  └───┬──────┘
                │            │            │
                └────────────┼────────────┘
                             │
                    ┌────────▼────────┐
                    │  Main Loop      │
                    │  (main.py)      │
                    └────────┬────────┘
                             │
                ┌────────────┼────────────┐
                │                        │
        ┌───────▼──────┐        ┌────────▼───────┐
        │   Day Model  │        │  Night Model   │
        │   (YOLOv8n)  │        │   (YOLOv5)     │
        └───────┬──────┘        └────────┬───────┘
                │                        │
                └────────────┬───────────┘
                             │
                    ┌────────▼────────┐
                    │  Detection      │
                    │  Processing     │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  Firebase       │
                    │  Firestore      │
                    │  (Cloud DB)     │
                    └─────────────────┘
```

---

## Project Structure

```
SSSVC2025_SWDD-Pole_WildGuard-Sentinel/
│
├── main.py                      # Main integration script - runs detection loop
├── yolo_detect.py               # Universal YOLO inference utility
├── PictureTaker.py              # Dataset capture script for training data
├── v8_Figure2.ipynb             # Google Colab training notebook (YOLOv8n)
├── my_model.pt                  # Trained YOLOv8n model weights
├── wildguardsentinel.json       # Firebase credentials (not included)
├── README.md                    # Original minimal README
└── README_DETAILED.md           # This file
```

### File Descriptions

| File | Purpose |
|------|---------|
| **main.py** | Core application that runs detection loop. Manages PIR/light sensors, switches between YOLOv8 (day) and YOLOv5 (night), processes camera frames, and updates Firebase |
| **yolo_detect.py** | Standalone inference tool for testing models. Supports images, image folders, video files, USB cameras, and Raspberry Pi cameras with customizable confidence thresholds and resolution |
| **PictureTaker.py** | Dataset capture utility. Records webcam images at specified resolution for training data preparation |
| **v8_Figure2.ipynb** | Complete YOLOv8n training workflow. Includes data preparation, Roboflow integration, training configuration, and model evaluation |
| **my_model.pt** | Pre-trained YOLOv8n weights optimized for elephant detection in daytime conditions |
| **wildguardsentinel.json** | Firebase service account credentials (must be obtained from Firebase Console) |

---

## Installation & Setup

### 1. Prerequisites

- **Python 3.10+** installed on your system
- **Git** for cloning the repository
- **Pip** package manager
- Camera device (USB webcam or Raspberry Pi camera)
- **Firebase project** with Firestore database (for cloud logging)

### 2. Clone Repository

```bash
git clone https://github.com/itznotpk/SSSVC2025_SWDD-Pole_WildGuard-Sentinel.git
cd SSSVC2025_SWDD-Pole_WildGuard-Sentinel
```

### 3. Create Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

**Key Dependencies:**
- `ultralytics` - YOLOv8 framework
- `opencv-python` - Computer vision and camera handling
- `torch` - PyTorch deep learning framework
- `gpiozero` - Raspberry Pi GPIO sensor control
- `firebase-admin` - Firebase/Firestore integration

### 5. Setup Firebase

1. Create a Firebase project at [firebase.google.com](https://firebase.google.com)
2. Enable Firestore database
3. Download service account credentials JSON
4. Place the JSON file in the project root as `wildguardsentinel.json`
5. Update the collection/document names in `main.py` if needed

### 6. Hardware Configuration (For Raspberry Pi)

Update GPIO pin numbers in `main.py`:
```python
PIR_PIN = 17       # GPIO pin for motion sensor
LIGHT_PIN = 27     # GPIO pin for light sensor
```

---

## Running the Application

### Main Detection Loop

Run the full system with dual-model detection:

```bash
python main.py
```

**What happens:**
1. System waits for motion detection from PIR sensor
2. Camera activates on motion trigger
3. Light sensor determines which model to use
4. Frames are processed in real-time
5. Elephants are detected and counted
6. Detection data is sent to Firebase
7. System returns to standby after 30 seconds of no detections
8. Video feed is displayed with annotations

**Exit:** Press `q` to quit the application

---

## Dataset Capture & Preparation

### Capturing Training Images

Use `PictureTaker.py` to collect dataset images:

```bash
# Save images to 'Elephants' folder at 1920x1080 resolution
python PictureTaker.py --imgdir=Elephants --resolution=1920x1080
```

**Controls:**
- **P** - Capture image
- **Q** - Quit

**Output:** Images saved as `Elephants_1.jpg`, `Elephants_2.jpg`, etc.

### Dataset Organization

For training, organize images as:
```
data/
├── images/
│   ├── train/     (90% of images)
│   └── val/       (10% of images)
└── labels/        (YOLO format .txt files)
```

---

## Model Training

### Training YOLOv8n (Google Colab)

Use the provided notebook: `v8_Figure2.ipynb`

**Workflow:**
1. Open in Google Colab
2. Enable T4 GPU runtime
3. Upload `data.zip` (organized training dataset)
4. Configure data.yaml with class names
5. Run training cells
6. Download trained `best.pt` weights

**Key Training Parameters:**
- Model: `yolov8n` (nano - fastest, suitable for edge deployment)
- Epochs: Configurable in notebook
- Image Size: 480px (matches inference size)
- Batch Size: Optimized for GPU memory

**Output:** `best.pt` model weights (rename to `my_model.pt` for main.py)

### Training YOLOv5 (For Night Detection)

Similar process using YOLOv5 framework. Pre-trained weight path specified in `main.py`:
```python
MODEL_NIGHT = str(ROOT / "nv_elephant.pt")
```

---

## Testing & Inference

### Test Models on Images/Videos

Universal testing script with multiple input options:

```bash
# Test on single image
python yolo_detect.py --model=my_model.pt --source=test.jpg --thresh=0.5

# Test on folder of images
python yolo_detect.py --model=my_model.pt --source=test_images/ --thresh=0.5

# Test on video file with recording
python yolo_detect.py --model=my_model.pt --source=video.mp4 --thresh=0.5 --resolution=640x480 --record

# Test on USB camera
python yolo_detect.py --model=my_model.pt --source=usb0 --thresh=0.5 --resolution=640x480

# Test on Raspberry Pi camera
python yolo_detect.py --model=my_model.pt --source=picamera0 --thresh=0.5 --resolution=640x480
```

**Arguments:**
| Argument | Description | Example |
|----------|-------------|---------|
| `--model` | Path to model weights (required) | `my_model.pt` |
| `--source` | Input source (required) | `test.jpg`, `usb0`, `video.mp4` |
| `--thresh` | Confidence threshold | `0.5` (default) |
| `--resolution` | Display resolution | `640x480` |
| `--record` | Record video output | (flag, saves as demo1.avi) |

**Output:** 
- Annotated images/video with bounding boxes
- Confidence scores displayed
- FPS counter
- Optional MP4 recording

---

## API & Integration

### Firebase Firestore Structure

**Collection:** `ElephantDetection`  
**Document:** `Elephas Maximus`

**Data Schema:**
```json
{
  "count": 5,                              // Number of elephants detected
  "last_detected": "2025-12-27T15:30:45Z"  // Timestamp of detection
}
```

### Firebase Integration Code

From `main.py`:
```python
def init_firebase():
    cred = credentials.Certificate("wildguardsentinel.json")
    firebase_admin.initialize_app(cred)
    return firestore.client()

def update_stock(num_elephants: int):
    doc_ref = db.collection("ElephantDetection").document("Elephas Maximus")
    doc_ref.set({
        "count": num_elephants,
        "last_detected": datetime.utcnow()
    })
```

### Real-Time Database Updates

- Detection count is updated immediately upon detection
- Timestamp records UTC time of detection
- Data persists in cloud for long-term analytics
- Can be queried for reporting and visualization

---

## Hardware Requirements

### Minimum Configuration (Daytime Only)

- Raspberry Pi 4+ (2GB RAM minimum)
- USB Webcam (720p+)
- 16GB SD Card
- 5V 3A Power supply

### Recommended Configuration (24/7 Monitoring)

- Raspberry Pi 4 8GB RAM
- Raspberry Pi Camera v2 or better
- PIR Motion Sensor (HC-SR501 or similar)
- Light Sensor (BH1750 or similar)
- High-quality USB hub for GPIO stability
- 32GB SD Card
- Active cooling for continuous operation
- Weatherproof enclosure for outdoor deployment

### GPIO Pinout (Raspberry Pi)

| Component | GPIO Pin | Usage |
|-----------|----------|-------|
| PIR Motion | GPIO 17 | Motion detection trigger |
| Light Sensor | GPIO 27 | Day/night mode switching |

---

## Model Performance

### YOLOv8n (Daytime)
- **Speed:** ~10-15 FPS on Raspberry Pi 4
- **Memory:** ~200MB
- **Accuracy:** Optimized for high confidence in daylight
- **Strengths:** Fast, efficient, ideal for real-time monitoring

### YOLOv5 (Nighttime)
- **Speed:** ~8-12 FPS on Raspberry Pi 4
- **Memory:** ~250MB
- **Accuracy:** Specialized for low-light conditions
- **Strengths:** Better performance in darkness and infrared

---

## Future Enhancements

- [ ] Multi-elephant tracking across frames
- [ ] Behavior classification (walking, running, stationary)
- [ ] Alert system with SMS/email notifications
- [ ] Web dashboard for real-time monitoring
- [ ] Edge deployment optimization for reduced latency
- [ ] Integration with conservation management systems
- [ ] Multi-species detection (lions, rhinos, etc.)
- [ ] Energy consumption monitoring for off-grid deployment

---

## References

- [Ultralytics YOLOv8 Documentation](https://docs.ultralytics.com/)
- [Firebase Firestore Documentation](https://firebase.google.com/docs/firestore)
- [GPIOZero Documentation](https://gpiozero.readthedocs.io/)

---

**WildGuard Sentinel** - Protecting Wildlife Through Intelligence
