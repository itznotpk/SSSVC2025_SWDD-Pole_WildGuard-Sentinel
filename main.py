import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import cv2
import torch
import time
from pathlib import Path
from ultralytics import YOLO  # YOLOv8
from gpiozero import MotionSensor, DigitalInputDevice

###################################################
#Firebase 007 part
from google.protobuf.timestamp_pb2 import Timestamp
from datetime import datetime
import firebase_admin
from firebase_admin import credentials, firestore

# Initialize Firebase
def init_firebase():
    cred = credentials.Certificate("wildguardsentinel.json")
    firebase_admin.initialize_app(cred)
    return firestore.client()

db = init_firebase()

# Firestore update
def update_stock(num_elephants: int):
    doc_ref = db.collection("ElephantDetection").document("Elephas Maximus")
    doc_ref.set({
        "count": num_elephants,                        # overwrite with detected elephants count
        "last_detected": datetime.utcnow() # save latest detection time
    })
    print(f"[✓] Firestore updated → Elephants: {num_elephants}")
####################################################

# === GPIO Devices ===
PIR_PIN = 17
LIGHT_PIN = 27
pir = MotionSensor(PIR_PIN)
light_sensor = DigitalInputDevice(LIGHT_PIN)

# === Load Models ===
FILE = Path(file).resolve()
ROOT = FILE.parents[0]

# YOLOv8 model for DAY
MODEL_DAY = str(ROOT / "my_model.pt")
model_day = YOLO(MODEL_DAY)

# Map YOLOv8 class names {name -> id}
name2id_day = {v: k for k, v in model_day.names.items()}
print("Class mapping (YOLOv8):", name2id_day)

# YOLOv5 model for NIGHT
MODEL_NIGHT = str(ROOT / "nv_elephant.pt")
model_night = torch.hub.load(str(ROOT), 'custom', path=MODEL_NIGHT, source='local')

# === Camera ===
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

# === Main Loop ===
while True:
    print("💤 Waiting for PIR trigger...")
    pir.wait_for_motion()  # blocks until motion detected
    print("⚡ Motion detected!")

    # Choose model based on light sensor
    use_day_model = light_sensor.value == 0  # 0=LOW , light present
    if use_day_model:
        model_type = "yolov8"
        model_label = "Day Model (YOLOv8)"
        print("☀ Using YOLOv8 Daytime model")
    else:
        model_type = "yolov5"
        model_label = "Night Model (YOLOv5)"
        print("🌙 Using YOLOv5 Nighttime model")

    no_detection_timer = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if model_type == "yolov8":
            results = model_day(frame, imgsz=480, verbose=False)
            annotated = results[0].plot()
            detections = results[0].boxes

            if "Elephas Maximus" in name2id_day:
                elephants = [
                    b for b in detections
                    if int(b.cls[0]) == name2id_day["Elephas Maximus"]
                ]
            else:
                elephants = []

        else:  # YOLOv5
            results = model_night(frame, size=480)
            annotated = results.render()[0].copy()  # <-- make writable
            detections = results.pandas().xyxy[0]

            # Get correct class name from the model
            v5_class_names = list(model_night.names.values())
            elephant_name = None
            for name in v5_class_names:
                if "elephant" in name.lower():
                    elephant_name = name
                    break

            if elephant_name is None:
                print("⚠ Warning: YOLOv5 model has no class containing 'elephant'")
                elephants = []
            else:
                elephants = detections[
                    (detections['name'] == elephant_name) & (detections['confidence'] > 0.25)
                ]

        # Overlay which model is active
        cv2.putText(
            annotated,
            model_label,
            (10, 30),                    # x, y position
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,                           # font scale
            (0, 255, 255),                 # yellow color (BGR)
            2                               # thickness
        )

        # Overlay number of elephants detected
        cv2.putText(
            annotated,
            f"Elephants: {len(elephants)}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),  # red color
            2
        )

        if len(elephants) > 0:
            print(f"🐘 Elephants detected: {len(elephants)}")
            no_detection_timer = time.time()
            update_stock(len(elephants))
        else:
            if time.time() - no_detection_timer > 30:
                print("😴 No detection, going standby...")
                break

        cv2.imshow("Elephant Detection", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
