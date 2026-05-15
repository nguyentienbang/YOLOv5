import cv2
import time
import imutils
from datetime import datetime

from yoloDet import YoloTRT

# ==========================================
# LOAD TENSORRT MODEL
# ==========================================

model = YoloTRT(
    library="yolov5/build/libmyplugins.so",
    engine="yolov5/build/yolov5s.engine",
    conf=0.5,
    yolo_ver="v5"
)

# ==========================================
# ADAS CLASSES
# ==========================================

ADAS_CLASSES = [
    "person",
    "car",
    "motorcycle",
    "bus",
    "truck"
]

# ==========================================
# CLASS COLORS
# ==========================================

COLORS = {
    "person": (0, 0, 255),
    "car": (255, 0, 0),
    "motorcycle": (0, 255, 255),
    "bus": (0, 255, 0),
    "truck": (255, 0, 255)
}

# ==========================================
# OPEN CAMERA
# ==========================================

cap = cv2.VideoCapture(0)

# ==========================================
# CAMERA OPTIMIZATION
# ==========================================

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# reduce webcam latency
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# ==========================================
# WARMUP TENSORRT
# ==========================================

print("Warming up TensorRT...")

ret, frame = cap.read()

if ret:

    frame = imutils.resize(frame, width=640)

    for i in range(10):
        model.Inference(frame)

print("Warmup completed.")

# ==========================================
# MAIN LOOP
# ==========================================

while True:

    # ======================================
    # TIMER START
    # ======================================

    start_time = time.time()

    # ======================================
    # READ FRAME
    # ======================================

    ret, frame = cap.read()

    if not ret:
        print("Camera error")
        break

    # ======================================
    # RESIZE FRAME
    # ======================================

    frame = imutils.resize(frame, width=640)

    # ======================================
    # YOLO INFERENCE
    # ======================================

    detections, t = model.Inference(frame)

    # ======================================
    # OBJECT COUNTERS
    # ======================================

    person_count = 0
    car_count = 0
    motorcycle_count = 0
    bus_count = 0
    truck_count = 0

    # ======================================
    # PROCESS DETECTIONS
    # ======================================

    for obj in detections:

        class_name = obj['class']

        # FILTER ONLY ADAS CLASSES
        if class_name not in ADAS_CLASSES:
            continue

        # GET OBJECT INFO
        x1, y1, x2, y2 = obj['box']
        conf = obj['conf']

        # COLOR
        color = COLORS.get(class_name, (255, 255, 255))

        # DRAW BOUNDING BOX
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # LABEL
        label = f"{class_name} {conf:.2f}"

        cv2.putText(
            frame,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2
        )

        # COUNT OBJECTS
        if class_name == "person":
            person_count += 1

        elif class_name == "car":
            car_count += 1

        elif class_name == "motorcycle":
            motorcycle_count += 1

        elif class_name == "bus":
            bus_count += 1

        elif class_name == "truck":
            truck_count += 1

    # ======================================
    # TIMER END
    # ======================================

    end_time = time.time()

    # ======================================
    # FPS
    # ======================================

    fps = 1 / (end_time - start_time)

    # ======================================
    # LATENCY
    # ======================================

    latency = (end_time - start_time) * 1000

    # ======================================
    # SYSTEM STATUS
    # ======================================

    status = "SAFE"

    total_vehicles = (
        car_count +
        motorcycle_count +
        bus_count +
        truck_count
    )

    if total_vehicles >= 5:
        status = "WARNING"

    if total_vehicles >= 8:
        status = "DANGER"

    # ======================================
    # DASHBOARD BACKGROUND
    # ======================================

    cv2.rectangle(frame, (0, 0), (260, 230), (0, 0, 0), -1)

    # ======================================
    # FPS DISPLAY
    # ======================================

    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    # ======================================
    # LATENCY DISPLAY
    # ======================================

    cv2.putText(
        frame,
        f"Latency: {latency:.1f} ms",
        (20, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    # ======================================
    # RESOLUTION DISPLAY
    # ======================================

    cv2.putText(
        frame,
        "Resolution: 640x480",
        (20, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),
        2
    )

    # ======================================
    # OBJECT COUNTS
    # ======================================

    cv2.putText(
        frame,
        f"Persons: {person_count}",
        (20, 130),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 255),
        2
    )

    cv2.putText(
        frame,
        f"Cars: {car_count}",
        (20, 155),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 0, 0),
        2
    )

    cv2.putText(
        frame,
        f"Motorcycles: {motorcycle_count}",
        (20, 180),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2
    )

    # ======================================
    # STATUS DISPLAY
    # ======================================

    status_color = (0, 255, 0)

    if status == "WARNING":
        status_color = (0, 255, 255)

    elif status == "DANGER":
        status_color = (0, 0, 255)

    cv2.putText(
        frame,
        f"STATUS: {status}",
        (20, 210),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        status_color,
        2
    )

    # ======================================
    # CENTER LINE
    # ======================================

    cv2.line(frame, (320, 0), (320, 480), (255, 255, 255), 1)

    # ======================================
    # TIMESTAMP
    # ======================================

    current_time = datetime.now().strftime("%H:%M:%S")

    cv2.putText(
        frame,
        current_time,
        (500, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    # ======================================
    # DEVICE NAME
    # ======================================

    cv2.putText(
        frame,
        "JETSON NANO ADAS",
        (400, 460),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    # ======================================
    # SHOW OUTPUT
    # ======================================

    cv2.imshow("ADAS AI Dashboard", frame)

    # ======================================
    # EXIT KEY
    # ======================================

    key = cv2.waitKey(1)

    if key == ord('q'):
        break

# ==========================================
# RELEASE
# ==========================================

cap.release()

cv2.destroyAllWindows()