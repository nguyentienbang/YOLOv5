import cv2
import time
import imutils
import numpy as np

from datetime import datetime
from yoloDet import YoloTRT

# ==========================================
# LOAD TENSORRT MODEL
# ==========================================

model = YoloTRT(
    library="yolov5/build_1/libmyplugins.so",
    engine="yolov5/build_1/yolov5n.engine",
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
# COLORS
# ==========================================

COLORS = {
    "person": (0, 0, 255),
    "car": (255, 0, 0),
    "motorcycle": (0, 255, 255),
    "bus": (0, 255, 0),
    "truck": (255, 0, 255)
}

# ==========================================
# CAMERA
# ==========================================

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 416)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 416)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# ==========================================
# WARMUP
# ==========================================

print("Warming up TensorRT...")

ret, frame = cap.read()

if ret:

    frame = imutils.resize(frame, width=416)

    for i in range(5):
        model.Inference(frame)

print("Warmup completed.")

# ==========================================
# ROI FUNCTION
# ==========================================

def region_of_interest(img):

    height = img.shape[0]

    polygons = np.array([
        [
            (0, height),
            (416, height),
            (300, 250),
            (120, 250)
        ]
    ])

    mask = np.zeros_like(img)

    cv2.fillPoly(mask, polygons, 255)

    masked = cv2.bitwise_and(img, mask)

    return masked

# ==========================================
# DRAW LINES
# ==========================================

def draw_lines(img, lines):

    if lines is None:
        return

    for line in lines:

        x1, y1, x2, y2 = line.reshape(4)

        cv2.line(
            img,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            5
        )

# ==========================================
# MAIN LOOP
# ==========================================

while True:

    start_time = time.time()

    ret, frame = cap.read()

    if not ret:
        break

    frame = imutils.resize(frame, width=416)

    # ======================================
    # YOLO DETECTION
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
    # PROCESS OBJECTS
    # ======================================

    for obj in detections:

        class_name = obj['class']

        if class_name not in ADAS_CLASSES:
            continue

        x1, y1, x2, y2 = map(int, obj['box'])

        conf = float(obj['conf'])

        color = COLORS.get(class_name, (255, 255, 255))

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            2
        )

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
    # LANE DETECTION
    # ======================================

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blur, 50, 150)

    cropped_edges = region_of_interest(edges)

    lines = cv2.HoughLinesP(
        cropped_edges,
        2,
        np.pi / 180,
        100,
        np.array([]),
        minLineLength=40,
        maxLineGap=5
    )

    draw_lines(frame, lines)

    # ======================================
    # CENTER LINE
    # ======================================

    cv2.line(
        frame,
        (208, 0),
        (208, 416),
        (255, 255, 255),
        1
    )

    # ======================================
    # LANE DEPARTURE WARNING
    # ======================================

    warning = "LANE OK"

    warning_color = (0, 255, 0)

    if lines is None:

        warning = "LANE LOST"

        warning_color = (0, 0, 255)

    # ======================================
    # FPS
    # ======================================

    end_time = time.time()

    fps = 1 / max(end_time - start_time, 0.0001)

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
    # DASHBOARD
    # ======================================

    cv2.rectangle(frame, (0, 0), (270, 280), (0, 0, 0), -1)

    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Latency: {latency:.1f} ms",
        (20, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Cars: {car_count}",
        (20, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 0, 0),
        2
    )

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
        f"Motorcycles: {motorcycle_count}",
        (20, 160),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"STATUS: {status}",
        (20, 200),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        warning,
        (20, 240),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        warning_color,
        2
    )

    # ======================================
    # TIMESTAMP
    # ======================================

    current_time = datetime.now().strftime("%H:%M:%S")

    cv2.putText(
        frame,
        current_time,
        (300, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "JETSON NANO ADAS",
        (150, 400),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    # ======================================
    # SHOW OUTPUT
    # ======================================

    cv2.imshow("ADAS AI Dashboard", frame)

    key = cv2.waitKey(1)

    time.sleep(0.001)

    if key == ord('q'):
        break

# ==========================================
# RELEASE
# ==========================================

cap.release()

cv2.destroyAllWindows()