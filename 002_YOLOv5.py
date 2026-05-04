import cv2
import time

def gstreamer_pipeline(
    sensor_id=0,
    capture_width=640,
    capture_height=480,
    display_width=640,
    display_height=480,
    framerate=30,
    flip_method=0,
):
    return (
        f"v4l2src device=/dev/video{sensor_id} ! "
        f"video/x-raw, width={capture_width}, height={capture_height}, framerate={framerate}/1 ! "
        f"videoconvert ! "
        f"video/x-raw, width={display_width}, height={display_height} ! appsink"
    )

cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)

if not cap.isOpened():
    print("❌ Không mở được USB camera")
    exit()

prev = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # FPS
    now = time.time()
    fps = 1/(now - prev) if prev != 0 else 0
    prev = now

    # Flip (tuỳ chọn)
    frame = cv2.flip(frame, 1)

    cv2.putText(frame, f"FPS: {int(fps)}", (20,40),
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0,255,0), 2)

    cv2.imshow("Jetson USB Cam", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()