# PPE Detection for MJPEG Stream from Raspberry Pi
import cv2
import numpy as np
import torch
from ultralytics import YOLO
import logging
from datetime import datetime
import os
import time
import sys

# ----------------------------- User settings -----------------------------
MJPEG_URL = "http://10.120.180.108:8080/?action=stream&dummy=1"  # Your Pi MJPEG stream
MODEL_PATH = "ppe-detection-main/ppe-detection-main/best.pt"     # Path to your PPE model
CONF_THRESHOLD = 0.7                           # keep detections above this
FRAME_WIDTH = 640                              # resize width for faster inference
FRAME_HEIGHT = 360                             # resize height for faster inference
FRAME_SKIP = 1                                 # process every Nth frame (1 == every frame)
BUFFER_SIZE = 1                                # OpenCV buffer size (keep latest frame)
RECONNECT_DELAY_SEC = 3.0                      # wait before reconnect attempts
MAX_RECONNECT_ATTEMPTS = 5                     # how many times to try reconnecting
# -------------------------------------------------------------------------

# ----------------------------- Logging Setup -----------------------------
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_filename = f"{log_dir}/ppe_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger().addHandler(console)

logging.info("Starting PPE Detection for MJPEG Stream")
# -------------------------------------------------------------------------

# ----------------------------- Model Setup -----------------------------
try:
    torch.jit.enable_onednn_fusion(True)
except Exception:
    pass

logging.info("Loading YOLO model from %s", MODEL_PATH)
if not os.path.exists(MODEL_PATH):
    logging.error("Model file not found: %s", MODEL_PATH)
    print(f"Error: Model file not found at {MODEL_PATH}")
    print("Please ensure the model file exists or update MODEL_PATH")
    sys.exit(1)

model = YOLO(MODEL_PATH)
try:
    model.fuse()
except Exception:
    pass
logging.info("Model loaded successfully")
# -------------------------------------------------------------------------

LABELS = {
    0: "Hardhat",
    1: "Mask", 
    2: "NO-Hardhat",
    3: "NO-Mask",
    4: "NO-Safety Vest",
    5: "Person",
    6: "Safety Cone",
    7: "Safety Vest",
    8: "Machinery",
    9: "Vehicle"
}

# ----------------------------- Helper functions -----------------------------
def open_mjpeg_stream(url, buffer_size=1):
    """Open MJPEG stream and configure buffer size. Return cv2.VideoCapture or None."""
    logging.info("Connecting to MJPEG stream: %s", url)
    cap = cv2.VideoCapture(url)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, buffer_size)
    except Exception:
        pass
    return cap

def ensure_open(cap, retries=MAX_RECONNECT_ATTEMPTS, delay=RECONNECT_DELAY_SEC):
    """Ensure cap is opened; try reconnecting several times."""
    attempt = 0
    while attempt <= retries:
        if cap is not None and cap.isOpened():
            return True
        logging.warning("MJPEG stream not open (attempt %d/%d). Retrying in %.1f s...", attempt+1, retries, delay)
        time.sleep(delay)
        cap = open_mjpeg_stream(MJPEG_URL, BUFFER_SIZE)
        attempt += 1
    return False
# -------------------------------------------------------------------------

def main():
    # ----------------------------- Open stream -------------------------
    cap = open_mjpeg_stream(MJPEG_URL, BUFFER_SIZE)
    if not ensure_open(cap):
        logging.error("Failed to open MJPEG stream after %d attempts. Exiting.", MAX_RECONNECT_ATTEMPTS)
        print(f"Error: Could not connect to stream at {MJPEG_URL}")
        print("Please check:")
        print("1. Raspberry Pi is running and accessible")
        print("2. MJPEG streamer is running on the Pi")
        print("3. IP address and port are correct")
        sys.exit(1)

    # warm up: discard a few frames to stabilise
    for _ in range(5):
        cap.read()
    logging.info("MJPEG stream opened and warmed up")
    # -------------------------------------------------------------------------

    # ----------------------------- Main loop ----------------------------------
    frame_count = 0
    last_time = time.time()
    
    print("PPE Detection started. Press 'q' to quit.")
    
    try:
        while True:
            # read raw frame
            ret, frame = cap.read()
            if not ret or frame is None:
                logging.warning("No frame received from MJPEG stream. Attempting reconnect...")
                cap.release()
                cap = open_mjpeg_stream(MJPEG_URL, BUFFER_SIZE)
                if not ensure_open(cap):
                    logging.error("Reconnect failed. Waiting before next attempt...")
                    time.sleep(RECONNECT_DELAY_SEC)
                    continue
                else:
                    logging.info("Reconnected to MJPEG stream.")
                    continue

            frame_count += 1
            # frame skipping to reduce inference load
            if FRAME_SKIP > 1 and (frame_count % FRAME_SKIP != 0):
                continue

            # optional resize to speed up inference
            h, w = frame.shape[:2]
            if (w, h) != (FRAME_WIDTH, FRAME_HEIGHT):
                frame_small = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT), interpolation=cv2.INTER_LINEAR)
            else:
                frame_small = frame

            # Run model inference
            results = model(frame_small)

            detections_count = 0
            detection_summary = {}

            # Process results
            for result in results:
                boxes = getattr(result, "boxes", None)
                if boxes is None:
                    continue

                for box in boxes:
                    xyxy = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])

                    confidence = float(box.conf[0].cpu().numpy()) if hasattr(box.conf, 'cpu') else float(box.conf[0])
                    class_id = int(box.cls[0].cpu().numpy()) if hasattr(box.cls, 'cpu') else int(box.cls[0])

                    if confidence < CONF_THRESHOLD:
                        continue

                    class_name = LABELS.get(class_id, f"Class_{class_id}")

                    # map small coords back to original frame size if we resized
                    if (w, h) != (FRAME_WIDTH, FRAME_HEIGHT):
                        x_scale = w / FRAME_WIDTH
                        y_scale = h / FRAME_HEIGHT
                        x1 = int(x1 * x_scale)
                        x2 = int(x2 * x_scale)
                        y1 = int(y1 * y_scale)
                        y2 = int(y2 * y_scale)

                    detections_count += 1
                    detection_summary[class_name] = detection_summary.get(class_name, 0) + 1

                    # draw box and label (red for violations prefixed 'NO-')
                    color = (0, 0, 255) if class_name.startswith("NO-") else (0, 255, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    label = f"{class_name} {confidence:.2f}"
                    cv2.putText(frame, label, (x1, max(20, y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                    # log detection
                    logging.info("Frame %d - %s (%.2f) bbox [%d,%d,%d,%d]", frame_count, class_name, confidence, x1, y1, x2, y2)

            # per-frame summary log
            if detections_count:
                logging.info("Frame %d summary: %d detections: %s", frame_count, detections_count, detection_summary)

            # Add frame info
            cv2.putText(frame, f"Frame: {frame_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Detections: {detections_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # show frame
            cv2.imshow("PPE Detection (MJPEG Stream)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logging.info("Exit requested by user (q).")
                break

            # optional simple FPS print once per second
            now = time.time()
            if now - last_time >= 1.0:
                last_time = now

    except KeyboardInterrupt:
        logging.info("KeyboardInterrupt received. Exiting.")
    finally:
        try:
            if cap is not None:
                cap.release()
        except Exception:
            pass
        cv2.destroyAllWindows()
        logging.info("Application closed")

if __name__ == "__main__":
    main()