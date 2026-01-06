# intrusion_rtsp.py
import cv2
import numpy as np
import torch
import time
from ultralytics import YOLO
from threading import Thread
from collections import defaultdict, deque
import sys
import logging
from datetime import datetime
import os

# ---------------------- User settings ----------------------
MJPEG_URL = "http://10.120.180.108:8080/?action=stream&dummy=1"  # <-- Your Pi MJPEG stream
MODEL_PATH = "yolo11n.pt"                      # <-- change if your model file has different name
CONFIDENCE = 0.5
THREAD_RECONNECT_DELAY = 2.0   # seconds between reconnect attempts in the capture thread
# -----------------------------------------------------------

# Logging (also prints to console)
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logfile = f"{log_dir}/intrusion_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(filename=logfile, level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger().addHandler(console)
logging.info("Starting Restricted Zone Detector (RTSP edition)")

# ---------------------- Torch / OpenCV tuning ----------------------
torch.jit.enable_onednn_fusion(True)
torch.set_num_threads(4)

cv2.setUseOptimized(True)
cv2.setNumThreads(2)
if cv2.ocl.haveOpenCL():
    cv2.ocl.setUseOpenCL(True)
# ---------------------------------------------------------------

class VideoCaptureThread:
    __slots__ = ['source', 'cap', 'ret', 'frame', 'stopped', 'thread', 'use_ffmpeg']

    def __init__(self, source=0):
        """
        source: either integer (camera index) or string (rtsp/http)
        """
        self.source = source
        self.use_ffmpeg = isinstance(source, str) and (source.startswith("rtsp://") or source.startswith("http://") or source.startswith("https://"))
        self.cap = self._make_capture()
        self.ret, self.frame = self.cap.read() if self.cap is not None else (False, None)
        self.stopped = False
        self.thread = Thread(target=self.update, daemon=True)

    def _make_capture(self):
        try:
            if self.use_ffmpeg:
                cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
            else:
                cap = cv2.VideoCapture(self.source)
            # Reduce internal buffering to keep latency low (best-effort)
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
            return cap
        except Exception as e:
            logging.exception("Failed to create VideoCapture: %s", e)
            return None

    def start(self):
        self.thread.start()
        return self

    def update(self):
        """
        Continuously read frames. If disconnected and source is network stream,
        attempt to reconnect every THREAD_RECONNECT_DELAY seconds.
        """
        while not self.stopped:
            if self.cap is None:
                logging.warning("Capture object is None, attempting to create a new one...")
                self.cap = self._make_capture()
                time.sleep(THREAD_RECONNECT_DELAY)
                continue

            try:
                self.ret, self.frame = self.cap.read()
                if not self.ret or self.frame is None:
                    # If network stream, try reconnecting
                    if self.use_ffmpeg or isinstance(self.source, str):
                        logging.warning("No frame from stream, attempting reconnect...")
                        try:
                            self.cap.release()
                        except Exception:
                            pass
                        self.cap = self._make_capture()
                        time.sleep(THREAD_RECONNECT_DELAY)
                    else:
                        # local camera: brief sleep and continue
                        time.sleep(0.01)
                else:
                    # small sleep to yield CPU (helps with very tight loops)
                    time.sleep(0.001)
            except Exception as e:
                logging.exception("Exception in capture update loop: %s", e)
                try:
                    if self.cap is not None:
                        self.cap.release()
                except Exception:
                    pass
                self.cap = self._make_capture()
                time.sleep(THREAD_RECONNECT_DELAY)

    def read(self):
        return self.ret, self.frame

    def stop(self):
        self.stopped = True
        # Wait for thread to exit
        try:
            self.thread.join(timeout=1.0)
        except Exception:
            pass
        try:
            if self.cap is not None:
                self.cap.release()
        except Exception:
            pass

    def isOpened(self):
        try:
            return self.cap is not None and self.cap.isOpened()
        except Exception:
            return False

class RestrictedZoneDetector:
    __slots__ = ['device', 'model', 'confidence', 'restricted_zones', 'zone_bboxes', 'frame_count', 'fps', 'start_time','track_history',
                  'max_track_points', 'objects_in_zones', 'zone_color', 'safe_color', 'breach_color', 'track_color']

    def __init__(self, model_path=MODEL_PATH, confidence=CONFIDENCE, device=None):
        self.device = device if device else 'cuda' if torch.cuda.is_available() else 'cpu'
        logging.info("Using device: %s", self.device)

        self.model = YOLO(model_path, task='detect')
        if self.device == 'cuda':
            try:
                self.model = self.model.half()
            except Exception:
                pass
        else:
            try:
                self.model = self.model.float()
            except Exception:
                pass
        try:
            self.model.fuse()
        except Exception:
            pass

        self.confidence = confidence
        self.restricted_zones = []
        self.zone_bboxes = []
        self.frame_count = 0
        self.fps = 0
        self.start_time = time.time()
        self.track_history = defaultdict(lambda: deque(maxlen=8))
        self.objects_in_zones = defaultdict(dict)

        self.zone_color = (0, 255, 0)
        self.safe_color = (0, 255, 0)
        self.breach_color = (0, 0, 255)
        self.track_color = (255, 0, 0)

    def add_zone(self, points):
        if len(points) >= 3:
            polygon = np.array(points, np.int32)
            self.restricted_zones.append(polygon)
            min_vals = np.min(polygon, axis=0)
            max_vals = np.max(polygon, axis=0)
            self.zone_bboxes.append((*min_vals, *max_vals))
            return True
        return False

    def clear_zones(self):
        self.restricted_zones.clear()
        self.objects_in_zones.clear()

    def point_in_polygon(self, point, polygon):
        return cv2.pointPolygonTest(polygon, point, False) >= 0

    def check_object_in_zones(self, bbox):
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2
        foot_point = (int(center_x), int(y2))

        breached_zones = []
        for idx, zone in enumerate(self.restricted_zones):
            if cv2.pointPolygonTest(zone, foot_point, False) >= 0:
                breached_zones.append(idx)

        return breached_zones

    def process_frame(self, frame):
        self.frame_count += 1

        # simple FPS calc
        if self.frame_count % 10 == 0:
            now = time.time()
            self.fps = 10.0 / (now - self.start_time) if (now - self.start_time) > 0 else 0.0
            self.start_time = now

        result_frame = frame.copy()

        # Using YOLO track to obtain track ids (requires ultralytics track API)
        try:
            # track(...) will return tracked objects with .boxes.id if tracking available
            results = self.model.track(frame, persist=False, conf=self.confidence, classes=[0])
        except Exception as e:
            logging.exception("Model track error: %s", e)
            return result_frame, []

        # draw zones
        if len(self.restricted_zones) > 0:
            zone_centers = np.mean(np.array(self.restricted_zones), axis=1, dtype=np.int32)
            for i, (zone, center) in enumerate(zip(self.restricted_zones, zone_centers)):
                cv2.polylines(result_frame, [zone], True, self.zone_color, 2)
                cv2.putText(result_frame, f"Zone {i+1}", tuple(center),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.zone_color, 2)

        current_breaches = []
        active_track_ids_in_frame = set()

        # extract boxes, classes, track ids safely
        try:
            boxes_obj = getattr(results[0], "boxes", None)
            if boxes_obj is not None and hasattr(boxes_obj, "xyxy"):
                boxes = boxes_obj.xyxy.cpu().numpy()
                # class array might not exist - handle safely
                class_ids = boxes_obj.cls.cpu().numpy() if hasattr(boxes_obj, "cls") else np.zeros((len(boxes),), dtype=int)
                # tracking ids - if available
                track_ids = boxes_obj.id.int().cpu().numpy() if hasattr(boxes_obj, "id") else np.arange(len(boxes))
            else:
                boxes = []
                class_ids = []
                track_ids = []
        except Exception as e:
            logging.exception("Error extracting boxes: %s", e)
            boxes = []
            class_ids = []
            track_ids = []

        for box, class_id, track_id in zip(boxes, class_ids, track_ids):
            x1, y1, x2, y2 = map(int, box)
            track_id = int(track_id)
            active_track_ids_in_frame.add(track_id)

            if int(class_id) == 0:  # person
                breached_zone_indices = self.check_object_in_zones((x1, y1, x2, y2))
                if breached_zone_indices:
                    # mark history
                    center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                    self.track_history[track_id].append(center)

                    # if first time in this zone for this id -> new alert
                    for zone_idx in breached_zone_indices:
                        is_new = zone_idx not in self.objects_in_zones.get(track_id, {})
                        if is_new:
                            if track_id not in self.objects_in_zones:
                                self.objects_in_zones[track_id] = {}
                            self.objects_in_zones[track_id][zone_idx] = time.time()
                        current_breaches.append((track_id, zone_idx, is_new))

                    # draw breached box and track
                    cv2.rectangle(result_frame, (x1, y1), (x2, y2), self.breach_color, 2)
                    cv2.putText(result_frame, f"ID:{track_id}", (x1, y1 - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.breach_color, 2)

                    pts = self.track_history[track_id]
                    if len(pts) >= 2:
                        for i in range(1, len(pts)):
                            cv2.line(result_frame, pts[i - 1], pts[i], self.breach_color, 2)
                else:
                    # if object leaves zones, clear its record
                    if track_id in self.objects_in_zones and not breached_zone_indices:
                        self.objects_in_zones[track_id] = {}

        # cleanup IDs not in frame
        to_remove = []
        for tid in list(self.objects_in_zones.keys()):
            if tid not in active_track_ids_in_frame:
                to_remove.append(tid)
        for tid in to_remove:
            self.objects_in_zones.pop(tid, None)
            self.track_history.pop(tid, None)

        # show alerts on frame
        alert_y = 30
        for obj_id, zone_idx, is_new in current_breaches:
            text = (f"NEW ALERT: Object ID {obj_id} in Zone {zone_idx+1}"
                    if is_new else f"ALERT: Object ID {obj_id} in Zone {zone_idx+1}")
            cv2.putText(result_frame, text, (10, alert_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.breach_color, 2)
            alert_y += 26

        cv2.putText(result_frame, f"FPS: {self.fps:.1f}", (10, result_frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        return result_frame, current_breaches

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        param['points'].append((x, y))
        param['drawing'] = True
    elif event == cv2.EVENT_MOUSEMOVE:
        param['current_pos'] = (x, y)
    elif event == cv2.EVENT_RBUTTONDOWN:
        if len(param['points']) >= 3:
            param['zones'].append(np.array(param['points'], np.int32))
        param['points'] = []
        param['drawing'] = False

def main():
    detector = RestrictedZoneDetector(model_path=MODEL_PATH, confidence=CONFIDENCE)

    # Use MJPEG stream from Pi - replace with an int to use local camera
    video_source = MJPEG_URL
    cap = VideoCaptureThread(video_source).start()

    # Wait a moment for the stream to initialize
    time.sleep(1.0)

    if not cap.isOpened():
        logging.error("Could not open video source: %s", video_source)
        print("Error: Could not open video source. Check RTSP URL or camera connection.")
        return

    window_name = "Restricted Zone Detection"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    params = {'points': [], 'zones': [], 'drawing': False, 'current_pos': None}
    cv2.setMouseCallback(window_name, mouse_callback, params)

    # Zone setup UI
    print("=== ZONE SETUP MODE ===")
    print("Left-click to place zone vertices.")
    print("Right-click to finish a zone (must have at least 3 points).")
    print("Press 'c' to clear all zones.")
    print("Press 's' to save zones and start detection.")
    setup_mode = True

    while setup_mode:
        ret, frame = cap.read()
        if not ret or frame is None:
            time.sleep(0.02)
            continue

        setup_frame = frame.copy()

        for zone in params['zones']:
            cv2.polylines(setup_frame, [zone], True, (0, 255, 0), 2)

        if params['points']:
            for point in params['points']:
                cv2.circle(setup_frame, point, 5, (255, 0, 0), -1)
            for i in range(1, len(params['points'])):
                cv2.line(setup_frame, params['points'][i-1], params['points'][i], (255, 0, 0), 2)
            if params['drawing'] and params['current_pos']:
                cv2.line(setup_frame, params['points'][-1], params['current_pos'], (255, 0, 0), 2)

        cv2.putText(setup_frame, "ZONE SETUP MODE", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(setup_frame, "Left-click: Add vertex | Right-click: Finish zone", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(setup_frame, "Press 'c': Clear zones | Press 's': Save and start", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow(window_name, setup_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            for zone in params['zones']:
                detector.add_zone(zone)
            setup_mode = False
        elif key == ord('c'):
            params['zones'] = []
            params['points'] = []
            detector.clear_zones()
        elif key == 27:
            # ESC to exit
            cap.stop()
            cv2.destroyAllWindows()
            return

    print("=== DETECTION MODE ===")
    print("Press 'q' to quit.")

    breach_record = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            processed_frame, breaches = detector.process_frame(frame)

            if breaches:
                current_time = time.strftime("%Y-%m-%d %H:%M:%S")
                for obj_id, zone_idx, is_new in breaches:
                    if is_new:
                        breach_record.append({
                            'timestamp': current_time,
                            'object_id': obj_id,
                            'zone': zone_idx + 1
                        })

            cv2.imshow(window_name, processed_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        logging.info("KeyboardInterrupt received, shutting down.")
    finally:
        cap.stop()
        cv2.destroyAllWindows()

        if breach_record:
            print("\n=== BREACH REPORT ===")
            for i, breach in enumerate(breach_record):
                print(f"{i+1}. Time: {breach['timestamp']}, Object ID: {breach['object_id']}, Zone: {breach['zone']}")

if __name__ == '__main__':
    main()
