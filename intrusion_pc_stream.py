print("🚀 Intrusion detection script started!")
import cv2
import numpy as np
import torch
import time
from ultralytics import YOLO
from threading import Thread
from collections import defaultdict, deque
import logging
from datetime import datetime
import os
import sys

# ----------------------------- User settings -----------------------------
MJPEG_URL = "http://10.120.180.108:8080/?action=stream&dummy=1"  # Your Pi MJPEG stream
MODEL_PATH = "intrusion-main/intrusion-main/yolo11n.pt"         # Path to your intrusion model
CONFIDENCE = 0.5
THREAD_RECONNECT_DELAY = 2.0   # seconds between reconnect attempts in the capture thread

# Performance settings to reduce lag
FRAME_SKIP = 2                 # Process every Nth frame (2 = every other frame)
RESIZE_WIDTH = 640             # Resize frame width for faster processing
RESIZE_HEIGHT = 360            # Resize frame height for faster processing
BUFFER_SIZE = 1                # Keep only latest frame in buffer
# -------------------------------------------------------------------------

# Logging setup
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logfile = f"{log_dir}/intrusion_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(filename=logfile, level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger().addHandler(console)
logging.info("Starting Restricted Zone Detector (MJPEG Stream)")

# Performance optimizations
torch.jit.enable_onednn_fusion(True)
torch.set_num_threads(4)
cv2.setUseOptimized(True)
cv2.setNumThreads(2)
if cv2.ocl.haveOpenCL():
    cv2.ocl.setUseOpenCL(True)

class VideoCaptureThread:
    __slots__ = ["source", "cap", "ret", "frame", "stopped", "thread", "use_mjpeg"]

    def __init__(self, source=0):
        self.source = source
        self.use_mjpeg = isinstance(source, str) and source.startswith("http")
        self.cap = self._make_capture()
        self.ret, self.frame = self.cap.read() if self.cap is not None else (False, None)
        self.stopped = False
        self.thread = Thread(target=self.update, daemon=True)

    def _make_capture(self):
        try:
            cap = cv2.VideoCapture(self.source)
            # Aggressive buffer settings to reduce latency
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, BUFFER_SIZE)
                # Set FPS if possible (though MJPEG streams may ignore this)
                cap.set(cv2.CAP_PROP_FPS, 30)
                # Disable any internal threading that might cause buffering
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
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
        """Continuously read frames with reconnection for network streams."""
        frame_count = 0
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
                    if self.use_mjpeg:
                        logging.warning("No frame from MJPEG stream, attempting reconnect...")
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
                    frame_count += 1
                    # Skip frames to reduce processing load and latency
                    if FRAME_SKIP > 1 and (frame_count % FRAME_SKIP != 0):
                        continue
                    
                    # Resize frame for faster processing
                    if self.frame is not None:
                        h, w = self.frame.shape[:2]
                        if w != RESIZE_WIDTH or h != RESIZE_HEIGHT:
                            self.frame = cv2.resize(self.frame, (RESIZE_WIDTH, RESIZE_HEIGHT), 
                                                  interpolation=cv2.INTER_LINEAR)
                    
                    # Minimal sleep to prevent CPU overload
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

# ---------------- Restriction detector ----------------
class RestrictedZoneDetector:
    """Enhanced RestrictedZoneDetector with YOLO person detection."""

    def __init__(self, model_path=MODEL_PATH, confidence=CONFIDENCE):
        self.confidence = confidence
        self.zones = []
        self.track_history = defaultdict(lambda: deque(maxlen=8))
        self.objects_in_zones = defaultdict(dict)
        self.frame_count = 0
        
        # Load YOLO model
        if not os.path.exists(model_path):
            logging.error("Model file not found: %s", model_path)
            print(f"Error: Model file not found at {model_path}")
            sys.exit(1)
            
        logging.info("Loading YOLO model from %s", model_path)
        self.model = YOLO(model_path)
        try:
            self.model.fuse()
        except Exception:
            pass
        logging.info("Model loaded successfully")

    def add_zone(self, points):
        if points:
            self.zones.append(list(points))

    def clear_zones(self):
        self.zones = []
        self.objects_in_zones.clear()

    def point_in_polygon(self, point, polygon):
        """Check if point is inside polygon using OpenCV."""
        polygon_np = np.array(polygon, dtype=np.int32)
        return cv2.pointPolygonTest(polygon_np, point, False) >= 0

    def check_object_in_zones(self, bbox):
        """Check which zones the object's foot point intersects."""
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2
        foot_point = (int(center_x), int(y2))  # Bottom center of bbox

        breached_zones = []
        for idx, zone in enumerate(self.zones):
            if self.point_in_polygon(foot_point, zone):
                breached_zones.append(idx)

        return breached_zones

    def update(self, frame):
        """Process frame and return detections and breaches."""
        self.frame_count += 1
        
        # Run YOLO detection for persons (class 0) with optimized settings
        try:
            # Use smaller input size and disable verbose output for speed
            results = self.model.track(frame, persist=True, conf=self.confidence, 
                                     classes=[0], verbose=False, imgsz=320)
        except Exception as e:
            logging.exception("Model inference error: %s", e)
            return [], []

        detections = []
        current_breaches = []
        active_track_ids = set()

        # Process results
        try:
            boxes_obj = getattr(results[0], "boxes", None)
            if boxes_obj is not None and hasattr(boxes_obj, "xyxy"):
                boxes = boxes_obj.xyxy.cpu().numpy()
                track_ids = boxes_obj.id.int().cpu().numpy() if hasattr(boxes_obj, "id") else np.arange(len(boxes))
                
                for box, track_id in zip(boxes, track_ids):
                    x1, y1, x2, y2 = map(int, box)
                    track_id = int(track_id)
                    active_track_ids.add(track_id)
                    
                    detection = {
                        'bbox': (x1, y1, x2, y2),
                        'track_id': track_id,
                        'class': 'person'
                    }
                    detections.append(detection)
                    
                    # Check zone breaches
                    breached_zone_indices = self.check_object_in_zones((x1, y1, x2, y2))
                    if breached_zone_indices:
                        # Update track history (limit to recent points for performance)
                        center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                        self.track_history[track_id].append(center)
                        
                        # Check for new breaches
                        for zone_idx in breached_zone_indices:
                            is_new = zone_idx not in self.objects_in_zones.get(track_id, {})
                            if is_new:
                                if track_id not in self.objects_in_zones:
                                    self.objects_in_zones[track_id] = {}
                                self.objects_in_zones[track_id][zone_idx] = time.time()
                                logging.info("NEW BREACH: Object ID %d entered Zone %d", track_id, zone_idx + 1)
                            current_breaches.append((track_id, zone_idx, is_new))
                    else:
                        # Clear zones if object left all zones
                        if track_id in self.objects_in_zones:
                            self.objects_in_zones[track_id] = {}
                            
        except Exception as e:
            logging.exception("Error processing detections: %s", e)

        # Cleanup inactive tracks (less frequent cleanup for performance)
        if self.frame_count % 30 == 0:  # Only cleanup every 30 frames
            to_remove = []
            for tid in list(self.objects_in_zones.keys()):
                if tid not in active_track_ids:
                    to_remove.append(tid)
            for tid in to_remove:
                self.objects_in_zones.pop(tid, None)
                self.track_history.pop(tid, None)

        return detections, current_breaches

    def draw_zones(self, frame):
        """Draw defined zones on the frame."""
        for i, zone in enumerate(self.zones):
            if len(zone) >= 2:
                pts = np.array(zone, dtype=np.int32)
                cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
                # Semi-transparent fill
                overlay = frame.copy()
                cv2.fillPoly(overlay, [pts], color=(0, 255, 0))
                cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
                
                # Zone label
                if len(zone) > 0:
                    center = np.mean(pts, axis=0).astype(int)
                    cv2.putText(frame, f"Zone {i+1}", tuple(center),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
# ------------------------------------------------------------------

def main():
    detector = RestrictedZoneDetector(model_path=MODEL_PATH, confidence=CONFIDENCE)

    # === CHANGE THIS LINE ===
    stream_url = MJPEG_URL  # Your Pi MJPEG stream
    cap = VideoCaptureThread(stream_url).start()

    # Wait for stream to initialize
    time.sleep(1.0)

    if not cap.isOpened():
        print("❌ Could not open MJPEG stream. Check Pi IP and port.")
        print(f"Attempted to connect to: {stream_url}")
        print("Please verify:")
        print("1. Raspberry Pi is running and accessible")
        print("2. MJPEG streamer is running on the Pi")
        print("3. IP address and port are correct")
        return

    window_name = "Restricted Zone Detection"
    cv2.namedWindow(window_name)

    # mouse callback params
    params = {"points": [], "drawing": False, "current_pos": None}

    def mouse_callback(event, x, y, flags, p):
        """Simple mouse callback to collect polygon points.

        Left click: add point. Right click: finish polygon (zone) and add to detector.
        Move: update current_pos for visual feedback.
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            p["points"].append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN:
            # finish current polygon
            if p["points"]:
                detector.add_zone(p["points"])
                p["points"] = []
                print(f"Zone {len(detector.zones)} added with {len(p['points'])} points")
        elif event == cv2.EVENT_MOUSEMOVE:
            p["current_pos"] = (x, y)

    cv2.setMouseCallback(window_name, mouse_callback, params)

    print("=== ZONE SETUP MODE ===")
    print("Left-click to place zone vertices.")
    print("Right-click to finish a zone (must have at least 3 points).")
    print("Press 'c' to clear all zones.")
    print("Press 's' to start detection.")
    print("Press 'q' to quit.")
    
    setup_mode = True
    breach_record = []

    # Main loop
    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.05)
                continue

            if setup_mode:
                # Zone setup mode
                setup_frame = frame.copy()
                detector.draw_zones(setup_frame)

                # Draw current polygon being created
                pts = params["points"]
                if pts:
                    for i in range(len(pts)):
                        cv2.circle(setup_frame, pts[i], 4, (0, 255, 0), -1)
                        if i > 0:
                            cv2.line(setup_frame, pts[i - 1], pts[i], (0, 255, 0), 2)
                    if params.get("current_pos"):
                        cv2.line(setup_frame, pts[-1], params["current_pos"], (0, 255, 0), 1)

                # Instructions
                cv2.putText(setup_frame, "ZONE SETUP MODE", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(setup_frame, "Left-click: Add vertex | Right-click: Finish zone", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.putText(setup_frame, "Press 'c': Clear | 's': Start detection | 'q': Quit", (10, 85),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                cv2.imshow(window_name, setup_frame)
                
            else:
                # Detection mode
                detections, breaches = detector.update(frame)
                detector.draw_zones(frame)
                
                # Draw detections
                for detection in detections:
                    x1, y1, x2, y2 = detection['bbox']
                    track_id = detection['track_id']
                    
                    # Check if this object is in any zone
                    in_zone = any(track_id == breach[0] for breach in breaches)
                    color = (0, 0, 255) if in_zone else (0, 255, 0)  # Red if in zone, green otherwise
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"ID:{track_id}", (x1, y1 - 8),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    # Draw track history if in zone
                    if in_zone and track_id in detector.track_history:
                        pts = detector.track_history[track_id]
                        if len(pts) >= 2:
                            for i in range(1, len(pts)):
                                cv2.line(frame, pts[i - 1], pts[i], color, 2)

                # Show breach alerts
                alert_y = 30
                for obj_id, zone_idx, is_new in breaches:
                    text = (f"NEW ALERT: ID {obj_id} in Zone {zone_idx+1}"
                           if is_new else f"ALERT: ID {obj_id} in Zone {zone_idx+1}")
                    cv2.putText(frame, text, (10, alert_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    alert_y += 25
                    
                    # Record new breaches
                    if is_new:
                        breach_record.append({
                            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                            'object_id': obj_id,
                            'zone': zone_idx + 1
                        })

                # Show detection info
                cv2.putText(frame, f"Detections: {len(detections)} | FPS: ~{30//max(1,FRAME_SKIP)}", 
                           (10, frame.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame, f"Zones: {len(detector.zones)} | Frame: {detector.frame_count}", 
                           (10, frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                cv2.imshow(window_name, frame)

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break
            elif key == ord("c"):
                # clear all zones
                detector.clear_zones()
                params["points"] = []
                print("All zones cleared")
            elif key == ord("s") and setup_mode:
                setup_mode = False
                print("=== DETECTION MODE ===")
                print("Press 'q' to quit.")
                
    except KeyboardInterrupt:
        logging.info("KeyboardInterrupt received, shutting down.")
    finally:
        cap.stop()
        cv2.destroyAllWindows()

        if breach_record:
            print("\n=== BREACH REPORT ===")
            for i, breach in enumerate(breach_record):
                print(f"{i+1}. Time: {breach['timestamp']}, Object ID: {breach['object_id']}, Zone: {breach['zone']}")
        else:
            print("\nNo breaches detected during this session.")

if __name__ == "__main__":
    main()

