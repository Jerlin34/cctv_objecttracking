import cv2
import sqlite3
import time
import os
from datetime import datetime
from ultralytics import YOLO

# ================= CONFIG =================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH  = os.path.join(BASE_DIR, "database", "object_history.db")

# ================= MODELS =================
custom_model = YOLO(os.path.join(BASE_DIR, "scripts", "best.pt"))
coco_model   = YOLO("yolov8n.pt")

# ================= CAMERA =================
#VIDEO_SOURCE = 0
#VIDEO_SOURCE = "http://192.168.230.110:8080/video"
VIDEO_SOURCE = "http://192.168.0.146:8080/video"



COOLDOWN_SEC = 1.5
DEMO_MODE = True

# ================= OBJECT FILTER =================
ALLOWED_OBJECTS = [
    "keys", "remote-control", "wallet",
    "bottle", "book", "earphone", "glasses-sunglasses"
]

# Keep demo output stable by focusing on the most reliable classes.
DEMO_ALLOWED_OBJECTS = {
    "keys",
    "remote-control",
    "wallet",
    "bottle",
    "earphone",
}

# Class-wise confidence tuning:
# lower for hard tiny objects (keys), higher for noisy classes (glasses/remote).
CUSTOM_MIN_CONF = {
    "keys": 0.32,
    "wallet": 0.18,
    "earphone": 0.08,
    "glasses-sunglasses": 0.65,
    "remote-control": 0.45,
}

COCO_OBJECT_MAP = {
    "bottle": "bottle",
    "book": "book",
}

COCO_OBJECT_MIN_CONF = {
    "bottle": 0.18,
    "book": 0.30,
}

MIN_SIZES = {
    "glasses-sunglasses": (40, 20),
    "remote-control":     (28, 18),
    "keys":               (20, 10),
    "wallet":             (20, 10),
    "bottle":             (15, 40),
    "book":               (30, 25),
    "earphone":           (8, 8),
}

# ================= FURNITURE CLASSES =================
FURNITURE_CLASSES = {
    "chair", "bed", "dining table", "laptop"
}

# Q2 FIX: Real human-readable zone names — no "Zone A/B/C/D" nonsense
ZONE_LABELS = {
    "dining table": "table/desk",
    "chair":        "chair",
    "bed":          "bed",
    "laptop":       "laptop",
}

COCO_IGNORE_CLASSES = {
    "bench", "tv", "cell phone", "remote", "mouse",
    "keyboard", "book", "bottle", "cup", "clock", "vase",
    "scissors", "toothbrush", "hair drier", "tie", "backpack",
    "handbag", "suitcase",
}

FURNITURE_MIN_CONF = {
    "chair": 0.10,
    "dining table": 0.14,
    "bed": 0.38,
    "laptop": 0.35,
}

FURNITURE_MAX_AREA_RATIO = {
    "chair": 0.70,
    "dining table": 0.92,
    "bed": 0.75,
    "laptop": 0.95,
}
MAX_ZONE_DISTANCE = 300

# ================= DATABASE =================
def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS object_history (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            object    TEXT,
            zone      TEXT,
            track_id  INTEGER,
            timestamp TEXT,
            movement  TEXT
        )
    """)
    conn.commit()
    conn.close()

def log_object(obj, zone, track_id, movement):
    conn = sqlite3.connect(DB_PATH)
    ts   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn.execute(
        "INSERT INTO object_history (object, zone, track_id, timestamp, movement) "
        "VALUES (?, ?, ?, ?, ?)",
        (obj, zone, track_id, ts, movement)
    )
    conn.commit()
    conn.close()

# ================= TRACKING =================
track_memory        = {}
movement_history    = {}
last_known_location = {}
track_class_scores  = {}
next_id             = 0
furniture_cache     = []
furniture_cache_ttl = 0
FURNITURE_CACHE_MAX = 20

def get_track_id(cx, cy):
    global next_id
    best_tid, best_dist = None, float("inf")
    for tid, (px, py) in track_memory.items():
        d = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
        if d < 90 and d < best_dist:
            best_dist = d
            best_tid  = tid
    if best_tid is not None:
        track_memory[best_tid] = (cx, cy)
        return best_tid
    next_id += 1
    track_memory[next_id] = (cx, cy)
    return next_id

def update_movement(track_id, zone):
    if track_id not in movement_history:
        movement_history[track_id] = []
    h = movement_history[track_id]
    if not h or h[-1] != zone:
        h.append(zone)
    if len(h) > 6:
        movement_history[track_id] = h[-6:]
    return " → ".join(movement_history[track_id])

# ================= LOCATION =================
def get_location(obj_box, furniture_boxes):
    """
    Returns the furniture name (chair, bed, dining table etc.)
    or "unknown" if no furniture is near the object.
    NO grid zone fallback — only real furniture names.
    """
    if not furniture_boxes:
        return "unknown"

    ox1, oy1, ox2, oy2 = obj_box
    ocx = (ox1 + ox2) // 2
    ocy = (oy1 + oy2) // 2

    # Smaller furniture boxes first so giant boxes don't dominate.
    furniture_sorted = sorted(
        furniture_boxes,
        key=lambda item: (item[1][2] - item[1][0]) * (item[1][3] - item[1][1])
    )

    # Strategy 1: lower 60% of object center is INSIDE furniture box
    obj_check_y = oy1 + (oy2 - oy1) * 0.6
    for fname, (fx1, fy1, fx2, fy2) in furniture_sorted:
        if fx1 <= ocx <= fx2 and fy1 <= obj_check_y <= fy2:
            return ZONE_LABELS.get(fname, fname)

    # Strategy 2: overlap area ratio >= 2%
    best_zone, best_ratio = None, 0.0
    for fname, (fx1, fy1, fx2, fy2) in furniture_sorted:
        iw = max(0, min(ox2, fx2) - max(ox1, fx1))
        ih = max(0, min(oy2, fy2) - max(oy1, fy1))
        ia = iw * ih
        obj_area = max((ox2 - ox1) * (oy2 - oy1), 1)
        ratio = ia / obj_area
        if ratio > best_ratio:
            best_ratio = ratio
            best_zone = ZONE_LABELS.get(fname, fname)
    if best_zone and best_ratio >= 0.18:
        return best_zone

    # Strategy 3: nearest furniture with a distance cap
    # to avoid assigning a far-away wrong location.
    nearest_zone, min_dist = None, float("inf")
    for fname, (fx1, fy1, fx2, fy2) in furniture_sorted:
        fcx = (fx1 + fx2) // 2
        fcy = (fy1 + fy2) // 2
        d = ((ocx - fcx) ** 2 + (ocy - fcy) ** 2) ** 0.5
        if d < min_dist:
            min_dist     = d
            nearest_zone = ZONE_LABELS.get(fname, fname)
    if nearest_zone and min_dist <= MAX_ZONE_DISTANCE:
        return nearest_zone

    # If furniture exists but object is far → unknown
    return "unknown"

# ================= ALERTS =================
def check_alert(obj, zone):
    if "keys" in obj and "table" in zone:
        print(f"⚠️  ALERT: Keys left on {zone}!")
    if "remote" in obj and "chair" in zone:
        print(f"📺  ALERT: Remote on {zone}")
    if "glasses" in obj and "chair" in zone:
        print(f"👓  ALERT: Glasses on {zone} — careful!")


def normalize_object_label(cls, conf, x1, y1, x2, y2):
    w = x2 - x1
    h = y2 - y1
    area = w * h

    # Earphone bundles are sometimes predicted as "keys" in low light.
    # Promote only large, low-confidence keys-like boxes to earphone.
    if cls == "keys" and conf < 0.50 and area > 4500 and (w > 120 or h > 120):
        return "earphone"

    # Custom model often flips keys <-> remote-control.
    # On keyboard/laptop scenes, tiny "remote-control" is usually keys.
    aspect = w / max(h, 1)
    if cls == "remote-control" and conf < 0.70 and w < 70 and h < 30 and area < 1600 and aspect < 2.8:
        return "keys"
    return cls


def stabilize_class(track_id, cls, conf):
    """
    Temporal smoothing: avoid class flicker (keys <-> remote-control)
    for the same tracked object across adjacent frames.
    """
    # For demo stability, do not smooth between keys and earphone.
    # Their shapes overlap in cluttered scenes and smoothing can lock the wrong label.
    if cls in {"keys", "earphone"}:
        return cls

    scores = track_class_scores.setdefault(track_id, {})
    for k in list(scores.keys()):
        scores[k] *= 0.85
        if scores[k] < 0.05:
            del scores[k]
    scores[cls] = scores.get(cls, 0.0) + conf
    return max(scores, key=scores.get)

# ================= DRAW =================
def draw_box(frame, x1, y1, x2, y2, label, color, conf=None):
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    text = f"{label} {conf:.0%}" if conf is not None else label
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
    cv2.putText(frame, text, (x1 + 3, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

# ================= FRAME PROCESSOR =================
def process_frame(frame, last_logged):
    global furniture_cache, furniture_cache_ttl
    frame_h, frame_w = frame.shape[:2]

    # ── Step 1: Detect furniture ────────────────────────────────
    results_coco = coco_model(frame, conf=0.12, verbose=False)
    current_furniture = []
    for r in results_coco:
        for box in r.boxes:
            cls = coco_model.names[int(box.cls[0])]
            if cls in COCO_IGNORE_CLASSES or cls not in FURNITURE_CLASSES:
                continue
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w = max(1, x2 - x1)
            h = max(1, y2 - y1)
            area_ratio = (w * h) / max(1, frame_w * frame_h)
            if conf < FURNITURE_MIN_CONF.get(cls, 0.14):
                continue
            if area_ratio > FURNITURE_MAX_AREA_RATIO.get(cls, 0.92):
                continue
            if cls == "laptop":
                aspect = w / max(h, 1)
                if area_ratio < 0.08 or aspect < 1.1:
                    continue

            pad = 24
            x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
            x2, y2 = min(frame_w, x2 + pad), min(frame_h, y2 + pad)
            current_furniture.append((cls, (x1, y1, x2, y2)))
            draw_box(frame, x1, y1, x2, y2, ZONE_LABELS.get(cls, cls), (200, 130, 0))

    if current_furniture:
        furniture_cache     = current_furniture
        furniture_cache_ttl = FURNITURE_CACHE_MAX
        print(f"[Furniture] {[f[0] for f in current_furniture]}")
    elif furniture_cache_ttl > 0:
        furniture_cache_ttl -= 1
        current_furniture    = furniture_cache
        print(f"[Furniture] cache TTL={furniture_cache_ttl}")
    else:
        current_furniture = []
        print("[Furniture] no memory yet")

    # ── Step 2: Detect objects ──────────────────────────────────
    # Run a lower global conf, then filter by class thresholds.
    results_custom = custom_model(frame, conf=0.08, verbose=False)

    detections = []
    for r in results_custom:
        for box in r.boxes:
            cls  = custom_model.names[int(box.cls[0])]
            conf = float(box.conf[0])
            if conf < CUSTOM_MIN_CONF.get(cls, 0.35):
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append((cls, conf, x1, y1, x2, y2))

    # best.pt has no "bottle" class; use COCO for bottle/book objects.
    for r in results_coco:
        for box in r.boxes:
            coco_cls = coco_model.names[int(box.cls[0])]
            mapped = COCO_OBJECT_MAP.get(coco_cls)
            if not mapped:
                continue
            conf = float(box.conf[0])
            if conf < COCO_OBJECT_MIN_CONF.get(coco_cls, 0.30):
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append((mapped, conf, x1, y1, x2, y2))

    # ── Step 3: Locate and log ──────────────────────────────────
    for cls, conf, x1, y1, x2, y2 in detections:
        cx, cy   = (x1 + x2) // 2, (y1 + y2) // 2
        track_id = get_track_id(cx, cy)
        cls = normalize_object_label(cls, conf, x1, y1, x2, y2)
        cls = stabilize_class(track_id, cls, conf)
        if cls not in ALLOWED_OBJECTS:
            continue
        if DEMO_MODE and cls not in DEMO_ALLOWED_OBJECTS:
            continue

        obj_w = x2 - x1
        obj_h = y2 - y1
        min_w, min_h = MIN_SIZES.get(cls, (25, 15))
        if obj_w < min_w or obj_h < min_h:
            print(f"⏭️  Skipped tiny detection: {cls} ({obj_w}x{obj_h}px)")
            continue

        location = get_location((x1, y1, x2, y2), current_furniture)

        # Keep last known zone only for brief dropouts.
        if location != "unknown":
            last_known_location[track_id] = location
        else:
            if track_id in last_known_location:
                location = last_known_location[track_id]

        movement = update_movement(track_id, location)

        key = (cls, track_id)
        now = time.time()
        if now - last_logged.get(key, 0) > COOLDOWN_SEC:
            log_object(cls, location, track_id, movement)
            last_logged[key] = now
            print(f"✅ {cls} → {location} | ID:{track_id} | conf:{conf:.0%}")

        check_alert(cls, location)
        draw_box(frame, x1, y1, x2, y2,
                 f"{cls} | {location} | #{track_id}", (0, 210, 0), conf)

    return frame

# ================= STREAM =================
def generate_frames():
    init_db()
    last_logged = {}

    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print("❌ Cannot open camera:", VIDEO_SOURCE)
        return

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    print("🚀 Stream started:", VIDEO_SOURCE)

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue

        frame = process_frame(frame, last_logged)

        _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
               + buffer.tobytes() + b"\r\n")
