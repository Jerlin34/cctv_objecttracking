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
VIDEO_SOURCE = "http://192.168.0.146:8080/video"

COOLDOWN_SEC = 5

# ================= OBJECT FILTER =================
ALLOWED_OBJECTS = [
    "keys", "remote-control", "wallet",
    "bottle", "book", "earphone", "glasses-sunglasses"
]

# Minimum size filters — only size, NO aspect ratio for glasses
MIN_SIZES = {
    "glasses-sunglasses": (30, 15),
    "remote-control":     (20, 30),
    "keys":               (25, 15),
    "wallet":             (30, 15),
    "bottle":             (15, 40),
    "book":               (30, 25),
    "earphone":           (15, 15),
}

# ================= FURNITURE =================
# NOTE: COCO does NOT have a "desk" class. The closest is
# "dining table" — which COCO uses for ALL flat surfaces
# including desks, coffee tables, study tables.
# So "dining table" from COCO = desk/table in your home.
FURNITURE_CLASSES = {
    "chair", "bed", "dining table", "couch", "sofa", "bench"
}

# Map COCO class → human-readable zone label
# "dining table" covers desks, tables, study tables in COCO
ZONE_LABELS = {
    "dining table": "desk/table",   # covers desks too — COCO has no "desk" class
    "chair":        "chair",
    "bed":          "bed",
    "couch":        "sofa/couch",
    "sofa":         "sofa/couch",
    "bench":        "bench",
}

# COCO classes to fully ignore — never treat as furniture
COCO_IGNORE_CLASSES = {
    "laptop", "tv", "cell phone", "remote", "mouse",
    "keyboard", "book", "bottle", "cup", "clock",
    "vase", "scissors", "toothbrush", "hair drier",
    "tie", "backpack", "handbag", "suitcase",
    "desk",   # not a real COCO class but block just in case
}

# Per-class confidence and box-size guardrails for furniture detections.
# These reduce false zones such as full-frame "bed" hallucinations.
FURNITURE_MIN_CONF = {
    "chair": 0.10,
    "dining table": 0.10,
    "couch": 0.12,
    "sofa": 0.12,
    "bench": 0.12,
    "bed": 0.30,
}

FURNITURE_MAX_AREA_RATIO = {
    "chair": 0.75,
    "dining table": 0.90,
    "couch": 0.90,
    "sofa": 0.90,
    "bench": 0.80,
    "bed": 0.78,
}

# ══════════════════════════════════════════════════════════════
# PERMANENT FURNITURE MEMORY
# Once we see furniture in a room, we never fully forget it.
# This fixes "unknown" when furniture briefly leaves the frame.
# permanent_furniture: persists forever until replaced by new
# furniture detection. furniture_cache: short-term (30 frames).
# ══════════════════════════════════════════════════════════════
permanent_furniture = []   # never cleared — last real furniture seen

# Class-level location memory across all objects
class_location_memory = {}   # e.g. {"glasses-sunglasses": "desk/table"}
scene_default_zone = None    # last dominant room zone (largest furniture seen)

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
next_id             = 0
furniture_cache     = []
furniture_cache_ttl = 0
FURNITURE_CACHE_MAX = 150   # ~5 seconds at 30fps — much longer than before

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
    if not furniture_boxes:
        return "unknown"

    ox1, oy1, ox2, oy2 = obj_box
    ocx = (ox1 + ox2) // 2
    ocy = (oy1 + oy2) // 2

    # Check smaller furniture first so a giant false "bed" box
    # does not override a valid nearby "chair/table".
    furniture_sorted = sorted(
        furniture_boxes,
        key=lambda item: (item[1][2] - item[1][0]) * (item[1][3] - item[1][1])
    )

    # Strategy 1: lower 70% of object box center inside furniture
    obj_check_y = oy1 + (oy2 - oy1) * 0.7
    for fname, (fx1, fy1, fx2, fy2) in furniture_sorted:
        if fx1 <= ocx <= fx2 and fy1 <= obj_check_y <= fy2:
            return ZONE_LABELS.get(fname, fname)

    # Strategy 2: best overlap by object-coverage ratio
    best_zone, best_coverage = None, 0.0
    obj_area = max(1, (ox2 - ox1) * (oy2 - oy1))
    for fname, (fx1, fy1, fx2, fy2) in furniture_sorted:
        iw = max(0, min(ox2, fx2) - max(ox1, fx1))
        ih = max(0, min(oy2, fy2) - max(oy1, fy1))
        ia = iw * ih
        coverage = ia / obj_area
        if coverage > best_coverage:
            best_coverage = coverage
            best_zone = ZONE_LABELS.get(fname, fname)
    if best_zone and best_coverage >= 0.20:
        return best_zone

    # Strategy 3: nearest furniture (no hard threshold)
    # A fixed max distance can force "unknown" even when furniture exists.
    # Choosing nearest avoids that.
    nearest_zone, min_dist = None, float("inf")
    for fname, (fx1, fy1, fx2, fy2) in furniture_sorted:
        fcx = (fx1 + fx2) // 2
        fcy = (fy1 + fy2) // 2
        d = ((ocx - fcx) ** 2 + (ocy - fcy) ** 2) ** 0.5
        if d < min_dist:
            min_dist     = d
            nearest_zone = ZONE_LABELS.get(fname, fname)
    if nearest_zone:
        return nearest_zone

    return "unknown"


def get_dominant_zone(furniture_boxes):
    if not furniture_boxes:
        return None
    fname, (x1, y1, x2, y2) = max(
        furniture_boxes,
        key=lambda item: (item[1][2] - item[1][0]) * (item[1][3] - item[1][1])
    )
    return ZONE_LABELS.get(fname, fname)

# ================= DETECTION FILTER =================
def is_valid_detection(cls, x1, y1, x2, y2):
    """
    Only size check — no aspect ratio filter for glasses.
    Aspect ratio was causing all glasses detections to be skipped.
    We trust the custom model's confidence (0.65+) to reject false positives.
    """
    w = x2 - x1
    h = y2 - y1
    min_w, min_h = MIN_SIZES.get(cls, (25, 15))
    if w < min_w or h < min_h:
        print(f"⏭  Skipped {cls}: too small ({w}×{h}px)")
        return False
    return True

# ================= ALERTS =================
def check_alert(obj, zone):
    if "keys" in obj and "table" in zone:
        print(f"⚠️  Keys on {zone}!")
    if "remote" in obj and "chair" in zone:
        print(f"📺  Remote on {zone}")
    if "glasses" in obj and "chair" in zone:
        print(f"👓  Glasses on {zone} — careful!")

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
    global furniture_cache, furniture_cache_ttl, permanent_furniture, scene_default_zone
    frame_h, frame_w = frame.shape[:2]

    # ── Step 1: Detect REAL furniture only ───────────────────
    # Lower conf to 0.12 so tables/desks at odd angles are caught
    results_coco = coco_model(frame, conf=0.12, verbose=False)
    current_furniture = []
    for r in results_coco:
        for box in r.boxes:
            cls = coco_model.names[int(box.cls[0])]
            if cls in COCO_IGNORE_CLASSES:
                continue
            if cls not in FURNITURE_CLASSES:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            w = max(1, x2 - x1)
            h = max(1, y2 - y1)
            area_ratio = (w * h) / max(1, frame_w * frame_h)
            if conf < FURNITURE_MIN_CONF.get(cls, 0.12):
                continue
            if area_ratio > FURNITURE_MAX_AREA_RATIO.get(cls, 0.92):
                continue

            pad = 25
            x1 = max(0, x1 - pad);  y1 = max(0, y1 - pad)
            x2 = min(frame_w, x2 + pad); y2 = min(frame_h, y2 + pad)
            current_furniture.append((cls, (x1, y1, x2, y2)))
            draw_box(frame, x1, y1, x2, y2,
                     ZONE_LABELS.get(cls, cls), (200, 130, 0))

    if current_furniture:
        # New furniture seen — update all three memory levels
        furniture_cache     = current_furniture
        furniture_cache_ttl = FURNITURE_CACHE_MAX
        permanent_furniture = current_furniture          # ← never cleared
        scene_default_zone  = get_dominant_zone(current_furniture)
        print(f"[Furniture] {[f[0] for f in current_furniture]}")
    elif furniture_cache_ttl > 0:
        furniture_cache_ttl -= 1
        current_furniture    = furniture_cache           # use short-term cache
        if scene_default_zone is None:
            scene_default_zone = get_dominant_zone(current_furniture)
        print(f"[Furniture] cache TTL={furniture_cache_ttl}")
    else:
        # Cache expired — fall back to permanent memory
        # This is why zone was "unknown" before: cache hit 0 and we gave up
        if permanent_furniture:
            current_furniture = permanent_furniture
            if scene_default_zone is None:
                scene_default_zone = get_dominant_zone(current_furniture)
            print(f"[Furniture] using permanent memory: {[f[0] for f in permanent_furniture]}")
        else:
            current_furniture = []
            print("[Furniture] no memory yet")

    # ── Step 2: Detect custom objects ────────────────────────
    # Confidence 0.60 — slightly lower so glasses are not missed,
    # but still high enough to reject noise
    results_custom = custom_model(frame, conf=0.60, verbose=False)

    detections = []
    for r in results_custom:
        for box in r.boxes:
            cls  = custom_model.names[int(box.cls[0])]
            conf = float(box.conf[0])
            if cls not in ALLOWED_OBJECTS:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Only size check — no aspect ratio kill for glasses
            if not is_valid_detection(cls, x1, y1, x2, y2):
                continue

            detections.append((cls, conf, x1, y1, x2, y2))

    # ── Step 3: Locate and log ───────────────────────────────
    for cls, conf, x1, y1, x2, y2 in detections:
        cx, cy   = (x1 + x2) // 2, (y1 + y2) // 2
        track_id = get_track_id(cx, cy)
        location = get_location((x1, y1, x2, y2), current_furniture)

        # 3-level location memory
        if location != "unknown":
            last_known_location[track_id] = location
            class_location_memory[cls]    = location
        else:
            if track_id in last_known_location:
                location = last_known_location[track_id]
                print(f"↩  Using track memory: {cls} → {location}")
            elif cls in class_location_memory:
                location = class_location_memory[cls]
                print(f"↩  Using class memory: {cls} → {location}")
            elif scene_default_zone:
                location = scene_default_zone
                print(f"↩  Using scene fallback: {cls} → {location}")
            else:
                print(f"❓  Genuinely unknown: {cls} (no furniture seen yet)")

        movement = update_movement(track_id, location)

        key = (cls, track_id)
        now = time.time()
        if now - last_logged.get(key, 0) > COOLDOWN_SEC:
            log_object(cls, location, track_id, movement)
            last_logged[key] = now
            print(f"✅ {cls} → {location} | ID:{track_id} | {conf:.0%}")

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
