print("🚀 SYSTEM STARTING")

from ultralytics import YOLO
import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort
import sqlite3
from datetime import datetime

# ------------------------
# LOAD YOLO
# ------------------------
model = YOLO("yolov8n.pt")
tracker = DeepSort(max_age=30)

# ------------------------
# CAMERA (REAL TIME)
# ------------------------
cap = cv2.VideoCapture(0)   # webcam

# For video file use:
# cap = cv2.VideoCapture(r"C:\Users\jerli\CCTV_Object_Tracking_Project\videos\test_video.mp4")

if not cap.isOpened():
    print("Camera not opened")
    exit()

print("Camera running...")

# ------------------------
# DATABASE
# ------------------------
conn = sqlite3.connect("../database/object_history.db")
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS object_history(
id INTEGER PRIMARY KEY AUTOINCREMENT,
object_name TEXT,
object_id TEXT,
zone TEXT,
timestamp TEXT)
""")

conn.commit()

# ------------------------
# ZONE FUNCTION
# ------------------------
def get_zone(x):

    if x < 300:
        return "Table Area"
    elif x < 600:
        return "Chair Area"
    else:
        return "Bed Area"

last_zone = {}

# ------------------------
# MAIN LOOP
# ------------------------
while True:

    ret, frame = cap.read()
    if not ret:
        break

    # YOLO DETECTION
    results = model(frame)

    detections = []

    if results[0].boxes is not None:

        for r in results[0].boxes:

            x1, y1, x2, y2 = r.xyxy[0]
            conf = float(r.conf[0])
            cls = int(r.cls[0])
            label = model.names[cls]

            # ⭐ ALLOW BOTTLE + PHONE
            if label not in ["bottle", "cell phone"]:
                continue

            detections.append(([x1, y1, x2-x1, y2-y1], conf, label))

    # DEEPSORT TRACKING
    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:

        if not track.is_confirmed():
            continue

        track_id = track.track_id
        l, t, r, b = track.to_ltrb()
        l, t, r, b = int(l), int(t), int(r), int(b)

        center_x = int((l + r)/2)
        zone = get_zone(center_x)

        # STORE ONLY IF ZONE CHANGED
        if track_id not in last_zone or last_zone[track_id] != zone:

            print(f"{track.det_class} ID {track_id} moved to {zone}")

            last_zone[track_id] = zone

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            cursor.execute("""
            INSERT INTO object_history(object_name,object_id,zone,timestamp)
            VALUES(?,?,?,?)
            """,(track.det_class,str(track_id),zone,timestamp))

            conn.commit()

        # DRAW BOX
        cv2.rectangle(frame,(l,t),(r,b),(0,255,0),2)
        cv2.putText(frame,f"{track.det_class} ID:{track_id}",
                    (l,t-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,(0,255,0),2)

    cv2.imshow("Realtime DeepSORT Tracking",frame)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
conn.close()

print("SYSTEM CLOSED")
