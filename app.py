from flask import Flask, render_template, Response, jsonify, request
import cv2
import sqlite3
import os
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from datetime import datetime
import threading

app = Flask(__name__)

# Load model and tracker
model = YOLO("scripts/yolov8n.pt")
tracker = DeepSort(max_age=30)

def get_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return None
    return cap

cap = get_camera()

# Database path
DB_PATH = "database/object_history.db"

def get_zone(x):
    if x < 300: return "Table Area"
    elif x < 600: return "Chair Area"
    else: return "Bed Area"

last_zone = {}

def gen_frames():
    global last_zone, cap
    while True:
        if cap is None or not cap.isOpened():
            cap = get_camera()
            if cap is None:
                # Still failing, wait a bit or yield a placeholder?
                # For now, let's just break or sleep
                import time
                time.sleep(1)
                continue

        success, frame = cap.read()
        if not success:
            print("Failed to read frame, retrying camera...")
            cap.release()
            cap = None
            continue
        else:
            # YOLO detection
            results = model(frame, verbose=False)
            detections = []
            
            if results[0].boxes is not None:
                for r in results[0].boxes:
                    x1, y1, x2, y2 = r.xyxy[0]
                    conf = float(r.conf[0])
                    cls = int(r.cls[0])
                    label = model.names[cls]
                    if label in ["bottle", "cell phone", "person"]:
                        detections.append(([x1, y1, x2-x1, y2-y1], conf, label))

            # Tracking
            tracks = tracker.update_tracks(detections, frame=frame)
            for track in tracks:
                if not track.is_confirmed(): continue
                track_id = track.track_id
                l, t, r, b = map(int, track.to_ltrb())
                center_x = (l + r) // 2
                zone = get_zone(center_x)

                if track_id not in last_zone or last_zone[track_id] != zone:
                    last_zone[track_id] = zone
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    with sqlite3.connect(DB_PATH) as conn:
                        cursor = conn.cursor()
                        cursor.execute("INSERT INTO object_history(object_name,object_id,zone,timestamp) VALUES(?,?,?,?)",
                                     (track.det_class, str(track_id), zone, timestamp))
                        conn.commit()

                # Draw components
                cv2.rectangle(frame, (l, t), (r, b), (124, 77, 255), 2)
                cv2.putText(frame, f"{track.det_class} #{track_id}", (l, t-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (124, 77, 255), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/logs')
def get_logs():
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM object_history ORDER BY id DESC LIMIT 10")
        rows = cursor.fetchall()
        return jsonify([dict(ix) for ix in rows])

@app.route('/api/search')
def search():
    name = request.args.get('name', '').lower()
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT zone, timestamp FROM object_history WHERE object_name LIKE ? ORDER BY id DESC LIMIT 1", (f"%{name}%",))
        result = cursor.fetchone()
        if result:
            zone = result[0].lower().replace(" area", "")
            return jsonify({
                "found": True,
                "message": f"on {zone} at {result[1]}"
            })
        return jsonify({
            "found": False,
            "message": "not found in recent history"
        })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
