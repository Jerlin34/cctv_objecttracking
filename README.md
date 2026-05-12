CCTV Object Tracking
Real-time indoor object tracking system using YOLOv8 + OpenCV with a Flask web dashboard and SQLite history logs.

Features
Detects and tracks indoor objects such as keys, remote-control, wallet, bottle, and glasses-sunglasses.
Maps objects to nearby furniture zones like table/desk, chair, bed, and laptop.
Stores object movement history in SQLite (database/object_history.db).
Provides web routes for live feed, logs, search, and missing-object intelligence.
Includes optional voice-based lookup via scripts/voice_search.py.
Tech Stack
Python
Flask + Flask-CORS
OpenCV
Ultralytics YOLOv8
SQLite
pyttsx3
Project Structure
cctv_objecttracking/
|-- app.py
|-- requirements.txt
|-- database/
|   `-- object_history.db
|-- scripts/
|   |-- detect.py
|   |-- voice_search.py
|   |-- jarvis_ai.py
|   `-- best.pt
|-- templates/
|   |-- index.html
|   |-- login.html
|   |-- feed.html
|   |-- logs.html
|   `-- search.html
|-- static/
|   |-- css/style.css
|   `-- js/main.js
|-- videos/
|   `-- test_video.mp4
`-- datasets/
Requirements
Python 3.10+
Webcam/IP camera/video source
YOLO weights (yolov8n.pt, scripts/best.pt)
Install dependencies:

pip install -r requirements.txt
Run the App
python app.py
Open: 
http://localhost:5000

Default Login
Username: admin | Password: admin123
Username: guest | Password: guest123
Change these in app.py (USERS dictionary) before production use.

Tracking Script (Direct)
If you want to run only detection/tracking logic:

python scripts/detect.py
Voice Search (Optional)
python scripts/voice_search.py
Notes
Database is auto-created by init_db() when the app starts.
Video source is configured in scripts/detect.py (VIDEO_SOURCE).
Current app secret key and default credentials are development-friendly values; rotate both for deployment.
License
MIT license 
