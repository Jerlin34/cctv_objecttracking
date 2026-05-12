# CCTV Object Tracking

Real-time indoor object tracking system using **YOLOv8 + OpenCV** with a Flask web dashboard and SQLite history logs.

---

# Features

* Detects and tracks indoor objects such as:

  * keys
  * remote control
  * wallet
  * bottle
  * glasses/sunglasses

* Maps objects to nearby furniture zones like:

  * table/desk
  * chair
  * bed
  * laptop

* Stores object movement history in SQLite database

* Provides web routes for:

  * live feed
  * logs
  * search
  * missing-object intelligence

* Includes optional voice-based lookup

---

# Tech Stack

* Python
* Flask
* Flask-CORS
* OpenCV
* Ultralytics YOLOv8
* SQLite
* pyttsx3

---

# Project Structure

```plaintext
cctv_objecttracking/
│
├── app.py
├── requirements.txt
├── database/
│   └── object_history.db
│
├── scripts/
│   ├── detect.py
│   ├── voice_search.py
│   ├── jarvis_ai.py
│   └── best.pt
│
├── templates/
│   ├── index.html
│   ├── login.html
│   ├── feed.html
│   ├── logs.html
│   └── search.html
│
├── static/
│   ├── css/style.css
│   └── js/main.js
│
├── videos/
│   └── test_video.mp4
│
└── datasets/
```

---

# Requirements

* Python 3.10+
* Webcam / IP Camera / Video Source
* YOLOv8 weights:

  * yolov8n.pt
  * scripts/best.pt

---

# Installation

```bash
pip install -r requirements.txt
```

---

# Run the Application

```bash
python app.py
```

Open in browser:

```plaintext
http://localhost:5000
```

---

# Default Login

```plaintext
Username: admin
Password: admin123

Username: guest
Password: guest123
```

Change credentials in `app.py` before deployment.

---

# Run Detection Only

```bash
python scripts/detect.py
```

---

# Voice Search (Optional)

```bash
python scripts/voice_search.py
```

---

# Notes

* Database is automatically created using `init_db()`
* Video source is configured in `scripts/detect.py`
* Replace default secret keys before deployment

---

# Future Enhancements

* Live CCTV stream integration
* Mobile notifications
* Cloud database support
* Multi-camera tracking
* Face recognition integration

---

# License

 licensed under the MIT License.
