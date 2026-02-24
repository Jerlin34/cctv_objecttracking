# 📹 CCTV Object Tracking System

A real-time object tracking and history management system using **YOLOv8** and **DeepSORT**. This system detects specific objects (like bottles and cell phones), tracks them across different zones, and stores their movement history in a database for later retrieval via voice commands.

## 🚀 Features

- **Real-time Detection:** Uses YOLOv8 for high-accuracy object detection.
- **Robust Tracking:** Implements DeepSORT (Simple Online and Realtime Tracking) to maintain object IDs across frames.
- **Zone Monitoring:** Automatically identifies which area (Table, Chair, or Bed) an object is in.
- **Historical Database:** Logs every movement and zone change into an SQLite database.
- **Voice Search:** Retrieve the last seen location of an object using voice feedback.

## 🛠️ Tech Stack

- **Computer Vision:** OpenCV, YOLOv8 (Ultralytics)
- **Tracking:** DeepSORT
- **Database:** SQLite3
- **Voice Synthesis:** pyttsx3
- **Language:** Python

## 📂 Project Structure

```bash
CCTV_Object_Tracking_Project/
├── database/
│   └── object_history.db    # SQLite database for tracking logs
├── scripts/
│   ├── detect.py            # Main tracking script
│   ├── voice_search.py      # Voice-enabled history retriever
│   └── yolov8n.pt           # Pre-trained YOLOv8 model
├── videos/
│   └── test_video.mp4       # Sample video for testing
└── README.md
```

## ⚙️ Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Jerlin34/cctv_objecttracking.git
   cd cctv_objecttracking
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 🚦 Usage

### 1. Real-time Tracking
Run the detection script to start monitoring via webcam or video file:
```bash
python scripts/detect.py
```

### 2. Voice Search
To find the last seen location of an object:
```bash
python scripts/voice_search.py
```

### 3. Web Dashboard (New!)
Experience the stunning real-time prototype:
```bash
python app.py
```
Then open `http://localhost:5000` in your browser.

## 📝 License
This project is licensed under the MIT License.
