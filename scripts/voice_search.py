import sqlite3
import os
import pyttsx3

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "database", "object_history.db")

def speak(text):
    print(text)
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

print("--- CCTV Voice Assistant CLI ---")
query = input("Ask about an object (e.g., 'where is the bottle'): ").lower()

# Basic object name extraction for CLI
keywords = ["bottle", "phone", "cell phone", "person", "chair", "bed"]
obj_name = "unknown"
for k in keywords:
    if k in query:
        obj_name = k
        break

if obj_name == "unknown":
    speak("I'm sorry, I don't know that object.")
else:
    cursor.execute("""
    SELECT zone, timestamp
    FROM object_history
    WHERE object_name LIKE ?
    ORDER BY id DESC
    LIMIT 1
    """, (f"%{obj_name}%",))

    result = cursor.fetchone()

    if result:
        zone, time = result
        speak(f"The {obj_name} was last seen in the {zone} at {time}.")
    else:
        speak(f"I couldn't find any records for {obj_name}.")

conn.close()
