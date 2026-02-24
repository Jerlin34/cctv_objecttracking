import sqlite3
import pyttsx3

engine = pyttsx3.init()

conn = sqlite3.connect("../database/object_history.db")
cursor = conn.cursor()

obj = input("Enter object name (bottle / cell phone): ").lower()

cursor.execute("""
SELECT zone,timestamp
FROM object_history
WHERE object_name=?
ORDER BY id DESC
LIMIT 1
""",(obj,))

result = cursor.fetchone()

if result:
    zone,time = result
    msg = f"{obj} was last seen in the {zone} at {time}"
    print(msg)
    engine.say(msg)
    engine.runAndWait()
else:
    print("Object not found")
    engine.say("Object not found")
    engine.runAndWait()

conn.close()
