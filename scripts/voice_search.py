import sqlite3
import os
import pyttsx3
import speech_recognition as sr
from scripts.jarvis_ai import jarvis_reply

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH  = os.path.join(BASE_DIR, "database", "object_history.db")


def speak(text):
    try:
        engine = pyttsx3.init()
        engine.setProperty("rate", 160)
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print("TTS Error:", e)


def listen_for_query():
    r = sr.Recognizer()

    try:
        with sr.Microphone() as source:
            print("🎤 Listening...")
            r.adjust_for_ambient_noise(source, duration=1)
            audio = r.listen(source, timeout=5)

        text = r.recognize_google(audio)
        print("User:", text)
        return text.lower()

    except Exception as e:
        print("Voice error:", e)
        return ""


def voice_search_flow():
    speak("Yes, how can I help you?")

    query = listen_for_query()

    if not query:
        speak("I didn't hear anything")
        return {"error": "No voice"}

    response = jarvis_reply(query)

    print("Jarvis:", response)
    speak(response)

    return {"response": response}