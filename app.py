import sqlite3
import os
import random
import hashlib
from datetime import datetime, timedelta
from functools import wraps
from flask import (Flask, Response, render_template, request,
                   jsonify, session, redirect, url_for)
from flask_cors import CORS

from scripts.detect import generate_frames, init_db

app = Flask(__name__)
app.secret_key = "indoorfinder_secret_2024"
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH  = os.path.join(BASE_DIR, "database", "object_history.db")

# Default users — username: sha256(password)
USERS = {
    "admin": hashlib.sha256("admin123".encode()).hexdigest(),
    "guest": hashlib.sha256("guest123".encode()).hexdigest(),
}

MISSING_THRESHOLD_MINUTES = 10   # object missing alert after 10 min


# ═══════════════════════════════════════════════
# AUTH
# ═══════════════════════════════════════════════

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user" not in session:
            if request.path.startswith("/api") or request.is_json:
                return jsonify({"error": "Unauthorized"}), 401
            return redirect(url_for("login_page"))
        return f(*args, **kwargs)
    return decorated


# ═══════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════

def clean_name(name):
    return name.replace("-", " ") if name else ""

def human_time(timestamp):
    try:
        t    = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
        diff = (datetime.now() - t).total_seconds()
        if diff < 60:    return "just now"
        if diff < 3600:  return f"{int(diff/60)} min ago"
        if diff < 86400: return f"{int(diff/3600)} hr ago"
        return t.strftime("%d %b")
    except:
        return ""

def clean_zone(zone):
    return None if not zone or zone == "unknown" else zone

def extract_object(query: str) -> str:
    q = query.lower().strip()
    aliases = {
        "key": "keys", "keys": "keys",
        "remote": "remote-control", "remote control": "remote-control",
        "wallet": "wallet", "bottle": "bottle", "book": "book",
        "glasses": "glasses-sunglasses", "sunglasses": "glasses-sunglasses",
        "earphone": "earphone",
    }
    for phrase, canonical in aliases.items():
        if phrase in q:
            return canonical
    return q

def human_response(obj, zone, time):
    obj = clean_name(obj); zone = clean_zone(zone); tt = human_time(time)
    if not zone:
        return random.choice([
            f"I saw your {obj} {tt}.",
            f"Your {obj} was detected {tt}, but I couldn't find the exact location.",
        ])
    return random.choice([
        f"Your {obj} is on the {zone}.",
        f"I found your {obj} on the {zone}.",
        f"You left your {obj} on the {zone} {tt}.",
        f"I last saw your {obj} on the {zone} {tt}.",
    ])

def smart_response(user_query: str) -> str:
    obj  = extract_object(user_query)
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT object, zone, timestamp, movement FROM object_history "
        "WHERE LOWER(object) LIKE ? ORDER BY id DESC LIMIT 5",
        (f"%{obj.lower()}%",)
    ).fetchall()
    conn.close()
    if not rows:
        return f"I couldn't find your {obj}. It hasn't been detected recently."
    lo, lz, lt, lm = rows[0]
    q = user_query.lower()
    if "when" in q:
        return f"You last saw your {clean_name(lo)} {human_time(lt)}."
    if "move" in q or "before" in q:
        return f"Your {clean_name(lo)} moved: {lm.replace('→',' → ')}" if lm else f"No movement history for your {clean_name(lo)}."
    return human_response(lo, lz, lt)


# ═══════════════════════════════════════════════
# UNIQUE FEATURE: Missing Object Intelligence
# ═══════════════════════════════════════════════

def get_missing_objects():
    conn   = sqlite3.connect(DB_PATH)
    cutoff = (datetime.now() - timedelta(minutes=MISSING_THRESHOLD_MINUTES)).strftime("%Y-%m-%d %H:%M:%S")
    rows   = conn.execute(
        "SELECT object, zone, MAX(timestamp) as last_seen, movement "
        "FROM object_history GROUP BY object ORDER BY last_seen DESC"
    ).fetchall()
    conn.close()
    missing = []
    for obj, zone, last_seen, movement in rows:
        try:
            diff = (datetime.now() - datetime.strptime(last_seen, "%Y-%m-%d %H:%M:%S")).total_seconds() / 60
            if diff > MISSING_THRESHOLD_MINUTES:
                missing.append({
                    "object":      clean_name(obj),
                    "last_zone":   zone or "Unknown",
                    "last_seen":   last_seen,
                    "missing_for": f"{int(diff)} min" if diff < 60 else f"{int(diff/60)} hr {int(diff%60)} min",
                    "minutes_ago": round(diff, 1),
                    "movement":    movement or "—",
                })
        except: pass
    return missing

def get_object_summary():
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT object, COUNT(*) as total, MIN(timestamp) as first_seen, "
        "MAX(timestamp) as last_seen, GROUP_CONCAT(DISTINCT zone) as zones "
        "FROM object_history GROUP BY object ORDER BY total DESC"
    ).fetchall()
    conn.close()
    result = []
    for obj, total, first_seen, last_seen, zones in rows:
        zone_list = [z for z in (zones or "").split(",") if z and z != "unknown"]
        result.append({
            "object": clean_name(obj), "total": total,
            "first_seen": first_seen, "last_seen": last_seen,
            "zones": zone_list, "last_human": human_time(last_seen),
        })
    return result


# ═══════════════════════════════════════════════
# AUTH ROUTES
# ═══════════════════════════════════════════════

@app.route("/login", methods=["GET"])
def login_page():
    if "user" in session:
        return redirect(url_for("index"))
    return render_template("login.html")

@app.route("/login", methods=["POST"])
def login_post():
    data    = request.get_json() or {}
    uname   = data.get("username", "").strip().lower()
    pw_hash = hashlib.sha256(data.get("password", "").encode()).hexdigest()
    if uname in USERS and USERS[uname] == pw_hash:
        session["user"] = uname
        return jsonify({"ok": True, "user": uname})
    return jsonify({"ok": False, "error": "Invalid username or password"}), 401

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login_page"))

@app.route("/api/session")
def api_session():
    if "user" in session:
        return jsonify({"logged_in": True, "user": session["user"]})
    return jsonify({"logged_in": False})


# ═══════════════════════════════════════════════
# PAGE ROUTES
# ═══════════════════════════════════════════════

@app.route("/")
@login_required
def index():
    return render_template("index.html")

@app.route("/feed")
@login_required
def feed():
    return render_template("feed.html")

@app.route("/find")
@login_required
def find():
    return render_template("search.html")

@app.route("/logs")
@login_required
def logs():
    return render_template("logs.html")


# ═══════════════════════════════════════════════
# API ROUTES
# ═══════════════════════════════════════════════

@app.route("/video_feed")
@login_required
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/api/logs")
@login_required
def get_logs():
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT object, zone, track_id, timestamp, movement "
        "FROM object_history ORDER BY id DESC LIMIT 100"
    ).fetchall()
    conn.close()
    return jsonify([{
        "object": clean_name(r[0]), "zone": r[1],
        "track_id": r[2], "timestamp": r[3], "movement": r[4],
    } for r in rows])

@app.route("/api/missing")
@login_required
def api_missing():
    return jsonify(get_missing_objects())

@app.route("/api/summary")
@login_required
def api_summary():
    return jsonify(get_object_summary())

@app.route("/api/heatmap")
@login_required
def api_heatmap():
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT strftime('%H', timestamp) as hour, COUNT(*) as cnt "
        "FROM object_history GROUP BY hour ORDER BY hour"
    ).fetchall()
    conn.close()
    hours = {str(i).zfill(2): 0 for i in range(24)}
    for h, c in rows:
        if h: hours[h] = c
    return jsonify(hours)

@app.route("/search", methods=["POST"])
@login_required
def search():
    data = request.get_json() or {}
    q    = data.get("object", "").strip()
    return jsonify({"message": smart_response(q) if q else "Please enter something."})

@app.route("/voice_search", methods=["POST"])
@login_required
def voice_search():
    data = request.get_json() or {}
    txt  = data.get("text", "").strip()
    return jsonify({"message": smart_response(txt) if txt else "I didn't catch that."})


# ═══════════════════════════════════════════════
# RUN
# ═══════════════════════════════════════════════
if __name__ == "__main__":
    init_db()
    app.run(debug=True, host="0.0.0.0", port=5000)