import sqlite3
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH  = os.path.join(BASE_DIR, "database", "object_history.db")

def get_last_seen(obj):
    conn = sqlite3.connect(DB_PATH)

    row = conn.execute(
        "SELECT zone, timestamp FROM object_history "
        "WHERE LOWER(object) LIKE ? ORDER BY id DESC LIMIT 1",
        (f"%{obj}%",)
    ).fetchone()

    conn.close()
    return row


def get_movement(obj):
    conn = sqlite3.connect(DB_PATH)

    rows = conn.execute(
        "SELECT zone FROM object_history "
        "WHERE LOWER(object) LIKE ? ORDER BY id DESC LIMIT 5",
        (f"%{obj}%",)
    ).fetchall()

    conn.close()

    if not rows:
        return None

    return " → ".join([r[0] for r in reversed(rows)])


def jarvis_reply(query):
    query = query.lower()

    objects = ["keys", "remote-control", "wallet", "bottle", "book"]

    obj = next((o for o in objects if o in query), None)

    if not obj:
        return "I didn't understand which object you're asking about."

    if "where" in query:
        last = get_last_seen(obj)

        if not last:
            return f"I couldn't find your {obj}"

        movement = get_movement(obj)

        return f"Your {obj} is in the {last[0]}. Last seen at {last[1]}. Movement: {movement}"

    if "when" in query:
        last = get_last_seen(obj)

        if not last:
            return f"No recent record for {obj}"

        return f"You last used {obj} at {last[1]}"

    if "move" in query or "before" in query:
        movement = get_movement(obj)

        if not movement:
            return f"No movement data for {obj}"

        return f"{obj} moved like this: {movement}"

    return f"{obj} was last seen in {get_last_seen(obj)[0]}"