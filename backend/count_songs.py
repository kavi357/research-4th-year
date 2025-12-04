import sqlite3
from pathlib import Path

# Path to your database
DB_PATH = Path(__file__).resolve().parent.parent / "database" / "music.db"

def count_tracks():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # Count total tracks
    cur.execute("SELECT COUNT(*) FROM tracks")
    total = cur.fetchone()[0]

    print(f"🎵 Total tracks in database: {total}")

    # Optional: list track IDs and titles
    cur.execute("SELECT id, title FROM tracks ORDER BY id")
    rows = cur.fetchall()
    for track_id, title in rows:
        print(f"  ID={track_id} → {title}")

    conn.close()

if __name__ == "__main__":
    count_tracks()
