import sqlite3
import numpy as np
from pathlib import Path

from backend.ingest.preprocess import compute_audio_hash

DB_PATH = "database/music.db"

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

cur.execute("""
    SELECT t.id, a.chroma, a.tempo, t.duration
    FROM tracks t
    JOIN audio_features a ON a.track_id = t.id
    WHERE t.audio_hash IS NULL
""")

rows = cur.fetchall()

print(f"🔄 Backfilling {len(rows)} tracks...")

for track_id, chroma_blob, tempo, duration in rows:
    chroma = np.frombuffer(chroma_blob, np.float32).reshape(12, -1)
    audio_hash = compute_audio_hash(chroma, tempo, duration)

    cur.execute(
        "UPDATE tracks SET audio_hash=? WHERE id=?",
        (audio_hash, track_id)
    )

conn.commit()
conn.close()

print("✅ Audio hash backfill complete.")
