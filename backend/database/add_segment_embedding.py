import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parents[2] / "database" / "music.db"

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

cur.execute("""
CREATE TABLE IF NOT EXISTS segment_embeddings (
    track_id    INTEGER NOT NULL,
    segment_idx INTEGER NOT NULL,
    embedding   BLOB NOT NULL,
    dim         INTEGER NOT NULL,
    start_time  REAL,
    end_time    REAL,
    PRIMARY KEY (track_id, segment_idx),
    FOREIGN KEY (track_id) REFERENCES tracks(id) ON DELETE CASCADE
)
""")

conn.commit()
conn.close()

print("✅ segment_embeddings table added successfully")
