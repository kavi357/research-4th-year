import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parents[2] / "database" / "music.db"

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

cur.execute("""
ALTER TABLE audio_features
ADD COLUMN rhythm_pattern BLOB
""")

conn.commit()
conn.close()

print("✅ rhythm_pattern column added")
