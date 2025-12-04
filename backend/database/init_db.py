import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parents[2] / "database" / "music.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

schema = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS tracks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    file_path TEXT NOT NULL,
    duration REAL,
    dataset TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS audio_features (
    track_id INTEGER PRIMARY KEY,
    tempo REAL,
    mfcc BLOB,
    chroma BLOB,
    -- pitch-related fields (CREPE)
    pitch_times BLOB,     -- times (float32 array)
    pitch_freqs BLOB,     -- frequencies (float32 array), 0.0 where no pitch
    pitch_conf BLOB,      -- confidences (float32 array)
    pitch_median REAL,
    FOREIGN KEY(track_id) REFERENCES tracks(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS embeddings (
    track_id INTEGER PRIMARY KEY,
    model TEXT NOT NULL,
    embedding BLOB NOT NULL,
    dim INTEGER NOT NULL,
    FOREIGN KEY(track_id) REFERENCES tracks(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS faiss_index_meta (
    id INTEGER PRIMARY KEY,
    model TEXT NOT NULL,
    index_path TEXT NOT NULL,
    dim INTEGER NOT NULL,
    total_vectors INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS yamnet_embeddings (
    track_id INTEGER PRIMARY KEY,
    embedding BLOB NOT NULL,
    dim INTEGER NOT NULL,
    FOREIGN KEY(track_id) REFERENCES tracks(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS fused_embeddings (
    track_id INTEGER PRIMARY KEY,
    embedding BLOB NOT NULL,
    dim INTEGER NOT NULL,
    FOREIGN KEY(track_id) REFERENCES tracks(id) ON DELETE CASCADE
);

"""

def initialize_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.executescript(schema)
    conn.commit()
    conn.close()
    print("🎉 Database initialized at:", DB_PATH)

if __name__ == "__main__":
    initialize_db()
