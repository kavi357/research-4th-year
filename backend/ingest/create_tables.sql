


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
