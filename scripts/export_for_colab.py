import sqlite3
import numpy as np
import pickle
from pathlib import Path
import os

DB_PATH = Path("../database/music.db")
OUT_FILE = "music_features_colab.pkl.tmp"
FINAL_FILE = "music_features_colab.pkl"

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

cur.execute("""
SELECT
  t.id,
  af.tempo,
  af.mfcc,
  af.chroma,
  af.pitch_median,
  fe.embedding
FROM tracks t
JOIN audio_features af ON t.id = af.track_id
JOIN fused_embeddings fe ON t.id = fe.track_id
""")

data = {}

for row in cur.fetchall():
    track_id, tempo, mfcc, chroma, pitch, fused = row

    fused = np.frombuffer(fused, np.float32)  # (1536,)
    mfcc = np.frombuffer(mfcc, np.float32).reshape(20, -1).mean(axis=1)
    chroma = np.frombuffer(chroma, np.float32).reshape(12, -1).mean(axis=1)

    vec = np.concatenate([
        fused,
        mfcc,
        chroma,
        np.array([tempo, pitch], dtype=np.float32)
    ])

    data[int(track_id)] = vec.astype(np.float32)

conn.close()

# ✅ SAFE WRITE
with open(OUT_FILE, "wb") as f:
    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

os.replace(OUT_FILE, FINAL_FILE)

print("✔ Export complete")
print("✔ Tracks:", len(data))
print("✔ Feature dim:", next(iter(data.values())).shape[0])
print("✔ File size (MB):", round(Path(FINAL_FILE).stat().st_size / 1024 / 1024, 2))
