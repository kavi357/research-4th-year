import sqlite3
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path

DB_PATH = "database/music.db"
OUT_PATH = "output/similarity_pairs_fusion.npz"
Path("output").mkdir(exist_ok=True)

def summarize_features(row):
    _, fused_blob, tempo, mfcc_blob, chroma_blob, pitch_blob = row

    fused = np.frombuffer(fused_blob, dtype=np.float32)

    mfcc = np.frombuffer(mfcc_blob, dtype=np.float32).reshape(20, -1)
    chroma = np.frombuffer(chroma_blob, dtype=np.float32).reshape(12, -1)
    pitch = np.frombuffer(pitch_blob, dtype=np.float32) if pitch_blob else np.array([])

    return {
        "fused": fused,
        "mfcc": mfcc.mean(axis=1),
        "chroma": chroma.mean(axis=1),
        "pitch": np.array([
            np.median(pitch) if pitch.size else 0.0,
            np.std(pitch) if pitch.size else 0.0
        ]),
        "tempo": np.array([tempo])
    }

# ---------------------------------
# Load DB
# ---------------------------------
conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

cur.execute("""
SELECT f.track_id, f.embedding, a.tempo, a.mfcc, a.chroma, a.pitch_freqs
FROM fused_embeddings f
JOIN audio_features a ON f.track_id = a.track_id
""")

rows = cur.fetchall()
conn.close()

features = [summarize_features(r) for r in rows]
N = len(features)

X_sim = []
y = []

# ---------------------------------
# Pairwise similarities
# ---------------------------------
for i in range(N):
    for j in range(i+1, N):
        sim_vec = [
            cosine_similarity(
                features[i]["fused"].reshape(1,-1),
                features[j]["fused"].reshape(1,-1)
            )[0,0],
            cosine_similarity(
                features[i]["chroma"].reshape(1,-1),
                features[j]["chroma"].reshape(1,-1)
            )[0,0],
            cosine_similarity(
                features[i]["pitch"].reshape(1,-1),
                features[j]["pitch"].reshape(1,-1)
            )[0,0],
            cosine_similarity(
                features[i]["mfcc"].reshape(1,-1),
                features[j]["mfcc"].reshape(1,-1)
            )[0,0],
            1 - abs(features[i]["tempo"][0] - features[j]["tempo"][0]) / 200
        ]

        X_sim.append(sim_vec)

        # pseudo-label
        y.append(1 if sim_vec[0] > 0.85 else 0)

X_sim = np.array(X_sim, dtype=np.float32)
y = np.array(y)

np.savez_compressed(OUT_PATH, X=X_sim, y=y)
print("✅ Saved:", OUT_PATH, X_sim.shape)
