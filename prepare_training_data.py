import sqlite3
import numpy as np
from pathlib import Path

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
DB_PATH = "database/music.db"
OUTPUT_PATH = Path(__file__).resolve().parents[1] / "output"
OUTPUT_PATH.mkdir(exist_ok=True)

OUT_FILE = OUTPUT_PATH / "training_features.npz"

print("Using DB:", DB_PATH)


# --------------------------------------------------
# STEP 1: LOAD DATA FROM SQLITE
# --------------------------------------------------
def load_rows():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("""
        SELECT 
            f.track_id,
            f.embedding,
            a.tempo,
            a.mfcc,
            a.chroma,
            a.pitch_freqs
        FROM fused_embeddings f
        JOIN audio_features a
        ON f.track_id = a.track_id
    """)

    rows = cur.fetchall()
    conn.close()

    print(f"Loaded {len(rows)} tracks from database")
    return rows


# --------------------------------------------------
# STEP 2: BUILD FINAL FEATURE VECTOR
# --------------------------------------------------
def build_feature_vector(row):
    _, fused_blob, tempo, mfcc_blob, chroma_blob, pitch_blob = row

    # Fused OpenL3 + YAMNet (1536,)
    fused = np.frombuffer(fused_blob, dtype=np.float32)

    # MFCC (20, T)
    mfcc = np.frombuffer(mfcc_blob, dtype=np.float32).reshape(20, -1)

    # Chroma (12, T)
    chroma = np.frombuffer(chroma_blob, dtype=np.float32).reshape(12, -1)

    # Pitch
    pitch = (
        np.frombuffer(pitch_blob, dtype=np.float32)
        if pitch_blob is not None else np.array([], dtype=np.float32)
    )

    # --- Statistical summarization ---
    mfcc_mean = mfcc.mean(axis=1)
    mfcc_std  = mfcc.std(axis=1)

    chroma_mean = chroma.mean(axis=1)

    pitch_median = np.median(pitch) if pitch.size else 0.0
    pitch_std    = np.std(pitch) if pitch.size else 0.0

    feature_vector = np.concatenate([
        fused,
        mfcc_mean,
        mfcc_std,
        chroma_mean,
        [pitch_median, pitch_std, tempo]
    ]).astype(np.float32)

    return feature_vector


# --------------------------------------------------
# STEP 3: BUILD DATASET & SAVE
# --------------------------------------------------
def main():
    rows = load_rows()

    X = []
    track_ids = []

    for row in rows:
        try:
            vec = build_feature_vector(row)
            X.append(vec)
            track_ids.append(row[0])
        except Exception as e:
            print(f"⚠ Skipping track {row[0]}: {e}")

    X = np.vstack(X)
    track_ids = np.array(track_ids)

    print("Final feature matrix shape:", X.shape)

    np.savez_compressed(
        OUT_FILE,
        X=X,
        ids=track_ids
    )

    print("✅ Saved:", OUT_FILE)
    print("File size (MB):", OUT_FILE.stat().st_size / (1024 * 1024))


# --------------------------------------------------
if __name__ == "__main__":
    main()
