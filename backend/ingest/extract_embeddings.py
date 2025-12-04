import numpy as np
import sqlite3
import librosa
import openl3
from pathlib import Path

# Import your existing YAMNet extractor
from .extract_yamnet import extract_yamnet_embedding

DB_PATH = str(Path(__file__).resolve().parents[2] / "database" / "music.db")


# -----------------------------------------------------
# Load OpenL3 model ONCE — using your local model
# -----------------------------------------------------
MODEL_DIR = Path(__file__).resolve().parents[2] / "models" / "openl3"
MODEL_PATH = str(MODEL_DIR / "openl3_music_mel256_512.h5")

OPENL3_MODEL = openl3.models.load_audio_embedding_model(
    input_repr="mel256",
    content_type="music",
    embedding_size=512
)


# -----------------------------------------------------
# OpenL3 embedding
# -----------------------------------------------------
def extract_openl3_embedding(y, sr):
    y = y.astype(np.float32)

    emb, ts = openl3.get_audio_embedding(
        y,
        sr,
        model=OPENL3_MODEL,
        hop_size=0.1,
        center=True
    )

    return emb.mean(axis=0).astype(np.float32)   # (512,)


# -----------------------------------------------------
# Fused embedding (OpenL3 + YAMNet)
# -----------------------------------------------------
def extract_fused_embedding(y, sr):
    """
    Returns 1536-dimensional vector: [512 OpenL3 | 1024 YAMNet]
    """
    emb_openl3 = extract_openl3_embedding(y, sr)
    emb_yamnet = extract_yamnet_embedding(y, sr)

    fused = np.concatenate([emb_openl3, emb_yamnet]).astype(np.float32)
    return fused


# -----------------------------------------------------
# Insert ONLY fused embeddings
# -----------------------------------------------------
def insert_fused_embedding(track_id, vector):

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("""
        INSERT OR REPLACE INTO fused_embeddings (track_id, embedding, dim)
        VALUES (?, ?, ?)
    """, (
        track_id,
        vector.tobytes(),
        vector.shape[0]
    ))

    conn.commit()
    conn.close()


# -----------------------------------------------------
# Full pipeline for one track
# -----------------------------------------------------
def process_track(file_path: str, track_id: int):
    """
    ALWAYS saves ONLY fused embeddings.
    """
    y, sr = librosa.load(file_path, sr=None, mono=True)

    emb_fused = extract_fused_embedding(y, sr)
    insert_fused_embedding(track_id, emb_fused)

    print(f"✔ Saved FUSED embedding for track {track_id}")
