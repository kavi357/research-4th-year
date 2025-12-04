# backend/test_upload_and_compare.py
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # reduce TF info/warning spam

import sqlite3
import numpy as np
from pathlib import Path
import shutil
import argparse

# Import your existing functions
from ingest.preprocess import preprocess_audio
from ingest.extract_features import extract_audio_features
from ingest.extract_embeddings import extract_openl3_embedding

# ---------- DB auto-detection ----------
POSSIBLE_DB_PATHS = [
    Path("../database/music.db"),      # <— correct real DB
    Path("database/music.db"),         # backend/database/music.db (wrong one)
    Path("../music.db"),
    Path("music.db"),
    Path("../../database/music.db"),
    Path("../../music.db")
]


def find_db_path():
    for p in POSSIBLE_DB_PATHS:
        if p.exists():
            return str(p)
    # fallback: search for any file named music.db in project tree (one level)
    for p in Path(".").rglob("music.db"):
        return str(p)
    return None

DB_PATH = find_db_path()

# ---------- Helpers ----------
def load_all_db_embeddings(db_path):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("SELECT track_id, embedding, dim FROM embeddings")
    rows = cur.fetchall()
    conn.close()

    if not rows:
        return np.array([]), np.empty((0,))

    ids = []
    vectors = []

    for track_id, blob, dim in rows:
        vec = np.frombuffer(blob, dtype=np.float32)
        # safety: if vec length mismatches expected dim, skip
        if vec.size != dim:
            print(f"⚠️ Skipping track {track_id}: stored dim={dim}, blob length={vec.size}")
            continue
        ids.append(track_id)
        vectors.append(vec)

    if len(vectors) == 0:
        return np.array([]), np.empty((0,))

    return np.array(ids), np.vstack(vectors)


def cosine_similarity(a, b):
    a = a / (np.linalg.norm(a) + 1e-9)
    b = b / (np.linalg.norm(b) + 1e-9)
    return float(np.dot(a, b))


# ---------- Main compare function ----------
def compare_with_database(user_file_path: str, top_k: int = 10):
    if DB_PATH is None:
        print("❌ Could not find music.db. Searched these locations:")
        for p in POSSIBLE_DB_PATHS:
            print("   -", p)
        print("\nPlace your music.db at one of those paths (or pass an explicit path).")
        return

    print("ℹ️ Using DB at:", DB_PATH)

    # prepare tmp dir
    tmp_dir = Path("tmp_uploads")
    tmp_dir.mkdir(exist_ok=True)

    user_path = Path(user_file_path)
    if not user_path.exists():
        print("❌ Uploaded file does not exist:", user_file_path)
        return

    # preprocess -> writes 60s WAV, returned path, duration, sr
    out_wav = tmp_dir / (user_path.stem + "_preproc.wav")
    try:
        preproc_path, duration, sr = preprocess_audio(user_path, out_wav)
    except Exception as e:
        print("❌ Preprocessing failed:", e)
        return

    print("🎵 Extracting features (tempo, mfcc, chroma)...")
    tempo, mfcc, chroma = extract_audio_features(preproc_path)

    print("🎼 Extracting OpenL3 embedding (this may take a while)...")
    try:
        user_emb = extract_openl3_embedding(preproc_path)
    except Exception as e:
        print("❌ OpenL3 embedding extraction failed:", e)
        return

    print("📥 Loading DB embeddings...")
    ids, db_vectors = load_all_db_embeddings(DB_PATH)

    if ids.size == 0 or db_vectors.size == 0:
        print("⚠️ No embeddings found in the database. Please ingest dataset first.")
        print("→ Run your ingestion script (e.g. ingest_covers80.py) to populate embeddings.")
        # cleanup
        try:
            shutil.rmtree(tmp_dir)
        except Exception:
            pass
        return

    print(f"📊 Comparing with {len(ids)} reference tracks...")

    results = []
    for track_id, vec in zip(ids, db_vectors):
        score = cosine_similarity(user_emb, vec)
        results.append((int(track_id), score))

    results.sort(key=lambda x: x[1], reverse=True)

    print("\n==== Top Matches (Cosine Similarity) ====")
    for track_id, score in results[:top_k]:
        print(f"Track {track_id} → Score: {score:.4f}")

    # cleanup
    try:
        shutil.rmtree(tmp_dir)
    except Exception:
        pass

    return results


# ---------- CLI ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload song & compare similarity (robust DB detection)")
    parser.add_argument("file", type=str, help="Path to audio file on your PC")
    parser.add_argument("--top", type=int, default=10, help="Top K results to show")
    args = parser.parse_args()

    compare_with_database(args.file, top_k=args.top)
