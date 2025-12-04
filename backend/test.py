# test_similarity.py

import sqlite3
import numpy as np
from pathlib import Path
import librosa

# -----------------------------
# Import preprocess and embedding functions from your backend
# -----------------------------
from backend.ingest.preprocess import preprocess_audio
from backend.ingest.extract_embeddings import extract_fused_embedding
from backend.ingest.extract_features import extract_audio_features  # make sure this exists

# -----------------------------
# Database path
# -----------------------------
DB_PATH = Path(__file__).resolve().parent.parent / "database" / "music.db"

# -----------------------------
# Hybrid score weights
# -----------------------------
WEIGHTS = {
    "embedding": 0.6,
    "mfcc": 0.2,
    "chroma": 0.1,
    "tempo": 0.1
}

# -----------------------------
# Similarity functions
# -----------------------------
def cosine_similarity(a, b):
    a_norm = a / np.linalg.norm(a)
    b_norm = b / np.linalg.norm(b)
    return float(np.dot(a_norm, b_norm))

def euclidean_distance(a, b):
    return float(np.linalg.norm(a - b))

# -----------------------------
# Compare a test song to DB
# -----------------------------
def compare_song_to_database(test_song_path, db_path=DB_PATH):
    test_song_path = Path(test_song_path)
    if not test_song_path.exists():
        raise FileNotFoundError(f"Test song not found: {test_song_path}")

    # 1️⃣ Preprocess audio
    y, duration, sr = preprocess_audio(test_song_path)

    # 2️⃣ Extract fused embeddings
    test_emb = extract_fused_embedding(y, sr)

    # 3️⃣ Extract MFCC / Chroma / Tempo
    tempo, mfcc, chroma, _, _, _, _ = extract_audio_features(y, sr)
    tempo = float(tempo)  # ⚡ Ensure scalar

    # 4️⃣ Connect to DB
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    results = []

    # 5️⃣ Fetch all tracks with embeddings & features
    cur.execute("""
        SELECT t.id, t.title, f.embedding, a.mfcc, a.chroma, a.tempo
        FROM tracks t
        JOIN fused_embeddings f ON t.id = f.track_id
        JOIN audio_features a ON t.id = a.track_id
    """)
    rows = cur.fetchall()

    for track_id, title, emb_blob, mfcc_blob, chroma_blob, db_tempo in rows:
        db_emb = np.frombuffer(emb_blob, dtype=np.float32)
        db_mfcc = np.frombuffer(mfcc_blob, dtype=np.float32)
        db_chroma = np.frombuffer(chroma_blob, dtype=np.float32)
        db_tempo = float(db_tempo)  # ⚡ Ensure scalar

        # 6️⃣ Compute similarities
        embedding_sim = cosine_similarity(test_emb, db_emb)  # 0-1

        # ⚡ Use cosine similarity for MFCC & Chroma instead of Euclidean
        mfcc_sim = cosine_similarity(mfcc.flatten(), db_mfcc.flatten())
        chroma_sim = cosine_similarity(chroma.flatten(), db_chroma.flatten())
        tempo_sim = 1 / (1 + abs(tempo - db_tempo))  # simple normalization

        # 7️⃣ Weighted hybrid score
        final_score = (
            WEIGHTS["embedding"] * embedding_sim +
            WEIGHTS["mfcc"] * mfcc_sim +
            WEIGHTS["chroma"] * chroma_sim +
            WEIGHTS["tempo"] * tempo_sim
        )

        results.append({
            "track_id": track_id,
            "title": title,
            "embedding_similarity": embedding_sim,
            "mfcc_similarity": mfcc_sim,
            "chroma_similarity": chroma_sim,
            "tempo_similarity": tempo_sim,
            "final_similarity": final_score
        })

    # Sort descending by final similarity
    results = sorted(results, key=lambda x: x["final_similarity"], reverse=True)
    conn.close()
    return results

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    test_song_path = Path("D:/Downloads/test1.mp3")  # <- update your test song path

    results = compare_song_to_database(test_song_path, DB_PATH)

    print("\n=========== SIMILARITY RESULTS ===========\n")
    for r in results[:10]:  # show top 10 matches
        print(f"Track {r['track_id']} - {r['title']}:")
        print(f"  Embedding Cosine:  {r['embedding_similarity']:.4f}")
        print(f"  MFCC Cosine:       {r['mfcc_similarity']:.4f}")
        print(f"  Chroma Cosine:     {r['chroma_similarity']:.4f}")
        print(f"  Tempo Similarity:  {r['tempo_similarity']:.4f}")
        print(f"  Final Score:       {r['final_similarity']:.4f}\n")
    print("===========================================\n")




#import sqlite3
#import numpy as np
#from pathlib import Path

#DB_PATH = Path("database/music.db")

#conn = sqlite3.connect(DB_PATH)
#cur = conn.cursor()

#cur.execute("SELECT embedding, dim FROM fused_embeddings LIMIT 1")
#blob, dim = cur.fetchone()

#vector = np.frombuffer(blob, dtype=np.float32)

#print("Shape:", vector.shape)
#print("First 10 values:", vector[:10])

#conn.close()
