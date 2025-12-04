# similarity/similarity_checker.py

import sqlite3
import numpy as np
import librosa
from scipy.spatial.distance import cosine, euclidean

# -----------------------------
# Settings
# -----------------------------
TARGET_SR = 48000
TARGET_DURATION = 60.0  # seconds

# Weights for hybrid similarity
WEIGHT_EMBEDDING = 0.6
WEIGHT_MFCC = 0.15
WEIGHT_CHROMA = 0.15
WEIGHT_TEMPO = 0.10

# -----------------------------
# Preprocess uploaded song
# -----------------------------
def preprocess_audio(wav_path):
    y, sr = librosa.load(wav_path, sr=TARGET_SR, mono=True)
    y_trim, _ = librosa.effects.trim(y, top_db=25)
    target_samples = int(TARGET_SR * TARGET_DURATION)
    if len(y_trim) > target_samples:
        y_out = y_trim[:target_samples]
    else:
        y_out = np.pad(y_trim, (0, target_samples - len(y_trim)), mode="constant")
    rms = np.sqrt(np.mean(y_out**2) + 1e-12)
    y_out = y_out * (10**(-20 / 20) / rms)
    y_out = np.clip(y_out, -1.0, 1.0).astype(np.float32)
    return y_out, TARGET_SR

# -----------------------------
# Extract features for uploaded song
# -----------------------------
def extract_features(y, sr):
    # MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfcc_mean = mfcc.mean(axis=1)

    # Chroma
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_mean = chroma.mean(axis=1)

    # Tempo
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

    return mfcc_mean, chroma_mean, tempo

# -----------------------------
# Load fused embeddings and features from DB
# -----------------------------
def load_db_features(db_path):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("""
        SELECT t.id, f.embedding, a.mfcc, a.chroma, a.tempo
        FROM tracks t
        JOIN fused_embeddings f ON t.id = f.track_id
        JOIN audio_features a ON t.id = a.track_id
    """)
    rows = cur.fetchall()
    conn.close()

    db_data = []
    for row in rows:
        track_id = row[0]
        fused_emb = np.frombuffer(row[1], dtype=np.float32)

        mfcc_db = np.frombuffer(row[2], dtype=np.float32).reshape(20, -1).mean(axis=1)
        chroma_db = np.frombuffer(row[3], dtype=np.float32).reshape(12, -1).mean(axis=1)
        tempo_db = row[4]

        db_data.append({
            "track_id": track_id,
            "embedding": fused_emb,
            "mfcc": mfcc_db,
            "chroma": chroma_db,
            "tempo": tempo_db
        })

    return db_data

# -----------------------------
# Compute hybrid similarity
# -----------------------------
def compute_similarity(uploaded_features, db_features):
    mfcc_u, chroma_u, tempo_u, emb_u = uploaded_features
    results = []

    for db in db_features:
        emb_sim = 1 - cosine(emb_u, db["embedding"])  # cosine similarity

        mfcc_dist = euclidean(mfcc_u, db["mfcc"])
        mfcc_sim = 1 / (1 + mfcc_dist)

        chroma_dist = euclidean(chroma_u, db["chroma"])
        chroma_sim = 1 / (1 + chroma_dist)

        tempo_diff = abs(tempo_u - db["tempo"])
        tempo_sim = 1 / (1 + tempo_diff / 100)

        final_score = (WEIGHT_EMBEDDING * emb_sim +
                       WEIGHT_MFCC * mfcc_sim +
                       WEIGHT_CHROMA * chroma_sim +
                       WEIGHT_TEMPO * tempo_sim)

        results.append({
            "track_id": db["track_id"],
            "embedding_similarity": emb_sim,
            "mfcc_distance": mfcc_dist,
            "chroma_distance": chroma_dist,
            "tempo_similarity": tempo_sim,
            "final_similarity": final_score
        })

    results.sort(key=lambda x: x["final_similarity"], reverse=True)
    return results

# -----------------------------
# Main comparison function
# -----------------------------
def compare_song_to_database(wav_path, db_path):
    # Preprocess + features
    y, sr = preprocess_audio(wav_path)
    mfcc_u, chroma_u, tempo_u = extract_features(y, sr)

    # Load fused embeddings from DB
    db_data = load_db_features(db_path)

    # Extract fused embedding for uploaded song
    import openl3
    import tensorflow_hub as hub

    openl3_model = openl3.models.load_audio_embedding_model("mel256", "music", 512)
    yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")

    emb_openl3, _ = openl3.get_audio_embedding(y, sr, model=openl3_model, hop_size=0.1, center=True)
    emb_openl3_mean = emb_openl3.mean(axis=0).astype(np.float32)

    y_16k = librosa.resample(y, sr, 16000)
    _, emb_yamnet, _ = yamnet_model(y_16k)
    emb_yamnet_mean = emb_yamnet.numpy().mean(axis=0).astype(np.float32)

    fused_embedding_vector = np.concatenate([emb_openl3_mean, emb_yamnet_mean]).astype(np.float32)

    results = compute_similarity((mfcc_u, chroma_u, tempo_u, fused_embedding_vector), db_data)

    # Include test song path in results
    for r in results:
        r["test_song"] = wav_path

    return results

# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    db_path = "database/music.db"
    test_song_path = "D:\Downloads\test1.mp3"

    results = compare_song_to_database(test_song_path, db_path)

    print(f"\nTest Song: {test_song_path}\n")
    print("=========== SIMILARITY RESULTS ===========\n")
    for r in results[:5]:  # top 5 similar tracks
        print(f"Track {r['track_id']}:")
        print(f"  Embedding Cosine:  {r['embedding_similarity']:.4f}")
        print(f"  MFCC Distance:     {r['mfcc_distance']:.2f}")
        print(f"  Chroma Distance:   {r['chroma_distance']:.2f}")
        print(f"  Tempo Similarity:  {r['tempo_similarity']:.4f}")
        print(f"  Final Score:       {r['final_similarity']:.4f}\n")
    print("===========================================\n")
