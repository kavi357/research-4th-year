import sqlite3
import numpy as np

DB_PATH = "database/music.db"

def load_all_embeddings():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("SELECT track_id, embedding FROM embeddings ORDER BY track_id")
    rows = cur.fetchall()
    conn.close()

    data = {}
    for track_id, blob in rows:
        data[track_id] = np.frombuffer(blob, dtype=np.float32)
    return data


def load_all_features():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # Include tempo and CREPE pitch data
    cur.execute("""
        SELECT track_id, tempo, mfcc, chroma, pitch_times, pitch_freqs, pitch_conf, pitch_median
        FROM audio_features
        ORDER BY track_id
    """)
    rows = cur.fetchall()
    conn.close()

    data = {}
    for tid, tempo, mfcc_blob, chroma_blob, pt_blob, pf_blob, pc_blob, pm in rows:
        data[tid] = {
            "tempo": float(tempo),
            "mfcc": np.frombuffer(mfcc_blob, dtype=np.float32),
            "chroma": np.frombuffer(chroma_blob, dtype=np.float32),
            "pitch_times": np.frombuffer(pt_blob, dtype=np.float32) if pt_blob else np.array([], dtype=np.float32),
            "pitch_freqs": np.frombuffer(pf_blob, dtype=np.float32) if pf_blob else np.array([], dtype=np.float32),
            "pitch_conf": np.frombuffer(pc_blob, dtype=np.float32) if pc_blob else np.array([], dtype=np.float32),
            "pitch_median": float(pm)
        }
    return data


def compare_vectors(v1, v2):
    """Compare two vectors: check if they are close and compute L2 norm."""
    return np.allclose(v1, v2), np.linalg.norm(v1 - v2)


if __name__ == "__main__":
    print("🔍 Checking uniqueness of embeddings, audio features, tempo, and CREPE pitch…\n")

    embeddings = load_all_embeddings()
    features = load_all_features()

    track_ids = list(embeddings.keys())

    for i in range(len(track_ids)):
        for j in range(i + 1, len(track_ids)):
            t1, t2 = track_ids[i], track_ids[j]

            # Compare embeddings
            same_emb, emb_dist = compare_vectors(embeddings[t1], embeddings[t2])

            # Compare MFCC
            same_mfcc, mfcc_dist = compare_vectors(features[t1]["mfcc"], features[t2]["mfcc"])

            # Compare Chroma
            same_chroma, chroma_dist = compare_vectors(features[t1]["chroma"], features[t2]["chroma"])

            # Compare tempo
            same_tempo = features[t1]["tempo"] == features[t2]["tempo"]
            tempo_diff = abs(features[t1]["tempo"] - features[t2]["tempo"])

            # Compare CREPE pitch frequencies
            same_pitch, pitch_dist = compare_vectors(features[t1]["pitch_freqs"], features[t2]["pitch_freqs"])

            # Compare CREPE pitch median
            same_pitch_median = features[t1]["pitch_median"] == features[t2]["pitch_median"]
            pitch_median_diff = abs(features[t1]["pitch_median"] - features[t2]["pitch_median"])

            print(f"Tracks {t1} vs {t2}:")
            print(f"  ▶ Embeddings identical? {same_emb} | Distance = {emb_dist:.3f}")
            print(f"  ▶ MFCC identical?       {same_mfcc} | Distance = {mfcc_dist:.3f}")
            print(f"  ▶ Chroma identical?     {same_chroma} | Distance = {chroma_dist:.3f}")
            print(f"  ▶ Tempo identical?      {same_tempo} | Diff = {tempo_diff:.3f}")
            print(f"  ▶ CREPE pitch identical? {same_pitch} | Distance = {pitch_dist:.3f}")
            print(f"  ▶ CREPE median identical? {same_pitch_median} | Diff = {pitch_median_diff:.3f}")
            print("")
