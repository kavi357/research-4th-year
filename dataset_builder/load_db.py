import sqlite3
import numpy as np

DB_PATH = 'database/music.db'


def load_fused_embedding(track_id):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT embedding FROM fused_embeddings WHERE track_id=?", (track_id,))
    row = cur.fetchone()
    conn.close()
    if row is None:
        return None
    return np.frombuffer(row[0], dtype=np.float32)


def load_audio_features(track_id):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        SELECT tempo, mfcc, chroma, pitch_freqs, pitch_conf, pitch_median
        FROM audio_features
        WHERE track_id=?
    """, (track_id,))
    row = cur.fetchone()
    conn.close()

    if row is None:
        return None

    tempo = float(row[0])

    mfcc = np.frombuffer(row[1], dtype=np.float32)
    chroma = np.frombuffer(row[2], dtype=np.float32)

    pitch_freqs = np.frombuffer(row[3], dtype=np.float32) if row[3] else np.array([], dtype=np.float32)

    if row[4]:
        pitch_conf_arr = np.frombuffer(row[4], dtype=np.float32)
        pitch_conf_mean = float(pitch_conf_arr.mean()) if pitch_conf_arr.size else 0.0
    else:
        pitch_conf_mean = 0.0

    pitch_median = float(row[5]) if row[5] is not None else 0.0

    return tempo, mfcc, chroma, pitch_freqs, pitch_conf_mean, pitch_median


def load_all_track_ids():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT id FROM tracks")
    rows = cur.fetchall()
    conn.close()
    return [r[0] for r in rows]
