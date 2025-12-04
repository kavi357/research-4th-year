# backend/similarity/load_from_db.py

import sqlite3
import numpy as np

def load_reference_data(db_path):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("SELECT track_id, embedding FROM embeddings")
    emb_rows = cur.fetchall()

    cur.execute("SELECT track_id, mfcc, chroma FROM audio_features")
    feat_rows = cur.fetchall()

    conn.close()

    embeddings = {}
    for track_id, emb_bytes in emb_rows:
        embeddings[track_id] = np.frombuffer(emb_bytes, dtype=np.float32)

    features = {}
    for track_id, mfcc_bytes, chroma_bytes in feat_rows:
        mfcc = np.frombuffer(mfcc_bytes, dtype=np.float32)
        chroma = np.frombuffer(chroma_bytes, dtype=np.float32)
        features[track_id] = (mfcc, chroma)

    return embeddings, features
