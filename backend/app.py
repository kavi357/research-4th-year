# backend/app.py

import uvicorn
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import sqlite3
from pathlib import Path
import tempfile
import os

# Import your pipeline
from ingest.preprocess import preprocess_audio
from ingest.extract_features import extract_audio_features
from ingest.extract_embeddings import extract_openl3_embedding

# ======================================
# DB PATH
# ======================================
DB_PATH = Path(__file__).resolve().parents[1] / "database" / "music.db"


# ======================================
# FASTAPI APP
# ======================================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ======================================
# SIMILARITY FUNCTIONS
# ======================================
from numpy.linalg import norm

def cosine_similarity(a, b):
    return float(np.dot(a, b) / (norm(a) * norm(b)))

def euclidean_similarity(a, b):
    dist = np.linalg.norm(a - b)
    return 1 / (1 + dist)

def mfcc_similarity(q_mfcc, db_mfcc):
    min_len = min(len(q_mfcc), len(db_mfcc))
    return cosine_similarity(q_mfcc[:min_len], db_mfcc[:min_len])

def chroma_similarity(q_chroma, db_chroma):
    min_len = min(len(q_chroma), len(db_chroma))
    return cosine_similarity(q_chroma[:min_len], db_chroma[:min_len])


# ======================================
# HYBRID SCORE
# ======================================
def hybrid_score(cos_emb, euc_emb, mfcc_sim, chroma_sim,
                 w_cos=0.45, w_euc=0.25, w_mfcc=0.20, w_chroma=0.10):
    return (
        w_cos * cos_emb +
        w_euc * euc_emb +
        w_mfcc * mfcc_sim +
        w_chroma * chroma_sim
    )


# ======================================
# LOAD DB
# ======================================
def load_db_embeddings():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT track_id, embedding FROM embeddings")

    ids, vecs = [], []
    for track_id, blob in cur.fetchall():
        ids.append(track_id)
        vecs.append(np.frombuffer(blob, dtype=np.float32))

    conn.close()
    return ids, vecs


def load_db_features():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT track_id, tempo, mfcc, chroma FROM audio_features")

    rows = []
    for track_id, tempo, mfcc_blob, chroma_blob in cur.fetchall():
        rows.append((
            track_id,
            tempo,
            np.frombuffer(mfcc_blob, dtype=np.float32),
            np.frombuffer(chroma_blob, dtype=np.float32)
        ))

    conn.close()
    return rows



# ======================================
# API ROUTE — UPLOAD AUDIO
# ======================================
@app.post("/analyze")
async def analyze_song(file: UploadFile = File(...)):

    # Save uploaded file temporarily
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    temp_file.write(await file.read())
    temp_file.close()

    # ---- 1) Preprocess ----
    out_wav = Path("verify_temp.wav")
    wav_path, duration, sr = preprocess_audio(temp_file.name, out_wav)

    # ---- 2) Features ----
    q_tempo, q_mfcc, q_chroma = extract_audio_features(wav_path)

    # ---- 3) Embedding ----
    q_emb = extract_openl3_embedding(wav_path)

    # ---- 4) Load DB ----
    ids, db_embs = load_db_embeddings()
    db_feats = load_db_features()

    if len(ids) == 0:
        return {"error": "Database is empty. Run ingestion first!"}

    # ---- 5) Compute similarity ----
    results = []

    for (track_id, tempo, db_mfcc, db_chroma), db_emb in zip(db_feats, db_embs):
        S1 = cosine_similarity(q_emb, db_emb)
        S2 = euclidean_similarity(q_emb, db_emb)
        S3 = mfcc_similarity(q_mfcc.flatten(), db_mfcc)
        S4 = chroma_similarity(q_chroma.flatten(), db_chroma)

        hybrid = hybrid_score(S1, S2, S3, S4)

        results.append({
            "track_id": int(track_id),
            "hybrid_score": float(hybrid),
            "cosine": float(S1),
            "euclidean": float(S2),
            "mfcc": float(S3),
            "chroma": float(S4)
        })

    # Sort top results
    results.sort(key=lambda x: x["hybrid_score"], reverse=True)

    return {
        "top_5": results[:5],
        "status": "success"
    }


# ======================================
# RUN SERVER
# ======================================
if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
