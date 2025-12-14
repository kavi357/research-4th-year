# backend/app.py

import uvicorn
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import librosa
import sqlite3
from pathlib import Path
import tempfile
import os
from similarity.similarity import tempo_similarity, pitch_similarity, hybrid_basic_score
from similarity.db_utils import load_audio_features


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

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    temp_file.write(await file.read())
    temp_file.close()

    # Load audio
    y, sr = librosa.load(temp_file.name, sr=None, mono=True)

    # Extract features (reuse your function)
    tempo, _, _, _, _, _, pitch_median = extract_audio_features(y, sr)

    db_rows = load_audio_features()
    results = []

    for track_id, db_tempo, db_pitch in db_rows:
        t_sim = tempo_similarity(tempo, db_tempo)
        p_sim = pitch_similarity(pitch_median, db_pitch)
        score = hybrid_basic_score(t_sim, p_sim)

        results.append({
            "track_id": track_id,
            "tempo_similarity": round(t_sim * 100, 2),
            "pitch_similarity": round(p_sim * 100, 2),
            "overall_score": round(score * 100, 2)
        })

    results.sort(key=lambda x: x["overall_score"], reverse=True)

    return {
        "query": {
            "tempo": round(float(tempo), 2),
            "pitch_median": round(float(pitch_median), 2)
        },
        "top_matches": results[:5],
        "status": "success"
    }


# ======================================
# RUN SERVER
# ======================================
if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
