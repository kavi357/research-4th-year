import json
import sqlite3
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

# =====================================================
# PATHS
# =====================================================
ROOT = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "database" / "music.db"
PAIRS_PATH = ROOT / "data" / "pairs_new.json"
MODEL_DIR = ROOT / "models" / "filter_net"

# =====================================================
# MODEL DEFINITION (MUST MATCH TRAINING)
# =====================================================
class SimilarityNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128)
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=1)

# =====================================================
# LOAD MODEL + PREPROCESSORS
# =====================================================
device = "cpu"

model = SimilarityNet(input_dim=384)
model.load_state_dict(
    torch.load(MODEL_DIR / "similarity_net_no_covers80.pth", map_location=device)
)
model.eval()

scaler = joblib.load(MODEL_DIR / "scaler_no_covers80.joblib")
pca = joblib.load(MODEL_DIR / "pca_384_no_covers80.joblib")

print("✅ Learned similarity model loaded")

# =====================================================
# FULL FEATURE LOADER (MATCHES TRAINING)
# =====================================================
def load_full_feature_vector(file_path):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("""
        SELECT
            f.embedding,
            a.tempo,
            a.mfcc,
            a.chroma,
            a.pitch_freqs
        FROM tracks t
        JOIN fused_embeddings f ON t.id = f.track_id
        JOIN audio_features a ON t.id = a.track_id
        WHERE REPLACE(t.file_path, '\\', '/') = ?
    """, (file_path.replace("\\", "/"),))

    row = cur.fetchone()
    conn.close()

    if not row:
        return None

    fused_blob, tempo, mfcc_blob, chroma_blob, pitch_blob = row

    # ---- fused embedding (1536)
    fused = np.frombuffer(fused_blob, np.float32)

    # ---- MFCC (20, T)
    mfcc = np.frombuffer(mfcc_blob, np.float32).reshape(20, -1)
    mfcc_mean = mfcc.mean(axis=1)
    mfcc_std  = mfcc.std(axis=1)

    # ---- Chroma (12, T)
    chroma = np.frombuffer(chroma_blob, np.float32).reshape(12, -1)
    chroma_mean = chroma.mean(axis=1)

    # ---- Pitch
    pitch = np.frombuffer(pitch_blob, np.float32) if pitch_blob else np.array([])
    pitch_median = np.median(pitch) if pitch.size else 0.0
    pitch_std    = np.std(pitch) if pitch.size else 0.0

    # ---- FINAL FEATURE VECTOR (1591)
    feature_vector = np.concatenate([
        fused,               # 1536
        mfcc_mean,           # 20
        mfcc_std,            # 20
        chroma_mean,         # 12
        [pitch_median, pitch_std, tempo]  # 3
    ]).astype(np.float32)

    return feature_vector

# =====================================================
# EMBEDDING THROUGH LEARNED MODEL
# =====================================================
def embed_learned(x):
    x = scaler.transform(x.reshape(1, -1))
    x = pca.transform(x)
    with torch.no_grad():
        z = model(torch.tensor(x, dtype=torch.float32))
    return z.numpy()[0]

# =====================================================
# LOAD PAIRS
# =====================================================
pairs = json.load(open(PAIRS_PATH))

scores = []
skipped = 0

# =====================================================
# EVALUATION LOOP
# =====================================================
for p in pairs:
    q = load_full_feature_vector(p["query"])
    d = load_full_feature_vector(p["db"])

    if q is None or d is None:
        skipped += 1
        continue

    zq = embed_learned(q)
    zd = embed_learned(d)

    sim = cosine_similarity(zq[None], zd[None])[0][0]
    scores.append((sim, p["label"]))

scores = np.array(scores)

print(f"Evaluated pairs: {len(scores)} | Skipped: {skipped}")

# =====================================================
# RECALL@K (PAIR-BASED)
# =====================================================
def recall_at_k(scores, k):
    pos = scores[scores[:, 1] == 1][:, 0]
    neg = scores[scores[:, 1] == 0][:, 0]

    hits = 0
    for s in pos:
        rank = np.sum(neg >= s) + 1
        if rank <= k:
            hits += 1

    return hits / len(pos)

print("\nLEARNED SIMILARITY MODEL (NO COVERS80 TRAINING)")
for k in [1, 5, 10, 50, 100, 200]:
    print(f"Recall@{k}: {recall_at_k(scores, k):.3f}")
