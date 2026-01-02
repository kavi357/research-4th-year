# ============================================================
# infer_uploaded_song.py
# Full inference pipeline for NEW uploaded song
# ============================================================

import numpy as np
import torch
import torch.nn.functional as F
import joblib
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path

# ============================================================
# IMPORT YOUR EXISTING BACKEND CODE (DO NOT CHANGE)
# ============================================================
from backend.ingest.preprocess import preprocess_audio
from backend.ingest.extract_features import extract_audio_features
from backend.ingest.extract_embeddings import extract_fused_embedding

# ============================================================
# PATHS
# ============================================================
MODEL_DIR = Path("models/net")

SCALER_PATH = MODEL_DIR / "scaler.joblib"
PCA_PATH = MODEL_DIR / "pca_256.joblib"
MODEL_PATH = MODEL_DIR / "similarity_net.pth"
STATS_PATH = MODEL_DIR / "sim_stats.npz"
REF_EMB_PATH = MODEL_DIR / "learned_embeddings.npz"

# ============================================================
# LOAD ARTIFACTS
# ============================================================
print("🔄 Loading artifacts...")

scaler = joblib.load(SCALER_PATH)
pca = joblib.load(PCA_PATH)

stats = np.load(STATS_PATH)
MEAN_SIM = float(stats["mean"])
STD_SIM = float(stats["std"])

ref_data = np.load(REF_EMB_PATH)
REF_EMBEDDINGS = ref_data["embeddings"]  # (648, 64)

print("✅ Artifacts loaded")

# ============================================================
# DEFINE MODEL (MUST MATCH TRAINING)
# ============================================================
class SimilarityNet(torch.nn.Module):
    def __init__(self, input_dim=256):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64)
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=1)

model = SimilarityNet()
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

print("✅ Similarity model loaded")

# ============================================================
# FEATURE VECTOR BUILDER (1591D)
# MUST MATCH TRAINING ORDER EXACTLY
# ============================================================
def build_feature_vector(audio_path: str) -> np.ndarray:
    """
    Build the EXACT SAME 1591D feature vector used in training
    """

    # ----------------------------
    # Preprocess
    # ----------------------------
    y, duration, sr = preprocess_audio(audio_path)

    # ----------------------------
    # Extract features
    # ----------------------------
    tempo, mfcc, chroma, pt, pf, pc, pitch_median = extract_audio_features(y, sr)

    # ----------------------------
    # FIX SCALARS
    # ----------------------------
    tempo = float(np.asarray(tempo).squeeze())
    pitch_median = float(np.asarray(pitch_median).squeeze())

    # Pitch contour → std
    if pf is not None and len(pf) > 0:
        pitch_std = float(np.std(pf))
    else:
        pitch_std = 0.0

    # ----------------------------
    # Statistical summaries
    # ----------------------------
    mfcc_mean = mfcc.mean(axis=1)
    mfcc_std  = mfcc.std(axis=1)
    chroma_mean = chroma.mean(axis=1)

    # ----------------------------
    # Deep embedding
    # ----------------------------
    fused_embedding = extract_fused_embedding(y, sr)
    fused_embedding = fused_embedding.reshape(-1)

    # ----------------------------
    # FINAL VECTOR — ORDER MUST MATCH TRAINING
    # ----------------------------
    feature_vector = np.concatenate([
        fused_embedding,     # 1536
        mfcc_mean,           # 20
        mfcc_std,            # 20
        chroma_mean,         # 12
        np.array([
            pitch_median,    # 1
            pitch_std,       # 1
            tempo             # 1
        ])
    ]).astype(np.float32)

    return feature_vector



# ============================================================
# SIMILARITY SCORE FUNCTION
# ============================================================
def compute_similarity_score(audio_path: str):
    """
    Returns:
        normalized_score (0–1)
        raw_cosine_similarity
    """

    # 1. Extract features
    feature_1591 = build_feature_vector(audio_path)

    # 2. Normalize
    x = scaler.transform(feature_1591.reshape(1, -1))

    # 3. PCA
    x = pca.transform(x)

    # 4. Metric embedding
    with torch.no_grad():
        emb = model(torch.tensor(x, dtype=torch.float32)).numpy()

    # 5. Cosine similarity
    sims = cosine_similarity(emb, REF_EMBEDDINGS)[0]
    max_sim = float(sims.max())

    # 6. Statistical normalization (Z + sigmoid)
    z = (max_sim - MEAN_SIM) / STD_SIM
    score = 1 / (1 + np.exp(-z))

    return float(score), max_sim

# ============================================================
# MAIN (TEST WITH NEW SONG)
# ============================================================
if __name__ == "__main__":

    # 🔴 CHANGE THIS TO ANY NEW SONG (NOT IN DB)
    test_song_path = "test8_gtzan.wav"

    print("\n🎵 Checking similarity for uploaded song:")
    print(test_song_path)

    score, raw_sim = compute_similarity_score(test_song_path)

    print("\n========= RESULT =========")
    print(f"Raw cosine similarity     : {raw_sim:.4f}")
    print(f"Copyright similarity score: {score:.4f}")

    if score > 0.8:
        print("⚠️ HIGH copyright risk")
    elif score > 0.6:
        print("⚠️ POSSIBLE similarity")
    else:
        print("✅ Low similarity")
