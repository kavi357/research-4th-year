import json
import sqlite3
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# =====================================================
# PATHS
# =====================================================
ROOT = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "database" / "music.db"
PAIRS_PATH = ROOT / "evaluation" / "melody" / "pairs_new.json"
MODEL_DIR = ROOT / "models" / "filter_net"
PLOT_DIR = Path(__file__).resolve().parent / "plots_learned_model"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# =====================================================
# MODEL DEFINITION
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
# LOAD MODEL
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
# FEATURE LOADING
# =====================================================
def normalize_path(path):
    p = Path(path)
    parts = p.parts
    try:
        data_idx = parts.index('data')
        relative = Path(*parts[data_idx:])
        return str(relative).replace('\\', '/')
    except ValueError:
        return str(p).replace('\\', '/')

def load_feature_for_path(path):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    
    normalized = normalize_path(path)
    filename = Path(path).name
    
    queries = [
        f"""
        SELECT f.embedding, a.tempo, a.mfcc, a.chroma, a.pitch_freqs
        FROM tracks t
        JOIN fused_embeddings f ON t.id = f.track_id
        JOIN audio_features a ON t.id = a.track_id
        WHERE REPLACE(t.file_path, '\\', '/') LIKE '%{normalized}'
        LIMIT 1
        """,
        f"""
        SELECT f.embedding, a.tempo, a.mfcc, a.chroma, a.pitch_freqs
        FROM tracks t
        JOIN fused_embeddings f ON t.id = f.track_id
        JOIN audio_features a ON t.id = a.track_id
        WHERE t.file_path LIKE '%{filename}'
        LIMIT 1
        """
    ]
    
    row = None
    for query in queries:
        cur.execute(query)
        row = cur.fetchone()
        if row:
            break
    
    conn.close()
    
    if not row:
        return None
    
    fused_blob, tempo, mfcc_blob, chroma_blob, pitch_blob = row
    
    fused = np.frombuffer(fused_blob, np.float32)
    
    mfcc = np.frombuffer(mfcc_blob, np.float32).reshape(20, -1)
    mfcc_mean = mfcc.mean(axis=1)
    mfcc_std = mfcc.std(axis=1)
    
    chroma = np.frombuffer(chroma_blob, np.float32).reshape(12, -1)
    chroma_mean = chroma.mean(axis=1)
    
    pitch = np.frombuffer(pitch_blob, np.float32) if pitch_blob else np.array([])
    pitch_median = np.median(pitch) if pitch.size else 0.0
    pitch_std = np.std(pitch) if pitch.size else 0.0
    
    feature_vector = np.concatenate([
        fused,
        mfcc_mean,
        mfcc_std,
        chroma_mean,
        [pitch_median, pitch_std, tempo]
    ]).astype(np.float32)
    
    return feature_vector

def embed_learned(x):
    x = scaler.transform(x.reshape(1, -1))
    x = pca.transform(x)
    with torch.no_grad():
        z = model(torch.tensor(x, dtype=torch.float32))
    return z.numpy()[0]

# =====================================================
# EVALUATE
# =====================================================
pairs = json.load(open(PAIRS_PATH))
print(f"\n📊 Loaded {len(pairs)} pairs from {PAIRS_PATH.name}")

feature_cache = {}

def get_feature_cached(path):
    if path not in feature_cache:
        feature_cache[path] = load_feature_for_path(path)
    return feature_cache[path]

scores = []
skipped = 0

print(f"\n🔄 Processing pairs...")

for i, p in enumerate(pairs):
    if (i + 1) % 20 == 0:
        print(f"   Processed {i + 1}/{len(pairs)} pairs...")
    
    q_feat = get_feature_cached(p["query"])
    d_feat = get_feature_cached(p["db"])
    
    if q_feat is None or d_feat is None:
        skipped += 1
        continue
    
    zq = embed_learned(q_feat)
    zd = embed_learned(d_feat)
    
    sim = cosine_similarity(zq[None], zd[None])[0][0]
    scores.append((sim, p["label"]))

scores = np.array(scores)

# =====================================================
# CALCULATE METRICS
# =====================================================
def recall_at_k(scores, k):
    pos = scores[scores[:, 1] == 1][:, 0]
    neg = scores[scores[:, 1] == 0][:, 0]
    
    if len(pos) == 0:
        return 0.0
    
    hits = 0
    for s in pos:
        rank = np.sum(neg >= s) + 1
        if rank <= k:
            hits += 1
    
    return hits / len(pos)

pos_scores = scores[scores[:, 1] == 1][:, 0]
neg_scores = scores[scores[:, 1] == 0][:, 0]

all_ranks = []
for s in pos_scores:
    rank = np.sum(neg_scores >= s) + 1
    all_ranks.append(rank)

TOP_KS = [1, 5, 10, 50, 100, 200]
recall_values = [recall_at_k(scores, k) for k in TOP_KS]

print("\n" + "="*60)
print("LEARNED SIMILARITY MODEL EVALUATION")
print("="*60)
for k, r in zip(TOP_KS, recall_values):
    print(f"Recall@{k:<3} = {r:.3f}")

print(f"Mean Rank  = {np.mean(all_ranks):.2f}")
print(f"Median     = {np.median(all_ranks):.0f}")
print("="*60)

# =====================================================
# PLOTTING
# =====================================================
ranks = np.array(all_ranks)

# Recall@K curve
plt.figure(figsize=(8,5))
plt.plot(TOP_KS, recall_values, marker='o', linestyle='-', color='b', label='Learned Model')
baseline_recall = [0.075, 0.175, 0.275, 0.588, 0.850, 1.000]
plt.plot(TOP_KS, baseline_recall, marker='s', linestyle='--', color='r', label='Baseline (Raw Fused)')
plt.xticks(TOP_KS)
plt.xlabel("Top-K")
plt.ylabel("Recall@K")
plt.title("Learned Similarity Model vs Baseline")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(PLOT_DIR / "recall_comparison.png")
print(f"\n✅ Saved: {PLOT_DIR / 'recall_comparison.png'}")
plt.close()

# Rank histogram
plt.figure(figsize=(8,5))
plt.hist(ranks, bins=30, color='orange', edgecolor='black', alpha=0.7)
plt.axvline(np.median(ranks), color='red', linestyle='--', linewidth=2, label=f'Median = {np.median(ranks):.0f}')
plt.axvline(np.mean(ranks), color='blue', linestyle='--', linewidth=2, label=f'Mean = {np.mean(ranks):.2f}')
plt.xlabel("Rank of Correct Cover")
plt.ylabel("Number of Queries")
plt.title("Distribution of Ranks (Learned Model)")
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(PLOT_DIR / "rank_histogram_learned.png")
print(f"✅ Saved: {PLOT_DIR / 'rank_histogram_learned.png'}")
plt.close()

# CDF of ranks
sorted_ranks = np.sort(ranks)
cdf = np.arange(1, len(ranks)+1) / len(ranks)

plt.figure(figsize=(8,5))
plt.plot(sorted_ranks, cdf, marker='.', linestyle='-', color='green', linewidth=2)
plt.xlabel("Rank")
plt.ylabel("Cumulative % of Queries")
plt.title("CDF of Correct Cover Ranks (Learned Model)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(PLOT_DIR / "rank_cdf_learned.png")
print(f"✅ Saved: {PLOT_DIR / 'rank_cdf_learned.png'}")
plt.close()

print(f"\n✅ All plots saved to: {PLOT_DIR.resolve()}")
print(f"✅ Evaluation complete! Processed {len(scores)} pairs using {len(feature_cache)} unique tracks.")