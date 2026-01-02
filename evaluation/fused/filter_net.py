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
PAIRS_PATH = ROOT / "evaluation" / "melody" / "pairs_new.json"
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
# FEATURE LOADING FROM DB
# =====================================================
def normalize_path(path):
    """Extract relative path from 'data' folder onwards"""
    p = Path(path)
    parts = p.parts
    try:
        data_idx = parts.index('data')
        relative = Path(*parts[data_idx:])
        return str(relative).replace('\\', '/')
    except ValueError:
        return str(p).replace('\\', '/')

def load_feature_for_path(path):
    """Load feature vector for a single file path"""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    
    # Normalize the path
    normalized = normalize_path(path)
    filename = Path(path).name
    
    # Try multiple matching strategies
    queries = [
        # Strategy 1: Match on normalized path ending
        f"""
        SELECT f.embedding, a.tempo, a.mfcc, a.chroma, a.pitch_freqs
        FROM tracks t
        JOIN fused_embeddings f ON t.id = f.track_id
        JOIN audio_features a ON t.id = a.track_id
        WHERE REPLACE(t.file_path, '\\', '/') LIKE '%{normalized}'
        LIMIT 1
        """,
        # Strategy 2: Match on filename only
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
    
    # Build feature vector
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
# LOAD PAIRS & EVALUATE
# =====================================================
pairs = json.load(open(PAIRS_PATH))
print(f"\n📊 Loaded {len(pairs)} pairs from {PAIRS_PATH.name}")

# Cache features to avoid reloading same tracks
feature_cache = {}

def get_feature_cached(path):
    """Get feature with caching"""
    if path not in feature_cache:
        feature_cache[path] = load_feature_for_path(path)
    return feature_cache[path]

scores = []
skipped = 0
skipped_details = []

print(f"\n🔄 Processing pairs...")

# =====================================================
# EVALUATION LOOP
# =====================================================
for i, p in enumerate(pairs):
    if (i + 1) % 20 == 0:
        print(f"   Processed {i + 1}/{len(pairs)} pairs...")
    
    q_feat = get_feature_cached(p["query"])
    d_feat = get_feature_cached(p["db"])
    
    if q_feat is None or d_feat is None:
        skipped += 1
        if len(skipped_details) < 5:
            skipped_details.append({
                "query": Path(p["query"]).name,
                "db": Path(p["db"]).name,
                "q_missing": q_feat is None,
                "d_missing": d_feat is None
            })
        continue
    
    # Embed through learned model
    zq = embed_learned(q_feat)
    zd = embed_learned(d_feat)
    
    # Compute similarity
    sim = cosine_similarity(zq[None], zd[None])[0][0]
    scores.append((sim, p["label"]))

scores = np.array(scores)

print(f"\n{'='*60}")
print(f"✅ Evaluated pairs: {len(scores)} | ❌ Skipped: {skipped}")
print(f"📦 Unique tracks cached: {len(feature_cache)}")

if skipped > 0 and skipped_details:
    print(f"\n⚠️  Sample skipped pairs:")
    for detail in skipped_details[:3]:
        print(f"  Query: {detail['query']} {'(MISSING)' if detail['q_missing'] else '(OK)'}")
        print(f"  DB: {detail['db']} {'(MISSING)' if detail['d_missing'] else '(OK)'}")
        print()

# =====================================================
# SAFETY CHECK
# =====================================================
if len(scores) == 0:
    print("\n❌ No valid pairs evaluated!")
    print("\n🔧 Debugging steps:")
    print("1. Check if tracks exist in database:")
    print("   SELECT COUNT(*) FROM fused_embeddings;")
    print("2. Check sample paths in database:")
    print("   SELECT file_path FROM tracks LIMIT 5;")
    print("3. Compare with paths in pairs_new.json")
    exit(1)

# =====================================================
# RECALL@K (PAIR-BASED)
# =====================================================
def recall_at_k(scores, k):
    """Calculate Recall@K for pair-based evaluation"""
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

# Calculate metrics
pos_scores = scores[scores[:, 1] == 1][:, 0]
neg_scores = scores[scores[:, 1] == 0][:, 0]

all_ranks = []
for s in pos_scores:
    rank = np.sum(neg_scores >= s) + 1
    all_ranks.append(rank)

print("\n" + "="*60)
print("LEARNED SIMILARITY MODEL EVALUATION")
print("="*60)
for k in [1, 5, 10, 50, 100, 200]:
    print(f"Recall@{k:<3} = {recall_at_k(scores, k):.3f}")

if len(all_ranks) > 0:
    print(f"Mean Rank  = {np.mean(all_ranks):.2f}")
    print(f"Median     = {np.median(all_ranks):.0f}")
print("="*60)

# =====================================================
# COMPARISON WITH BASELINE
# =====================================================
print("\n📊 BASELINE (Raw Fused Cosine):")
print("Recall@1   = 0.075")
print("Recall@5   = 0.175")
print("Recall@10  = 0.275")
print("Recall@50  = 0.588")
print("Recall@100 = 0.850")
print("Recall@200 = 1.000")
print("Mean Rank  = 50.69")
print("Median     = 35")
print("="*60)

print(f"\n✅ Evaluation complete! Processed {len(scores)} pairs using {len(feature_cache)} unique tracks.")