# evaluation/melody/evaluate_retrieval.py

import sys
import json
import numpy as np
import sqlite3
from pathlib import Path
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, roc_curve
import matplotlib.pyplot as plt

# ---------- PATH SETUP ----------
ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "backend"
sys.path.insert(0, str(BACKEND))

DB_PATH = ROOT / "database" / "music.db"
PAIRS_PATH = ROOT / "evaluation" / "melody" / "pairs.json"
OUTPUT_DIR = ROOT / "evaluation" / "melody" / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------- IMPORTS ----------
from melody.melody_similarity import melody_similarity
from melody.melody_fingerprint import extract_melody_fingerprint
from melody.melody_preprocess import preprocess_melody

# ---------- GET TRACK ID FROM FILE PATH ----------
def get_track_id_from_path(file_path):
    """Look up track_id from file_path in database."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    
    # Normalize path separators
    normalized_path = str(file_path).replace("\\", "/")
    
    cur.execute("""
        SELECT id FROM tracks
        WHERE REPLACE(file_path, '\\', '/') = ?
    """, (normalized_path,))
    
    row = cur.fetchone()
    conn.close()
    
    return row[0] if row else None

# ---------- LOAD MELODY ----------
def load_melody_from_path(file_path):
    """Load melody features given a file path."""
    track_id = get_track_id_from_path(file_path)
    
    if track_id is None:
        return None
    
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        SELECT pitch_freqs, pitch_conf
        FROM audio_features
        WHERE track_id=?
    """, (track_id,))
    row = cur.fetchone()
    conn.close()

    if not row or row[0] is None or row[1] is None:
        return None

    pf = np.frombuffer(row[0], np.float32)
    pc = np.frombuffer(row[1], np.float32)
    return preprocess_melody(pf, pc)

# ---------- EVALUATION ----------
print("="*60)
print("MELODY SIMILARITY EVALUATION - Covers80")
print("="*60)

print(f"\n📂 Loading pairs from: {PAIRS_PATH}")
pairs = json.load(open(PAIRS_PATH))
print(f"📊 Total pairs to evaluate: {len(pairs)}")
print(f"   Positive pairs: {sum(p['label'] == 1 for p in pairs)}")
print(f"   Negative pairs: {sum(p['label'] == 0 for p in pairs)}")

scores = []
skipped = 0
details = []

print("\n🔄 Processing pairs...")
for i, p in enumerate(pairs):
    if (i + 1) % 20 == 0:
        print(f"   Progress: {i+1}/{len(pairs)}")
    
    q = load_melody_from_path(p["query"])
    d = load_melody_from_path(p["db"])

    if q is None or d is None:
        skipped += 1
        continue

    fp_q = extract_melody_fingerprint(q)
    fp_d = extract_melody_fingerprint(d)

    if fp_q[0] is None or fp_d[0] is None:
        skipped += 1
        continue

    sim, motif_len, q_start, q_end, db_start, db_end = melody_similarity(fp_q, fp_d)
    scores.append((sim, p["label"]))
    
    details.append({
        "query": p["query"],
        "db": p["db"],
        "label": p["label"],
        "similarity": float(sim),
        "motif_length": int(motif_len)
    })

scores = np.array(scores)

print("\n" + "="*60)
print(f"✅ Successfully evaluated: {len(scores)} pairs")
print(f"⚠️  Skipped: {skipped} pairs")
print("="*60)

if len(scores) == 0:
    print("\n❌ ERROR: No pairs could be evaluated!")
    sys.exit(1)

# ---------- COMPUTE METRICS ----------
y_true = scores[:, 1]
y_score = scores[:, 0]

# Check class balance
unique, counts = np.unique(y_true, return_counts=True)
print(f"\n📊 Class distribution:")
for label, count in zip(unique, counts):
    print(f"   Label {int(label)}: {count} samples")

if len(unique) < 2:
    print("\n⚠️  WARNING: Only one class present!")
    sys.exit(1)

# ROC-AUC
auc = roc_auc_score(y_true, y_score)

# Test multiple thresholds
thresholds = [0.5, 0.6, 0.7]
best_f1 = 0
best_threshold = 0.6
best_metrics = {}

print(f"\n📊 TESTING THRESHOLDS:")
print("="*60)
for thresh in thresholds:
    y_pred = (y_score > thresh).astype(int)
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
    
    print(f"Threshold: {thresh}")
    print(f"   Precision: {p:.3f}")
    print(f"   Recall:    {r:.3f}")
    print(f"   F1-Score:  {f:.3f}")
    print()
    
    if f > best_f1:
        best_f1 = f
        best_threshold = thresh
        best_metrics = {"precision": p, "recall": r, "f1": f}

print("="*60)
print(f"🎯 FINAL RESULTS (Best Threshold: {best_threshold})")
print("="*60)
print(f"ROC-AUC:    {auc:.3f}")
print(f"Precision:  {best_metrics['precision']:.3f}")
print(f"Recall:     {best_metrics['recall']:.3f}")
print(f"F1-Score:   {best_metrics['f1']:.3f}")
print("="*60)

# ---------- COMPUTE RECALL@K ----------
def recall_at_k(scores, k=5):
    """
    For each positive pair, check if it's in top-K results
    """
    positives = scores[scores[:, 1] == 1]
    negatives = scores[scores[:, 1] == 0]
    
    if len(positives) == 0:
        return 0.0
    
    retrieved = 0
    for pos_score in positives[:, 0]:
        # Count how many negatives have higher score
        rank = np.sum(negatives[:, 0] >= pos_score) + 1
        if rank <= k:
            retrieved += 1
    
    return retrieved / len(positives)

recall_5 = recall_at_k(scores, k=5)
recall_10 = recall_at_k(scores, k=10)

print(f"\n📊 RETRIEVAL METRICS:")
print("="*60)
print(f"Recall@5:   {recall_5:.3f}")
print(f"Recall@10:  {recall_10:.3f}")
print("="*60)

# ---------- SAVE RESULTS ----------
results = {
    "dataset": "Covers80",
    "total_pairs": len(pairs),
    "evaluated_pairs": int(len(scores)),
    "skipped_pairs": int(skipped),
    "best_threshold": float(best_threshold),
    "metrics": {
        "roc_auc": float(auc),
        "precision": float(best_metrics['precision']),
        "recall": float(best_metrics['recall']),
        "f1_score": float(best_metrics['f1']),
        "recall_at_5": float(recall_5),
        "recall_at_10": float(recall_10)
    },
    "pair_details": details
}

results_path = OUTPUT_DIR / "evaluation_results.json"
with open(results_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"\n💾 Results saved to: {results_path}")

# ---------- PLOT ROC CURVE ----------
fpr, tpr, _ = roc_curve(y_true, y_score)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {auc:.3f})')
plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('Melody Similarity - ROC Curve (Covers80)', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()

roc_path = OUTPUT_DIR / "roc_curve.png"
plt.savefig(roc_path, dpi=300, bbox_inches='tight')
print(f"📊 ROC curve saved to: {roc_path}")

# ---------- PLOT SCORE DISTRIBUTION ----------
plt.figure(figsize=(10, 6))

pos_scores = scores[scores[:, 1] == 1, 0]
neg_scores = scores[scores[:, 1] == 0, 0]

plt.hist(pos_scores, bins=30, alpha=0.7, label=f'Positive (n={len(pos_scores)})', color='green')
plt.hist(neg_scores, bins=30, alpha=0.7, label=f'Negative (n={len(neg_scores)})', color='red')
plt.axvline(best_threshold, color='blue', linestyle='--', linewidth=2, label=f'Threshold = {best_threshold}')

plt.xlabel('Melody Similarity Score', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Score Distribution: Positive vs Negative Pairs', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(alpha=0.3, axis='y')
plt.tight_layout()

dist_path = OUTPUT_DIR / "score_distribution.png"
plt.savefig(dist_path, dpi=300, bbox_inches='tight')
print(f"📊 Distribution plot saved to: {dist_path}")

print("\n" + "="*60)
print("✅ EVALUATION COMPLETE")
print("="*60)