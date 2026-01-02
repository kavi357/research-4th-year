import sqlite3
import numpy as np
import random

DB_PATH = "database/music.db"

def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

cur.execute("""
SELECT f.track_id, f.embedding, t.dataset
FROM fused_embeddings f
JOIN tracks t ON f.track_id = t.id
""")

rows = cur.fetchall()

by_dataset = {}
for tid, blob, ds in rows:
    emb = np.frombuffer(blob, np.float32)
    by_dataset.setdefault(ds, []).append(emb)

# positives: same song
pos = []
neg = []

for ds in by_dataset:
    emb_list = by_dataset[ds]
    if len(emb_list) < 2:
        continue

    pos.append(cosine(emb_list[0], emb_list[0]))

    # HARD NEGATIVE: different dataset
    other_ds = random.choice([d for d in by_dataset if d != ds])
    neg.append(cosine(emb_list[0], by_dataset[other_ds][0]))

print("Positive mean:", np.mean(pos))
print("Hard negative mean:", np.mean(neg))
