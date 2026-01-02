# cache_siamese_embeddings.py
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn.functional as F
import sqlite3
import numpy as np
from pathlib import Path
from models.siamese_encoder.siamese_encoder import SimilarityNetwork
from backend.ingest.siamese_feature_extractor import extract_siamese_features

# -------------------------------
# Paths
# -------------------------------
ROOT = Path(__file__).resolve().parent 
MODEL_PATH = ROOT / "models/siamese_encoder/music_siamese_encoder.pth" 
DB_PATH = ROOT / "database/music.db"
# -------------------------------

# -------------------------------
# Load trained Siamese model
# -------------------------------
model = SimilarityNetwork(input_dim=1570)
checkpoint = torch.load(MODEL_PATH, map_location="cpu")
model.load_state_dict(checkpoint["model_state"])
model.eval()

# -------------------------------
# Connect to DB
# -------------------------------
conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

# Drop existing table (if any) and recreate
cur.execute("DROP TABLE IF EXISTS siamese_embeddings")
cur.execute("""
    CREATE TABLE siamese_embeddings (
        track_id INTEGER PRIMARY KEY,
        embedding BLOB
    )
""")
conn.commit()

# -------------------------------
# Compute embeddings for all tracks
# -------------------------------
cur.execute("SELECT id, file_path, title FROM tracks")
tracks = cur.fetchall()

print(f"⚡ Computing Siamese embeddings for {len(tracks)} tracks...")

for track_id, path, title in tracks:
    feats = extract_siamese_features(path)
    x = torch.tensor(feats).unsqueeze(0)
    with torch.no_grad():
        emb = model(x).squeeze(0).numpy()
    cur.execute(
        "INSERT OR REPLACE INTO siamese_embeddings (track_id, embedding) VALUES (?, ?)",
        (track_id, emb.tobytes())
    )
    print(f"✔ Track {track_id}: {title} → cached")

# -------------------------------
# Finish
# -------------------------------
conn.commit()
conn.close()
print("✅ All Siamese embeddings cached successfully!")
