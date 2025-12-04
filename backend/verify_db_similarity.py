import sqlite3
import numpy as np
from pathlib import Path

# ----------------------------------------
# IMPORT FIX (use relative package import)
# ----------------------------------------
from backend.ingest.preprocess import preprocess_audio

# ----------------------------------------
# ALWAYS use database from project root
# ----------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "database" / "music.db"


# ----------------------------------------
# Count rows
# ----------------------------------------
def count_rows():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    tables = ["tracks", "audio_features", "embeddings", "yamnet_embeddings", "fused_embeddings"]

    print("\n========= DATABASE ROW COUNTS =========")
    for t in tables:
        try:
            cur.execute(f"SELECT COUNT(*) FROM {t}")
            print(f"{t}: {cur.fetchone()[0]} rows")
        except:
            print(f"{t}: TABLE NOT FOUND")

    conn.close()


# ----------------------------------------
# Verify track files
# ----------------------------------------
def verify_tracks():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    print("\n========= VERIFY TRACKS =========")
    cur.execute("SELECT id, file_path FROM tracks")
    rows = cur.fetchall()

    for track_id, file_path in rows:
        file_path = Path(file_path)

        if not file_path.exists():
            print(f"❌ Track {track_id}: Missing file → {file_path}")
            continue

        y, duration, sr = preprocess_audio(file_path)

        issues = []
        if round(duration, 1) != 60.0:
            issues.append(f"duration mismatch ({duration:.2f}s)")
        if sr != 48000:
            issues.append(f"sample rate mismatch ({sr})")
        if y.ndim != 1:
            issues.append("not mono")

        if issues:
            print(f"⚠️ Track {track_id} ({file_path.name}): " + ", ".join(issues))
        else:
            print(f"✓ Track {track_id}: OK (60s, 48kHz, mono)")

    conn.close()


# ----------------------------------------
# Verify MFCC / Chroma / Tempo
# ----------------------------------------
def verify_audio_features():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    print("\n========= VERIFY AUDIO FEATURES =========")
    cur.execute("SELECT track_id, tempo, mfcc, chroma FROM audio_features")
    rows = cur.fetchall()

    for track_id, tempo, mfcc_blob, chroma_blob in rows:
        mfcc = np.frombuffer(mfcc_blob, dtype=np.float32)
        chroma = np.frombuffer(chroma_blob, dtype=np.float32)

        if mfcc.size % 20 != 0:
            print(f"❌ Track {track_id}: Broken MFCC shape ({mfcc.size})")
        if chroma.size % 12 != 0:
            print(f"❌ Track {track_id}: Broken Chroma shape ({chroma.size})")
        if tempo <= 0:
            print(f"⚠️ Track {track_id}: Tempo suspicious ({tempo})")

        print(f"✓ Track {track_id}: Features OK → MFCC={mfcc.shape}, Chroma={chroma.shape}, Tempo={tempo}")

    conn.close()


# ----------------------------------------
# Verify embeddings
# ----------------------------------------
def verify_embeddings():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    print("\n========= VERIFY EMBEDDINGS =========")

    checks = [
        ("embeddings", 512),
        ("yamnet_embeddings", 1024),
        ("fused_embeddings", 1536),
    ]

    for table, expected_dim in checks:
        cur.execute(f"SELECT track_id, embedding, dim FROM {table}")
        rows = cur.fetchall()

        for track_id, emb_blob, dim in rows:
            emb = np.frombuffer(emb_blob, dtype=np.float32)

            if emb.size != dim:
                print(f"❌ Track {track_id} in {table}: Blob size mismatch ({emb.size} != {dim})")

            if dim != expected_dim:
                print(f"⚠️ Track {track_id} in {table}: Unexpected embedding dim {dim} (expected {expected_dim})")

            print(f"✓ Track {track_id} in {table}: Embedding OK → dim={dim}")

    conn.close()


# ----------------------------------------
# Main
# ----------------------------------------
if __name__ == "__main__":
    print("🔍 Starting full database + audio verification…")

    print("Using DB:", DB_PATH)
    count_rows()
    verify_tracks()
    verify_audio_features()
    verify_embeddings()

    print("\n🎉 Verification Completed.")



#import sqlite3

#DB_PATH = "database/music.db"

# IDs to delete
#delete_ids = list(range(84, 89))  # 84 to 88 inclusive

#conn = sqlite3.connect(DB_PATH)
#cur = conn.cursor()

#for track_id in delete_ids:
 #   print(f"Deleting track_id={track_id}…")
    
    # Delete from embeddings
  #  cur.execute("DELETE FROM embeddings WHERE track_id=?", (track_id,))
   # cur.execute("DELETE FROM yamnet_embeddings WHERE track_id=?", (track_id,))
    #cur.execute("DELETE FROM fused_embeddings WHERE track_id=?", (track_id,))
    
    # Delete from audio features
    #cur.execute("DELETE FROM audio_features WHERE track_id=?", (track_id,))
    
    # Delete from tracks
    #cur.execute("DELETE FROM tracks WHERE id=?", (track_id,))

#conn.commit()
#conn.close()
#print("✅ Deletion complete.")
