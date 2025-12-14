import sqlite3
import numpy as np
import soundfile as sf
import librosa
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1] / "backend"))

from backend.ingest.preprocess import preprocess_audio


DB_PATH = "database/music.db"

def count_rows():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    tables = ["tracks", "audio_features", "embeddings", "yamnet_embeddings", "fused_embeddings"]

    print("\n========= DATABASE ROW COUNTS =========")
    for t in tables:
        try:
            cur.execute(f"SELECT COUNT(*) FROM {t}")
            print(f"{t}: {cur.fetchone()[0]} rows")
        except sqlite3.OperationalError:
            print(f"{t}: TABLE NOT FOUND")

    conn.close()


def verify_tracks():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    print("\n========= VERIFY TRACKS =========")
    cur.execute("SELECT id, file_path, duration FROM tracks")
    rows = cur.fetchall()

    for track_id, file_path, expected_duration in rows:
        file_path = Path(file_path)

        if not file_path.exists():
            print(f"❌ Track {track_id}: File missing → {file_path}")
            continue

        # Re-preprocess exactly the same way as ingestion
        y, out_duration, out_sr = preprocess_audio(file_path)

        issues = []
        if out_duration != 60.0:
            issues.append(f"duration mismatch ({out_duration:.2f}s)")
        if out_sr != 48000:
            issues.append(f"wrong sample rate ({out_sr})")
        if y.ndim != 1:
            issues.append("not mono audio")

        if issues:
            print(f"⚠️ Track {track_id} ({file_path.name}): " + ", ".join(issues))
        else:
            print(f"✓ Track {track_id}: OK (60s, 48kHz, mono)")

    conn.close()


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
            print(f"❌ Track {track_id}: MFCC shape broken")
        if chroma.size % 12 != 0:
            print(f"❌ Track {track_id}: Chroma shape broken")
        if tempo <= 0:
            print(f"⚠️ Track {track_id}: tempo seems wrong ({tempo})")

        print(f"✓ Track {track_id}: Features OK → MFCC={mfcc.shape}, Chroma={chroma.shape}, Tempo={tempo}")

    conn.close()


def verify_embeddings():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    tables_expected_dims = {
        "embeddings": 512,           # OpenL3
        "yamnet_embeddings": 1024,   # YAMNet
        "fused_embeddings": 1536     # OpenL3 + YAMNet
    }

    print("\n========= VERIFY EMBEDDINGS =========")
    for table, expected_dim in tables_expected_dims.items():
        cur.execute(f"SELECT track_id, embedding, dim FROM {table}")
        rows = cur.fetchall()

        for track_id, emb_blob, dim in rows:
            emb = np.frombuffer(emb_blob, dtype=np.float32)

            if emb.shape[0] != dim:
                print(f"❌ Track {track_id} in {table}: Embedding dimension mismatch {emb.shape[0]} != {dim}")

            if dim != expected_dim:
                print(f"⚠️ Track {track_id} in {table}: Unexpected embedding size {dim} (expected {expected_dim})")

            print(f"✓ Track {track_id} in {table}: Embedding OK → dim={dim}")

    conn.close()


if __name__ == "__main__":
    print("🔍 Starting full database + audio verification…")

    count_rows()
    verify_tracks()
    verify_audio_features()
    verify_embeddings()

    print("\n🎉 Verification Completed.")
