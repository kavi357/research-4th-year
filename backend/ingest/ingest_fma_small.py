import os
import sqlite3
import numpy as np
from pathlib import Path

from .preprocess import preprocess_audio
from .extract_features import extract_audio_features, insert_audio_features
from .extract_embeddings import (
    extract_openl3_embedding,
    extract_yamnet_embedding,
    extract_fused_embedding,
    insert_openl3_embedding,
    insert_yamnet_embedding,
    insert_fused_embedding
)

# Global DB path
DB_PATH = Path(__file__).resolve().parents[2] / "database" / "music.db"
DB_PATH = str(DB_PATH)


def insert_track(db, title, file_path, duration, dataset):
    conn = sqlite3.connect(db)
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO tracks (title, file_path, duration, dataset)
        VALUES (?, ?, ?, ?)
    """, (title, file_path, duration, dataset))
    conn.commit()
    track_id = cur.lastrowid
    conn.close()
    return track_id


def ingest_fma(root_folder, max_songs=12):

    root_folder = Path(root_folder)
    processed_count = 0

    # Load already ingested file paths
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT file_path FROM tracks")
    existing_files = set(row[0] for row in cur.fetchall())
    conn.close()

    print(f"\n🎵 Starting FMA ingestion from: {root_folder}\n")

    for subfolder in sorted(root_folder.rglob("*")):
        if not subfolder.is_dir():
            continue

        for file in subfolder.iterdir():
            if processed_count >= max_songs:
                print(f"🚫 Stopping – processed {max_songs} new songs.")
                return

            if file.suffix.lower() not in [".wav", ".mp3", ".flac", ".m4a"]:
                continue

            if str(file) in existing_files:
                print(f"⏭ Skipping already ingested: {file}")
                continue

            print(f"▶ Processing {file}")

            try:
                # --- PREPROCESS (60s segment) ---
                y, duration, sr = preprocess_audio(file)

                # --- INSERT TRACK ---
                track_id = insert_track(DB_PATH, file.stem, str(file), duration, "fma")

                # --- AUDIO FEATURES ---
                tempo, mfcc, chroma, pitch_times, pitch_freqs, pitch_conf, pitch_median = extract_audio_features(y, sr)
                insert_audio_features(
                    DB_PATH,
                    track_id,
                    tempo,
                    mfcc,
                    chroma,
                    pitch_times=pitch_times,
                    pitch_freqs=pitch_freqs,
                    pitch_conf=pitch_conf,
                    pitch_median=pitch_median
                )

                # --- EMBEDDINGS ---
                openl3_emb = extract_openl3_embedding(y, sr)
                insert_openl3_embedding(track_id, openl3_emb)

                yamnet_emb = extract_yamnet_embedding(y, sr)
                insert_yamnet_embedding(track_id, yamnet_emb)

                fused_emb = extract_fused_embedding(y, sr)
                insert_fused_embedding(track_id, fused_emb)

                processed_count += 1
                print(f"✓ Saved to DB (track_id={track_id})")

            except Exception as e:
                print(f"❌ ERROR processing {file} — {e}")

    print("\n🎉 Finished FMA ingestion.\n")


if __name__ == "__main__":
    ingest_fma("data/fma_small")
