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


def ingest_gtzan(root_folder, max_songs=10):
    root_folder = Path(root_folder)
    processed_count = 0

    # Load already ingested file paths
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT file_path FROM tracks")
    existing_files = set(row[0] for row in cur.fetchall())
    conn.close()

    # GTZAN files organized as: genre/track.wav
    for genre_folder in sorted(root_folder.iterdir()):
        if not genre_folder.is_dir():
            continue

        for file in genre_folder.iterdir():
            if processed_count >= max_songs:
                print(f"🚫 Stopping – already processed {max_songs} new songs.")
                return

            if not file.suffix.lower() in [".wav", ".mp3", ".flac", ".m4a"]:
                continue

            if str(file) in existing_files:
                # print(f"⏭ Skipping already ingested track: {file}")
                continue

            print("▶ Processing", file)

            # --- PREPROCESS IN MEMORY (60 sec max, but GTZAN is 30 sec) ---
            y, duration, sr = preprocess_audio(file)

            # --- INSERT TRACK INTO DB ---
            track_id = insert_track(DB_PATH, file.stem, str(file), duration, "gtzan")

            # --- EXTRACT AUDIO FEATURES ---
            tempo, mfcc, chroma, pitch_times, pitch_freqs, pitch_conf, pitch_median = extract_audio_features(y, sr)
            insert_audio_features(DB_PATH, track_id, tempo, mfcc, chroma,
                                  pitch_times=pitch_times, pitch_freqs=pitch_freqs,
                                  pitch_conf=pitch_conf, pitch_median=pitch_median)

            

            fused_emb = extract_fused_embedding(y, sr)
            insert_fused_embedding(track_id, fused_emb)

            processed_count += 1
            print(f"✓ Saved to DB: Track ID {track_id}")

    print("🎉 Finished scanning GTZAN.")


if __name__ == "__main__":
    ingest_gtzan("data/gtzan")   # <-- CHANGE ONLY THIS PATH
