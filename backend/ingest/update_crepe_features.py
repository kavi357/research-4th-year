import sqlite3
import librosa
from pathlib import Path

from .extract_crepe_only import (
    preprocess_for_crepe,
    extract_crepe_pitch,
    update_crepe_features
)

# DATABASE IS IN ROOT/database/music.db
DB_PATH = str(Path(__file__).resolve().parents[2] / "database" / "music.db")

BATCH_SIZE = 300


def update_all_crepe_features():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # Fetch only songs where pitch_median is NULL
    cur.execute("""
        SELECT tracks.id, tracks.file_path
        FROM tracks
        JOIN audio_features ON tracks.id = audio_features.track_id
        WHERE audio_features.pitch_median IS NULL
    """)

    pending = cur.fetchall()
    conn.close()

    total_pending = len(pending)
    print(f"\n🎵 Songs needing CREPE features: {total_pending}")

    if total_pending == 0:
        print("✔ All tracks already processed. Nothing to do.")
        return

    # Take first batch of 5
    batch = pending[:BATCH_SIZE]
    print(f"\n🔹 Processing batch of {len(batch)} songs...\n")

    for track_id, file_path in batch:
        print(f"  ▶ Track {track_id}: {file_path}")

        try:
            y, sr = librosa.load(file_path, sr=None, mono=True)

            y16, sr16 = preprocess_for_crepe(y, sr)

            pitch_times, pitch_freqs, pitch_conf, pitch_median = \
                extract_crepe_pitch(y16, sr16)

            update_crepe_features(
                DB_PATH,
                track_id,
                pitch_times,
                pitch_freqs,
                pitch_conf,
                pitch_median
            )

            print(f"    ✓ Updated pitch (median={pitch_median:.2f})")

        except Exception as e:
            print(f"    ❌ Error: {e}")

    print("\n🎉 Batch completed! Run again for next 5 songs.\n")


if __name__ == "__main__":
    update_all_crepe_features()
