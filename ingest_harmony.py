import sqlite3
import librosa
import numpy as np
from backend.harmony.chord_extraction import extract_chord_sequence
from pathlib import Path

DB_PATH = "database/music.db"

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

cur.execute("SELECT id, file_path FROM tracks")
tracks = cur.fetchall()

total = len(tracks)
print(f"🎵 Total tracks to process: {total}\n")

for idx, (tid, path) in enumerate(tracks, start=1):
    print(f"▶ [{idx}/{total}] Processing Track ID: {tid}")
    print(f"   📂 File: {path}")

    try:
        # Load audio
        y, sr = librosa.load(path, sr=16000, mono=True, duration=60)
        print(f"   ✅ Audio loaded (sr={sr}, samples={len(y)})")

        # Extract harmony
        chords, beats = extract_chord_sequence(y, sr)
        print(f"   🎼 Chords extracted: {len(chords)}")
        print(f"   🥁 Beats extracted: {len(beats)}")

        # Store in DB
        cur.execute("""
            INSERT OR REPLACE INTO harmony_features
            VALUES (?, ?, ?)
        """, (
            tid,
            chords.tobytes(),
            beats.tobytes()
        ))

        print(f"   💾 Saved to database\n")

    except Exception as e:
        print(f"   ❌ Error processing Track ID {tid}: {e}\n")

conn.commit()
conn.close()

print("✅ Harmony feature extraction completed for all tracks.")
