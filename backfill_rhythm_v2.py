#backfill_rythem_v2

import sqlite3
import librosa
from pathlib import Path
from backend.ingest.extract_rhythm_features_v2 import extract_rhythm_v2

DB_PATH = "database/music.db"

def backfill_rhythm_v2():
    print("🔍 Opening database:", DB_PATH)

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("""
        SELECT t.id, t.file_path
        FROM tracks t
        JOIN audio_features af ON t.id = af.track_id
        WHERE af.common_rhythm IS NULL
           OR af.rhythm_fingerprint IS NULL
    """)

    rows = cur.fetchall()
    print(f"🎵 Tracks to ingest: {len(rows)}")

    if len(rows) == 0:
        print("✅ Nothing to ingest. Exiting.")
        conn.close()
        return

    for i, (track_id, path) in enumerate(rows, 1):
        print(f"\n[{i}/{len(rows)}] Processing track_id={track_id}")
        print("   Path:", path)

        if not Path(path).exists():
            print("   ❌ File not found")
            continue

        try:
            print("   ⏳ Loading audio...")
            y, sr = librosa.load(path, sr=None, mono=True, duration=60)

            print(f"   ✔ Loaded ({len(y)/sr:.1f}s)")

            print("   ⏳ Extracting rhythm...")
            common, fp = extract_rhythm_v2(y, sr)

            if common is None or fp is None:
                print("   ❌ Extraction failed")
                continue

            cur.execute("""
                UPDATE audio_features
                SET common_rhythm = ?, rhythm_fingerprint = ?
                WHERE track_id = ?
            """, (common.tobytes(), fp.tobytes(), track_id))

            print(
                f"   ✅ Saved common_dim={len(common)}, "
                f"fp_len={len(fp)}"
            )

        except Exception as e:
            print("   ❌ ERROR:", e)

    conn.commit()
    conn.close()
    print("\n🎉 Rhythm ingestion finished")

if __name__ == "__main__":
    backfill_rhythm_v2()
