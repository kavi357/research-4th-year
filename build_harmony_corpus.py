import sqlite3
import json
import numpy as np
from backend.harmony.utils import convert_to_roman_sequence

DB_PATH = "database/music.db"
OUTPUT_PATH = "data/harmony_corpus.json"

def main():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("""
        SELECT h.track_id, h.chord_sequence
        FROM harmony_features h
    """)

    corpus = []

    for track_id, chord_blob in cur.fetchall():
        chords = np.frombuffer(chord_blob, dtype=np.int8)

        # ---- Estimate key using simple histogram fallback ----
        key_pc = int(np.bincount(chords % 12).argmax())

        roman_seq = convert_to_roman_sequence(chords, key_pc)

        # Filter unknowns
        roman_seq = [r for r in roman_seq if r != "X"]

        if len(roman_seq) >= 4:
            corpus.append({
                "song_id": int(track_id),
                "roman": roman_seq
            })

    conn.close()

    with open(OUTPUT_PATH, "w") as f:
        json.dump(corpus, f, indent=2)

    print(f"✅ Harmony corpus saved: {len(corpus)} songs")

if __name__ == "__main__":
    main()
