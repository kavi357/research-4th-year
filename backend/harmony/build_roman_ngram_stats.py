"""
Build Roman-numeral n-gram corpus statistics
Used for Harmonic Rarity / Commonness scoring

This script:
- Reads chord sequences from harmony_features
- Converts to Roman numerals (key-invariant)
- Builds 3–5 gram frequency statistics
- Stores normalized frequencies in roman_ngram_stats
"""

import sqlite3
import numpy as np
from pathlib import Path

from .utils import convert_to_roman_sequence

# ==========================
# PATHS
# ==========================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = PROJECT_ROOT / "database" / "music.db"

# Roman n-gram sizes
N_GRAMS = [3, 4, 5]

# ==========================
# HELPERS
# ==========================
def extract_ngrams(seq, n):
    """
    Extract Roman n-grams, skipping non-diatonic ('X')
    """
    grams = []
    for i in range(len(seq) - n + 1):
        window = seq[i:i+n]
        if "X" in window:
            continue
        grams.append("-".join(window))
    return grams


def estimate_key_from_chords(chords):
    """
    Fast key estimation fallback using chord histogram
    """
    if len(chords) == 0:
        return 0
    return int(np.bincount(chords % 12).argmax())


# ==========================
# MAIN
# ==========================
def main():
    print("🎼 Building Roman n-gram corpus statistics...")
    print(f"📂 Database: {DB_PATH}")

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # ----------------------------------
    # Load harmony features (FAST)
    # ----------------------------------
    cur.execute("""
        SELECT h.track_id, h.chord_sequence
        FROM harmony_features h
    """)
    rows = cur.fetchall()

    print(f"🎧 Tracks loaded: {len(rows)}")

    # Init counters
    ngram_counts = {n: {} for n in N_GRAMS}

    # ----------------------------------
    # Process each track
    # ----------------------------------
    for tid, blob in rows:
        try:
            chords = np.frombuffer(blob, dtype=np.int8)

            if len(chords) < 6:
                continue

            key_pc = estimate_key_from_chords(chords)
            roman_seq = convert_to_roman_sequence(chords, key_pc)

            for n in N_GRAMS:
                for ng in extract_ngrams(roman_seq, n):
                    ngram_counts[n][ng] = ngram_counts[n].get(ng, 0) + 1

        except Exception as e:
            print(f"❌ Track {tid} skipped: {e}")

    # ----------------------------------
    # Write to database
    # ----------------------------------
    cur.execute("DELETE FROM roman_ngram_stats")

    for n, grams in ngram_counts.items():
        total = sum(grams.values())
        if total == 0:
            continue

        for ng, count in grams.items():
            freq = count / total
            cur.execute("""
                INSERT OR REPLACE INTO roman_ngram_stats
                (ngram, n, count, frequency)
                VALUES (?, ?, ?, ?)
            """, (ng, n, count, freq))

    conn.commit()
    conn.close()

    print("✅ Roman n-gram stats built successfully")
    print("📊 N-gram sizes:", N_GRAMS)


# ==========================
# ENTRY POINT
# ==========================
if __name__ == "__main__":
    main()
