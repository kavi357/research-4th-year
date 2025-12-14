import sqlite3
import numpy as np

DB_PATH = 'database/music.db'


def bytes_to_array(blob):
    """Convert BLOB from DB to numpy array."""
    if blob is None:
        return np.array([], dtype=np.float32)
    return np.frombuffer(blob, dtype=np.float32)


def verify_crepe_data(limit=None):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # Total tracks
    cur.execute("SELECT COUNT(*) FROM audio_features")
    total = cur.fetchone()[0]

    # Tracks missing pitch_median
    cur.execute("SELECT COUNT(*) FROM audio_features WHERE pitch_median IS NULL")
    missing = cur.fetchone()[0]

    print(f"Total tracks in audio_features : {total}")
    print(f"Tracks missing pitch_median    : {missing}\n")

    # Build query
    query = """
        SELECT track_id, pitch_times, pitch_freqs, pitch_conf, pitch_median
        FROM audio_features
    """

    if limit is not None:
        query += " LIMIT ?"
        cur.execute(query, (limit,))
    else:
        cur.execute(query)

    rows = cur.fetchall()

    for track_id, times, freqs, conf, median in rows:
        print(f"Track {track_id}:")
        print("  pitch_times :", bytes_to_array(times))
        print("  pitch_freqs :", bytes_to_array(freqs))
        print("  pitch_conf  :", bytes_to_array(conf))
        print(f"  pitch_median: {median}")
        print("-" * 50)

    conn.close()


if __name__ == "__main__":
    verify_crepe_data()   # ✅ SHOW ALL SONGS
    # verify_crepe_data(limit=5)  # optional: show only first 5
