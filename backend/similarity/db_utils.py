import sqlite3
import numpy as np
from pathlib import Path

DB_PATH = Path(__file__).resolve().parents[1] / "database" / "music.db"

def load_audio_features():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("""
        SELECT track_id, tempo, pitch_median
        FROM audio_features
        WHERE pitch_median IS NOT NULL
    """)

    rows = cur.fetchall()
    conn.close()
    return rows
