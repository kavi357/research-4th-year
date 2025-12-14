import sqlite3

DB_PATH = "database/music.db"

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

print("Resetting CREPE pitch columns...")

cur.execute("""
    UPDATE audio_features 
    SET pitch_times = NULL,
        pitch_freqs = NULL,
        pitch_conf = NULL,
        pitch_median = NULL
""")

conn.commit()
conn.close()

print("✔ CREPE columns reset successfully!")
