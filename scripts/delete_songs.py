import sqlite3
from pathlib import Path

DB_PATH = Path("database/music.db")  # adjust if needed

def delete_last_n_tracks(n=2):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # 1. Get last N track IDs
    cur.execute("SELECT id FROM tracks ORDER BY id DESC LIMIT ?", (n,))
    last_ids = [row[0] for row in cur.fetchall()]

    if not last_ids:
        print("No tracks to delete.")
        conn.close()
        return

    print(f"Deleting track IDs: {last_ids}")

    # 2. Delete from tracks (ON DELETE CASCADE will remove related audio_features/embeddings)
    cur.execute(f"DELETE FROM tracks WHERE id IN ({','.join(['?']*len(last_ids))})", last_ids)

    conn.commit()
    conn.close()
    print(f"✅ Deleted last {n} tracks successfully.")

delete_last_n_tracks(2)
