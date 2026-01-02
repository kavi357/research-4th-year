import sqlite3
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "database" / "music.db"
PAIRS_PATH = ROOT / "evaluation" / "melody" / "pairs_new.json"

# Check database paths
conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()
cur.execute("SELECT file_path FROM tracks WHERE file_path LIKE '%covers80%' LIMIT 5")
db_paths = [r[0] for r in cur.fetchall()]
conn.close()

# Check pairs paths
pairs = json.load(open(PAIRS_PATH))
pair_paths = [pairs[0]["query"], pairs[0]["db"]]

print("Database paths (Covers80):")
for p in db_paths:
    print(f"  {p}")

print("\nPairs.json paths:")
for p in pair_paths:
    print(f"  {p}")