# dataset_builder/generate_pairs.py

import random
import sqlite3
from dataset_builder.load_db import load_all_track_ids, DB_PATH


def generate_positive_pairs():
    """
    Positive pairs = Covers80 (each song has 2 versions: a/b)
    """
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("SELECT id, title FROM tracks WHERE dataset='covers80'")
    rows = cur.fetchall()
    conn.close()

    groups = {}
    for track_id, title in rows:
        key = title[:-1]  # remove the "a" or "b"
        groups.setdefault(key, []).append(track_id)

    positive_pairs = []
    for g in groups.values():
        if len(g) == 2:
            positive_pairs.append((g[0], g[1], 1))

    print(f"✔ Covers80 positive pairs found: {len(positive_pairs)}")
    return positive_pairs


def generate_negative_pairs(count=300):
    """
    Negative = completely random pairs from full dataset
    """
    all_ids = load_all_track_ids()

    neg_pairs = set()

    while len(neg_pairs) < count:
        a, b = random.sample(all_ids, 2)
        if a != b:
            neg_pairs.add((a, b, 0))

    print(f"✔ Negative pairs generated: {len(neg_pairs)}")

    return list(neg_pairs)
