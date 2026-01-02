import json
import random
from itertools import combinations
from pathlib import Path

random.seed(42)

ROOT = Path(__file__).resolve().parents[2] / "data"

# ---------------- DATASET PATHS ----------------
COVERS_ROOT = ROOT / "covers80" / "coversongs" / "covers32k"
GTZAN_ROOT = ROOT / "gtzan"
FMA_ROOT = ROOT / "fma_small"
MAGNA_ROOT = ROOT / "magnatagatune"

OUTPUT = Path(__file__).resolve().parent / "pairs_new.json"

# ---------------- HELPERS ----------------
def extract_cover_id(path: Path):
    """Covers80 song identity = parent folder"""
    return path.parent.name.lower()

def collect_audio_files(root, exts=(".mp3", ".wav")):
    return [p for p in root.rglob("*") if p.suffix.lower() in exts]

# ---------------- COLLECT FILES ----------------
covers_files = collect_audio_files(COVERS_ROOT)
gtzan_files = collect_audio_files(GTZAN_ROOT)
fma_files = collect_audio_files(FMA_ROOT)
magna_files = collect_audio_files(MAGNA_ROOT)

# ---------------- GROUP COVERS ----------------
covers_by_song = {}
for p in covers_files:
    sid = extract_cover_id(p)
    covers_by_song.setdefault(sid, []).append(str(p))

# ---------------- POSITIVE PAIRS (COVERS80) ----------------
positive_pairs = []
for song_id, files in covers_by_song.items():
    if len(files) >= 2:
        for a, b in combinations(files, 2):
            positive_pairs.append({
                "query": a,
                "db": b,
                "label": 1,
                "source": "covers80"
            })

# Sample exactly 80 positives
positive_pairs = random.sample(positive_pairs, 80)

# ---------------- NEGATIVE PAIRS ----------------
negative_pairs = []

def add_negatives(files, count, tag):
    added = 0
    while added < count:
        q = random.choice(positive_pairs)["query"]
        d = str(random.choice(files))
        negative_pairs.append({
            "query": q,
            "db": d,
            "label": 0,
            "source": tag
        })
        added += 1

# 30 hard negatives (covers80 different song)
covers_flat = [str(p) for p in covers_files]
add_negatives(covers_flat, 30, "covers80_hard")

# 20 GTZAN
add_negatives(gtzan_files, 20, "gtzan")

# 20 FMA-small
add_negatives(fma_files, 20, "fma_small")

# 10 MagnaTagATune
add_negatives(magna_files, 10, "magnaTagATune")

pairs = positive_pairs + negative_pairs
random.shuffle(pairs)

# ---------------- SAVE ----------------
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT, "w") as f:
    json.dump(pairs, f, indent=2)

print("✅ Dataset built")
print("Positive:", len(positive_pairs))
print("Negative:", len(negative_pairs))
print("Total:", len(pairs))
