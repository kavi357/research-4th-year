import json
from collections import Counter
from itertools import islice
from pathlib import Path

# -------- CONFIG --------
NGRAM_SIZE = 3
BASE_DIR = Path(__file__).resolve().parents[2]  # project root

CORPUS_PATH = BASE_DIR / "data" / "harmony_corpus.json"
OUTPUT_PATH = BASE_DIR / "data" / "harmonic_rarity_stats.json"

# ------------------------

def ngrams(seq, n):
    it = iter(seq)
    window = list(islice(it, n))
    if len(window) == n:
        yield tuple(window)
    for x in it:
        window = window[1:] + [x]
        yield tuple(window)

def build_harmonic_rarity(corpus):
    """
    corpus = [
      {"song_id": "...", "roman": ["I","V","vi","IV",...]},
      ...
    ]
    """
    counter = Counter()
    song_count = len(corpus)

    for song in corpus:
        roman = song["roman"]
        seen = set()

        for ng in ngrams(roman, NGRAM_SIZE):
            seen.add(ng)

        # Count once per song (document frequency)
        for ng in seen:
            counter["-".join(ng)] += 1

    rarity = {
        k: 1 - (v / song_count)
        for k, v in counter.items()
    }

    return rarity


def main():
    with open(CORPUS_PATH, "r") as f:
        corpus = json.load(f)

    rarity_stats = build_harmonic_rarity(corpus)

    with open(OUTPUT_PATH, "w") as f:
        json.dump(rarity_stats, f, indent=2)

    print("✅ Harmonic rarity corpus built")


if __name__ == "__main__":
    main()
