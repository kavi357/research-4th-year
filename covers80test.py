import os
import re
from collections import defaultdict

DATASET_PATH = r"D:\SLIIT\4Y1S\Research Copyright New\data\covers80\coversongs\covers32k"   # <-- change if different

def extract_title(filename):
    # Remove extension
    name = os.path.splitext(filename)[0]

    # Keep only characters (remove track numbers etc.)
    name = re.sub(r'\d+', '', name)

    # Split by '+' or '-' or '_'
    parts = re.split(r'[+_\-]', name)

    # Song title assumed to be the last token
    title = parts[-1].strip().lower()

    return title

def analyze_songs(path):
    song_lookup = defaultdict(list)

    for root, dirs, files in os.walk(path):
        for f in files:
            if f.lower().endswith(('.mp3', '.flac', '.wav', '.ogg')):
                title = extract_title(f)
                song_lookup[title].append(os.path.join(root, f))

    return song_lookup


if __name__ == "__main__":
    songs = analyze_songs(DATASET_PATH)

    print("\n--- COVER RELATIONSHIPS ---")

    for title, files in songs.items():
        print(f"\n🎵 Song: {title}")
        if len(files) > 1:
            print(f" ✔ Found {len(files)} versions (cover(s) exist):")
            for f in files:
                print("   -", f)
        else:
            print(" ❌ Only one version found (no cover).")
