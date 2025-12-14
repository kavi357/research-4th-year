import pandas as pd
from tqdm import tqdm

from dataset_builder.load_db import (
    load_all_track_ids,
    load_fused_embedding,
    load_audio_features
)
from dataset_builder.similarity import cosine_sim


def build_dataset(
    top_pct=0.05,
    mid_pct=0.05,
    bottom_pct=0.05,
    out_csv="data/labeled_pairs_3class.csv"
):
    track_ids = load_all_track_ids()
    print(f"🔍 Building 3-class dataset | Tracks: {len(track_ids)}")

    rows = []
    seen = set()

    for t1 in tqdm(track_ids):
        emb1 = load_fused_embedding(t1)
        af1 = load_audio_features(t1)

        if emb1 is None or af1 is None:
            continue

        tempo1, mfcc1, chroma1, _, _, pitch_med1 = af1
        scores = []

        for t2 in track_ids:
            if t1 == t2:
                continue

            emb2 = load_fused_embedding(t2)
            af2 = load_audio_features(t2)

            if emb2 is None or af2 is None:
                continue

            tempo2, mfcc2, chroma2, _, _, pitch_med2 = af2

            fused_cos = cosine_sim(emb1, emb2)
            mfcc_cos = cosine_sim(mfcc1, mfcc2)
            chroma_cos = cosine_sim(chroma1, chroma2)

            pitch_median_diff = abs(pitch_med1 - pitch_med2)
            tempo_diff = abs(tempo1 - tempo2)

            scores.append((
                fused_cos, t1, t2,
                fused_cos, mfcc_cos, chroma_cos,
                pitch_median_diff, tempo_diff
            ))

        if not scores:
            continue

        # -----------------------------
        # SORT BY FUSED COSINE
        # -----------------------------
        scores.sort(key=lambda x: x[0], reverse=True)
        n = len(scores)

        top_k = max(1, int(top_pct * n))
        mid_k = max(1, int(mid_pct * n))
        bot_k = max(1, int(bottom_pct * n))

        mid_start = (n // 2) - (mid_k // 2)
        mid_end = mid_start + mid_k

        # -----------------------------
        # HIGH SIMILARITY → 1
        # -----------------------------
        for s in scores[:top_k]:
            key = frozenset({s[1], s[2]})
            if key in seen:
                continue

            rows.append({
                "track1": s[1],
                "track2": s[2],
                "fused_cos": s[3],
                "mfcc_cos": s[4],
                "chroma_cos": s[5],
                "pitch_median_diff": s[6],
                "tempo_diff": s[7],
                "label": 1
            })
            seen.add(key)

        # -----------------------------
        # MEDIUM SIMILARITY → 0.5
        # -----------------------------
        for s in scores[mid_start:mid_end]:
            key = frozenset({s[1], s[2]})
            if key in seen:
                continue

            rows.append({
                "track1": s[1],
                "track2": s[2],
                "fused_cos": s[3],
                "mfcc_cos": s[4],
                "chroma_cos": s[5],
                "pitch_median_diff": s[6],
                "tempo_diff": s[7],
                "label": 0.5
            })
            seen.add(key)

        # -----------------------------
        # LOW SIMILARITY → 0
        # -----------------------------
        for s in scores[-bot_k:]:
            key = frozenset({s[1], s[2]})
            if key in seen:
                continue

            rows.append({
                "track1": s[1],
                "track2": s[2],
                "fused_cos": s[3],
                "mfcc_cos": s[4],
                "chroma_cos": s[5],
                "pitch_median_diff": s[6],
                "tempo_diff": s[7],
                "label": 0
            })
            seen.add(key)

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)

    print(f"✔ Dataset saved → {out_csv}")
    print(f"✔ Total labeled pairs: {len(df)}")
