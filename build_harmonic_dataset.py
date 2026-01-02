import sqlite3
import numpy as np
from pathlib import Path

# ================= CONFIG =================
DB_PATH = "database/music.db"
OUT_PATH = "data/harmonic_segments_v2.npz"

SEG_LEN = 200        # frames (~4 sec)
HOP = 100            # 50% overlap
N_BINS = 10          # temporal pooling bins
# ==========================================


def normalize_chroma(C):
    """
    Per-song chroma z-normalization
    C shape: (12, T)
    """
    C = C - C.mean(axis=1, keepdims=True)
    C = C / (C.std(axis=1, keepdims=True) + 1e-6)
    return C


def main():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("""
        SELECT track_id, chroma
        FROM audio_features
        WHERE chroma IS NOT NULL
    """)

    rows = cur.fetchall()
    conn.close()

    print(f"Tracks found: {len(rows)}")
    print("-" * 60)

    Xs = []
    labels = []

    for i, (track_id, blob) in enumerate(rows, 1):
        chroma = np.frombuffer(blob, dtype=np.float32).reshape(12, -1)

        # 🔥 STEP 1: normalize chroma per song
        chroma = normalize_chroma(chroma)

        T = chroma.shape[1]
        if T < SEG_LEN:
            continue

        seg_id = 0
        for start in range(0, T - SEG_LEN + 1, HOP):
            # ---------------- SEGMENT ----------------
            seg = chroma[:, start:start + SEG_LEN]    # (12, 200)

            # 🔥 STEP 2: ΔChroma (melodic motion)
            delta = np.diff(seg, axis=1)              # (12, 199)
            seg = seg[:, 1:]                           # (12, 199)

            feat = np.vstack([seg, delta])            # (24, 199)

            # 🔥 STEP 3: SAFE TEMPORAL POOLING
            T_feat = feat.shape[1]
            T_trim = (T_feat // N_BINS) * N_BINS
            feat = feat[:, :T_trim]

            feat = feat.reshape(24, N_BINS, -1).mean(axis=2)
            seg_feat = feat.flatten()                 # (240,)

            # sanity check (optional but safe)
            assert seg_feat.shape[0] == 240

            Xs.append(seg_feat.astype(np.float32))
            labels.append((track_id, seg_id))
            seg_id += 1

        if i == 1 or i % 50 == 0:
            print(f"✔ Track {i}/{len(rows)} | Segments: {seg_id}")

    # ---------------- SAVE ----------------
    Xs = np.stack(Xs)
    labels = np.array(labels, dtype=object)

    # L2 normalize segment vectors
    Xs = Xs / (np.linalg.norm(Xs, axis=1, keepdims=True) + 1e-8)

    Path("data").mkdir(exist_ok=True)
    np.savez_compressed(OUT_PATH, X=Xs, labels=labels)

    print("-" * 60)
    print("Saved:", OUT_PATH)
    print("Total segments:", Xs.shape[0])
    print("Feature dim:", Xs.shape[1])


if __name__ == "__main__":
    main()
