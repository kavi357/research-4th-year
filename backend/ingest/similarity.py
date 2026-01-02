import numpy as np
import librosa
from scipy.stats import percentileofscore

# ----------------------------
# Utility
# ----------------------------
def l2_norm(x):
    return x / (np.linalg.norm(x) + 1e-9)

# ----------------------------
# Tempo (weak signal)
# ----------------------------
def tempo_similarity(t1, t2):
    if t1 <= 0 or t2 <= 0:
        return 0.0
    diff = abs(t1 - t2)
    return float(np.exp(-diff / 15.0))

# ----------------------------
# Pitch (display + weak signal)
# ----------------------------
def pitch_similarity(p1, p2):
    if p1 <= 0 or p2 <= 0:
        return 0.0
    cents = abs(1200 * np.log2(p1 / p2))
    cents = cents % 1200
    cents = min(cents, 1200 - cents)
    return float(np.exp(-cents / 300.0))

# ----------------------------
# Rhythm (DTW-based)
# ----------------------------
def rhythm_similarity_dtw(r1_bytes, r2_bytes):
    if r1_bytes is None or r2_bytes is None:
        return 0.0

    r1 = np.frombuffer(r1_bytes, dtype=np.float32)
    r2 = np.frombuffer(r2_bytes, dtype=np.float32)

    if len(r1) < 10 or len(r2) < 10:
        return 0.0

    r1 = l2_norm(r1)
    r2 = l2_norm(r2)

    D, _ = librosa.sequence.dtw(
        r1.reshape(1, -1),
        r2.reshape(1, -1),
        metric="euclidean"
    )
    return float(np.exp(-np.mean(D)))

# ----------------------------
# Percentile (TOP-K only)
# ----------------------------
def robust_percentile(results, key="score", top_k=20):
    if not results:
        return 0.0

    top = sorted(results, key=lambda x: x[key], reverse=True)[:top_k]
    scores = [x[key] for x in top]

    best = scores[0]
    if best < 0.5:
        return round(best * 0.6, 3)

    perc = percentileofscore(scores, best) / 100.0
    return round(float(perc), 3)
