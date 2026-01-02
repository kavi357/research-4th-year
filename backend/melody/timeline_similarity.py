import numpy as np


def normalize_pitch(pitch_seq):
    pitch_seq = np.asarray(pitch_seq, dtype=np.float32)
    if pitch_seq.size == 0:
        return pitch_seq
    return pitch_seq - pitch_seq.mean()


def timeline_similarity(
    query_pitch: list,
    db_pitch: list,
    window_size: int = 16,
    hop_size: int = 8
) -> dict:
    """
    Fast segment-wise melodic similarity using vectorized cosine similarity
    """

    q = normalize_pitch(query_pitch)
    d = normalize_pitch(db_pitch)

    max_len = min(len(q), len(d))
    if max_len < window_size:
        return {
            "window_size": window_size,
            "hop_size": hop_size,
            "segments": [],
            "similarities": []
        }

    starts = range(0, max_len - window_size + 1, hop_size)

    similarities = []
    segments = []

    for start in starts:
        q_seg = q[start:start + window_size]
        d_seg = d[start:start + window_size]

        # ---- FAST cosine similarity ----
        denom = (np.linalg.norm(q_seg) * np.linalg.norm(d_seg))
        sim = 0.0 if denom == 0 else np.dot(q_seg, d_seg) / denom
        sim = float(np.clip(sim, 0.0, 1.0))

        similarities.append(sim)
        segments.append({
            "start_note": start,
            "end_note": start + window_size - 1,
            "similarity": round(sim, 4)
        })

    return {
        "window_size": window_size,
        "hop_size": hop_size,
        "segments": segments,
        "similarities": similarities
    }
