import numpy as np
from scipy.signal import correlate

def rhythm_fingerprint_similarity(fp1, fp2):
    """
    Measures rhythmic copying likelihood.
    Returns:
    - similarity score
    - evidence dict
    """

    if fp1 is None or fp2 is None:
        return 0.0, {}

    fp1 = np.frombuffer(fp1, dtype=np.int8)
    fp2 = np.frombuffer(fp2, dtype=np.int8)

    if len(fp1) != len(fp2):
        min_len = min(len(fp1), len(fp2))
        fp1 = fp1[:min_len]
        fp2 = fp2[:min_len]

    # Exact match shortcut
    if np.array_equal(fp1, fp2):
        return 1.0, {
            "motif_match": True,
            "matched_beats": int(fp1.sum())
        }

    # Cross-correlation (motif alignment)
    corr = correlate(fp1, fp2, mode="valid")
    max_corr = np.max(corr)
    possible_max = max(fp1.sum(), fp2.sum(), 1)

    similarity = max_corr / possible_max

    evidence = {
        "motif_match": similarity > 0.6,
        "matched_beats": int(max_corr),
        "total_beats": int(possible_max)
    }

    return float(np.clip(similarity, 0.0, 1.0)), evidence
