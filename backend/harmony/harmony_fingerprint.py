#backend/harmony/harmonic_fingerprint.py

import numpy as np

import librosa

def extract_harmony_fingerprint(y, sr, hop_length=512):
    """
    Harmonic progression fingerprint (SAFE)
    Returns discrete pitch-class sequence
    """

    chroma = librosa.feature.chroma_cqt(
        y=y, sr=sr, hop_length=hop_length
    )

    # Dominant pitch class per frame
    fingerprint = np.argmax(chroma, axis=0)

    return fingerprint.astype(np.int8)
