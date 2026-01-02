#backend/rhythm/common_rythm.py

import numpy as np
import librosa

def extract_common_rhythm(y, sr):
    """
    Extract weak, non-protectable rhythm features.
    GUARANTEED to return a (6,) float32 vector.
    """

    # Onset envelope
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)

    # Beat tracking
    tempo, beats = librosa.beat.beat_track(
        onset_envelope=onset_env,
        sr=sr
    )

    # Duration
    duration = float(len(y)) / float(sr)

    # Beat interval stats (SAFE)
    if beats is None or len(beats) < 2:
        mean_ibi = 0.0
        std_ibi = 0.0
        beat_density = 0.0
    else:
        beat_intervals = np.diff(beats).astype(np.float32)
        mean_ibi = float(np.mean(beat_intervals))
        std_ibi = float(np.std(beat_intervals))
        beat_density = float(len(beats) / (duration + 1e-6))

    # Onset stats (SAFE)
    mean_onset = float(np.mean(onset_env)) if onset_env.size > 0 else 0.0
    std_onset = float(np.std(onset_env)) if onset_env.size > 0 else 0.0

    # Tempo normalization (SAFE)
    tempo_norm = float(tempo) / 300.0 if tempo > 0 else 0.0

    feats = np.array([
        tempo_norm,
        mean_ibi,
        std_ibi,
        beat_density,
        mean_onset,
        std_onset
    ], dtype=np.float32)

    # Normalize (optional but stable)
    norm = np.linalg.norm(feats)
    if norm > 0:
        feats = feats / norm

    return feats
